"""Expanded handwritten character recognizer and prediction arbitration.

The original app began as MNIST digit recognition. This module extends it to
letters and punctuation by combining a curated character model, the combined
MNIST+EMNIST alphanumeric model, an alphabet-only model, digit fallback logic,
and shape-based punctuation post-processing.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, WeightedRandomSampler
from torchvision import transforms

from alnum_model import EMNIST_MEAN as ALNUM_MEAN
from alnum_model import EMNIST_STD as ALNUM_STD
from alnum_model import WEIGHTS_PATH as ALNUM_WEIGHTS_PATH
from alnum_model import load_alnum_model
from emnist_experiment import EMNIST_MEAN, EMNIST_STD, WEIGHTS_PATH as EMNIST_WEIGHTS_PATH
from emnist_experiment import EmnistCNN, TinyEmnistCNN, WideEmnistCNN
from mnist_model import WEIGHTS_PATH as DIGIT_WEIGHTS_PATH
from mnist_model import DigitRegion, _foreground_from_image, _predict_digit_image, get_device, load_model, segment_digit_regions


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_DIR / "data" / "unipen_chars" / "curated"
WEIGHTS_PATH = PROJECT_DIR / "character_cnn.pt"
METRICS_PATH = PROJECT_DIR / "character_training_metrics.json"
LABELS_PATH = PROJECT_DIR / "character_labels.json"
EXEMPLARS_PATH = PROJECT_DIR / "character_exemplars.pt"

IMAGE_SIZE = 32
CACHE_PATH = PROJECT_DIR / "data" / "unipen_chars" / f"character_cache_segmented_{IMAGE_SIZE}.pt"
CHAR_MEAN = 0.173
CHAR_STD = 0.331
LETTER_MODEL_TYPES = {
    "cnn": EmnistCNN,
    "tinycnn": TinyEmnistCNN,
    "widecnn": WideEmnistCNN,
}
# The digit fallback model (see _load_digit_model_for_fallback) is loaded
# lazily and shared process-wide since it's only needed occasionally; the
# lock protects the check-then-load against a race if two request threads
# hit the first low-confidence digit-like character at the same time.
_DIGIT_MODEL: nn.Module | None = None
_DIGIT_MODEL_LOCK = threading.Lock()
_DIGIT_AMBIGUOUS_LABELS = {"B", "I", "J", "L", "O", "S", "Y", "Z", "l", "o"}


@dataclass(frozen=True)
class CharacterEpochMetrics:
    """Metrics captured during curated character-model training."""

    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    seconds: float


class CharacterDataset(Dataset):
    """Dataset wrapper around curated UNIPEN-style character image folders."""

    def __init__(self, root: Path, transform=None) -> None:
        self.root = root
        self.transform = transform
        class_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
        self.classes = [path.name for path in sorted(class_dirs, key=lambda item: int(item.name))]
        self.labels = [chr(int(item)) for item in self.classes]
        self.samples: list[tuple[Path, int]] = []
        for index, class_name in enumerate(self.classes):
            for image_path in sorted((root / class_name).glob("*.png")):
                self.samples.append((image_path, index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CharacterCNN(nn.Module):
    """Small MLP-style classifier for curated character glyphs.

    Despite the "CNN" name (kept for interface consistency/checkpoint
    compatibility with other models), this is a plain fully-connected
    network with no convolutions — the curated dataset is small enough that
    a simple MLP was sufficient and faster to iterate on.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CharacterConvCNN(nn.Module):
    """Convolutional classifier for the 93-class curated character dataset."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.12),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(96 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


CHARACTER_MODEL_TYPES = {
    "mlp": CharacterCNN,
    "cnn": CharacterConvCNN,
}


def train_transform() -> transforms.Compose:
    """Return the training transform for curated character images."""

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((CHAR_MEAN,), (CHAR_STD,)),
        ]
    )


def eval_transform() -> transforms.Compose:
    """Return the deterministic evaluation transform."""

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((CHAR_MEAN,), (CHAR_STD,)),
        ]
    )


def normalize_character_image(image: Image.Image) -> Image.Image:
    """Center and scale one character crop into a 28x28 foreground image.

    Re-runs segmentation on the already-cropped character (with mark merging
    enabled) purely to rebuild a clean foreground mask that includes
    disconnected parts like dots/crossbars while excluding stray background
    noise pixels that might sit just outside the actual glyph strokes. If no
    regions are found (blank/degenerate crop), falls back to a plain
    grayscale conversion of the whole crop.
    """

    regions = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)
    if regions:
        x0 = min(region.box[0] for region in regions)
        y0 = min(region.box[1] for region in regions)
        x1 = max(region.box[2] for region in regions)
        y1 = max(region.box[3] for region in regions)
        foreground = np.zeros((image.height, image.width), dtype=np.float32)
        for region in regions:
            rx0, ry0, rx1, ry1 = region.box
            foreground[ry0:ry1, rx0:rx1] = np.maximum(
                foreground[ry0:ry1, rx0:rx1],
                np.asarray(region.image, dtype=np.float32) / 255.0,
            )
        array = foreground[y0:y1, x0:x1]
    else:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    ys, xs = np.where(array > 0.18)
    if len(xs) > 0:
        array = array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = array.shape
    scale = 28.0 / max(height, width)
    new_width = max(1, min(IMAGE_SIZE, int(round(width * scale))))
    new_height = max(1, min(IMAGE_SIZE, int(round(height * scale))))
    glyph = Image.fromarray((array * 255).astype(np.uint8), mode="L").resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    canvas.paste(glyph, ((IMAGE_SIZE - new_width) // 2, (IMAGE_SIZE - new_height) // 2))
    return canvas


def character_tensor_from_image(image: Image.Image) -> torch.Tensor:
    """Convert a character image into the 32x32 curated-model tensor format."""

    normalized = normalize_character_image(image)
    array = np.asarray(normalized, dtype=np.float32) / 255.0
    array = (array - CHAR_MEAN) / CHAR_STD
    return torch.from_numpy(array).unsqueeze(0)


def split_dataset(dataset: CharacterDataset, validation_size: float = 0.15) -> tuple[list[int], list[int]]:
    """Create a stratified train/validation split for curated characters."""

    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=42,
        stratify=labels,
    )
    return train_indices, validation_indices


def build_or_load_cache(root: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Cache curated character tensors to avoid repeated PNG decoding."""

    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH, weights_only=True)
        return cache["images"], cache["targets"], list(cache["labels"])

    images = []
    targets = []
    dataset = CharacterDataset(root)
    for image_path, target in dataset.samples:
        image = Image.open(image_path).convert("L")
        images.append(character_tensor_from_image(image))
        targets.append(target)
    image_tensor = torch.stack(images)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": image_tensor, "targets": target_tensor, "labels": dataset.labels}, CACHE_PATH)
    return image_tensor, target_tensor, dataset.labels


def build_character_exemplars(
    root: Path = DATASET_ROOT,
    exemplars_per_class: int = 80,
    output_path: Path = EXEMPLARS_PATH,
) -> None:
    """Save one exemplar tensor per curated sample for nearest-neighbor fallback.

    Stores at most 80 examples per class (not the whole dataset) so the
    nearest-neighbor search in `_nearest_exemplar_label` stays cheap enough
    to run on every low-confidence prediction. Saved as float16 to keep the
    checkpoint file small since exemplar precision doesn't need to match
    training precision.
    """

    images, targets, labels = build_or_load_cache(root)
    selected_indices: list[int] = []
    for label_index in range(len(labels)):
        indices = torch.where(targets == label_index)[0][:exemplars_per_class]
        selected_indices.extend(int(index) for index in indices)
    selected = torch.tensor(selected_indices, dtype=torch.long)
    torch.save(
        {
            "images": images.index_select(0, selected).to(torch.float16),
            "targets": targets.index_select(0, selected).to(torch.long),
            "labels": labels,
            "image_size": IMAGE_SIZE,
            "normalization": {"mean": CHAR_MEAN, "std": CHAR_STD},
            "exemplars_per_class": exemplars_per_class,
        },
        output_path,
    )


def make_loaders(root: Path, batch_size: int) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build weighted train and validation loaders for curated characters."""

    images, targets, labels = build_or_load_cache(root)
    indices = list(range(len(targets)))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.15,
        random_state=42,
        stratify=targets.numpy(),
    )

    dataset = TensorDataset(images, targets)
    train_labels = [int(targets[index]) for index in train_indices]
    counts = np.bincount(train_labels, minlength=len(labels))
    weights = [1.0 / max(counts[label], 1) for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_indices), replacement=True)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, validation_loader, labels


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Evaluate the curated character classifier."""

    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, labels).item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return loss_total / max(len(loader), 1), 100.0 * correct / max(total, 1)


def train_character_model(
    epochs: int = 12,
    batch_size: int = 128,
    min_accuracy: float = 75.0,
    dataset_root: Path = DATASET_ROOT,
    model_type: str = "cnn",
    device_name: str = "auto",
    learning_rate: float = 0.001,
    label_smoothing: float = 0.03,
    seed: int = 42,
    warm_start: bool = False,
) -> list[CharacterEpochMetrics]:
    """Train the curated character model and save weights/labels/exemplars."""

    if not dataset_root.exists():
        raise RuntimeError(f"Missing dataset at {dataset_root}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device_name == "cpu":
        device = torch.device("cpu")
    elif device_name == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        device = torch.device("mps")
    else:
        device = get_device()
    train_loader, validation_loader, labels = make_loaders(dataset_root, batch_size)
    model_class = CHARACTER_MODEL_TYPES[model_type]
    model = model_class(num_classes=len(labels)).to(device)
    if warm_start and WEIGHTS_PATH.exists():
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        if checkpoint.get("model_type", "mlp") == model_type and list(checkpoint.get("labels", [])) == labels:
            model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[CharacterEpochMetrics] = []
    best_accuracy = 0.0
    best_state = None
    if warm_start:
        _, best_accuracy = evaluate(model, validation_loader, criterion, device)
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for images, labels_tensor in train_loader:
            images = images.to(device)
            labels_tensor = labels_tensor.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels_tensor).sum().item()
            train_total += labels_tensor.size(0)

        scheduler.step()
        train_loss = train_loss_total / max(len(train_loader), 1)
        train_accuracy = 100.0 * train_correct / max(train_total, 1)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        metrics = CharacterEpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            seconds=time.time() - start,
        )
        history.append(metrics)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch}/{epochs} "
            f"train_acc={train_accuracy:.2f}% validation_acc={validation_accuracy:.2f}%"
        )

    if best_state is None or best_accuracy < min_accuracy:
        raise RuntimeError(f"Best validation accuracy {best_accuracy:.2f}% is below {min_accuracy:.2f}%")

    torch.save(
        {
            "model_state_dict": best_state,
            "labels": labels,
            "validation_accuracy": best_accuracy,
            "model_type": model_type,
            "learning_rate": learning_rate,
            "label_smoothing": label_smoothing,
            "seed": seed,
            "warm_start": warm_start,
            "image_size": IMAGE_SIZE,
            "normalization": {"mean": CHAR_MEAN, "std": CHAR_STD},
        },
        WEIGHTS_PATH,
    )
    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    build_character_exemplars(dataset_root)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "labels": labels,
                "model_type": model_type,
                "learning_rate": learning_rate,
                "label_smoothing": label_smoothing,
                "seed": seed,
                "device": str(device),
                "warm_start": warm_start,
                "best_checkpoint": {"validation_accuracy": best_accuracy},
                "history": [asdict(item) for item in history],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return history


def load_character_model(weights_path: Path = WEIGHTS_PATH, device: torch.device | None = None) -> tuple[nn.Module, list[str]]:
    """Load the curated character classifier and optional exemplar bank."""

    selected_device = device or get_device()
    checkpoint = torch.load(weights_path, map_location=selected_device, weights_only=True)
    labels = list(checkpoint["labels"])
    model_type = str(checkpoint.get("model_type", "mlp"))
    model_class = CHARACTER_MODEL_TYPES.get(model_type, CharacterCNN)
    model = model_class(num_classes=len(labels)).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if EXEMPLARS_PATH.exists():
        exemplars = torch.load(EXEMPLARS_PATH, map_location="cpu", weights_only=True)
        if list(exemplars.get("labels", [])) == labels:
            model.character_exemplars = exemplars["images"].float()
            model.character_exemplar_targets = exemplars["targets"].long()
    model.eval()
    return model, labels


def load_letter_model(
    weights_path: Path = EMNIST_WEIGHTS_PATH,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[str]] | tuple[None, None]:
    """Load the alphabet-only EMNIST model when available."""

    if not weights_path.exists():
        return None, None
    selected_device = device or get_device()
    checkpoint = torch.load(weights_path, map_location=selected_device, weights_only=True)
    labels = [str(label).upper() for label in checkpoint["labels"]]
    model_type = str(checkpoint.get("model_type", "cnn"))
    model_class = LETTER_MODEL_TYPES.get(model_type, EmnistCNN)
    model = model_class(num_classes=len(labels)).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, labels


def tensor_from_character_region(region: DigitRegion, device: torch.device) -> torch.Tensor:
    """Convert a segmented region for the curated character model."""

    return character_tensor_from_image(region.image).unsqueeze(0).to(device)


def tensor_from_letter_region(region: DigitRegion, device: torch.device) -> torch.Tensor:
    """Convert a segmented region for the alphabet-only EMNIST model."""

    normalized = normalize_character_image(region.image).resize((28, 28), Image.Resampling.LANCZOS)
    array = np.asarray(normalized, dtype=np.float32) / 255.0
    array = (array - EMNIST_MEAN) / EMNIST_STD
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def tensor_from_alnum_region(region: DigitRegion, device: torch.device) -> torch.Tensor:
    """Convert a segmented region for the combined alphanumeric model."""

    normalized = normalize_character_image(region.image).resize((28, 28), Image.Resampling.LANCZOS)
    array = np.asarray(normalized, dtype=np.float32) / 255.0
    array = (array - ALNUM_MEAN) / ALNUM_STD
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def _nearest_exemplar_label(model: nn.Module, tensor: torch.Tensor) -> tuple[int, float] | None:
    """Find the closest curated exemplar for low-confidence predictions.

    A simple k=1 nearest-neighbor lookup (by raw pixel-space Euclidean
    distance) over a small cached sample of training images per class — see
    `build_character_exemplars`. This acts as a sanity-check fallback for
    when the softmax classifier itself is unsure, since a very close visual
    match to a real training example is often a better signal than a
    marginal softmax probability.
    """

    exemplars = getattr(model, "character_exemplars", None)
    targets = getattr(model, "character_exemplar_targets", None)
    if exemplars is None or targets is None:
        return None
    vector = tensor.detach().cpu().flatten().float().unsqueeze(0)
    distances = torch.cdist(vector, exemplars.reshape(exemplars.size(0), -1)).squeeze(0)
    distance, exemplar_index = torch.min(distances, dim=0)
    return int(targets[int(exemplar_index)].item()), float(distance.item())


def _looks_like_letter_region(region: DigitRegion, label: str, confidence: float) -> bool:
    """Decide whether a region is worth sending to the alphabet model.

    The alphabet-only model is slower to invoke and can wrongly steal
    confident digit predictions (e.g. relabeling a clean "4" as a letter),
    so it's only consulted when there's a real chance the curated model got
    it wrong: the curated label is already alphabetic, it's one of the
    digit/letter shapes that are visually ambiguous (1/I/l, -/_/|), or the
    curated model itself was unsure and the region's shape and size are
    consistent with a normal letter glyph.
    """

    x0, y0, x1, y1 = region.box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    aspect_ratio = width / height
    if label.isalpha():
        return True
    if label in {"-", "_", "|", "1", "I", "l"} and aspect_ratio >= 0.28:
        return True
    return confidence < 0.82 and aspect_ratio >= 0.32 and width * height >= 650


def _letter_prediction(
    letter_model: nn.Module,
    letter_labels: list[str],
    region: DigitRegion,
    device: torch.device,
) -> tuple[str, float]:
    """Predict one region with the alphabet-only model."""

    tensor = tensor_from_letter_region(region, device)
    probabilities = torch.softmax(letter_model(tensor), dim=1).squeeze(0)
    confidence, label_index = torch.max(probabilities, dim=0)
    return letter_labels[int(label_index.item())], float(confidence.item())


def _alnum_prediction(
    alnum_model: nn.Module,
    alnum_labels: list[str],
    region: DigitRegion,
    device: torch.device,
) -> tuple[str, float, list[dict[str, float | str]]]:
    """Predict one region with the combined digit+letter model."""

    tensor = tensor_from_alnum_region(region, device)
    probabilities = torch.softmax(alnum_model(tensor), dim=1).squeeze(0)
    confidence, label_index = torch.max(probabilities, dim=0)
    top_count = min(5, len(alnum_labels))
    top_values, top_indexes = torch.topk(probabilities, top_count)
    alternatives = [
        {"label": alnum_labels[int(index.item())], "confidence": float(value.item())}
        for value, index in zip(top_values, top_indexes)
    ]
    return alnum_labels[int(label_index.item())], float(confidence.item()), alternatives


def _same_letter_different_case(first: str, second: str) -> bool:
    """Return true when two one-character labels differ only by case."""

    return len(first) == 1 and len(second) == 1 and first.isalpha() and second.isalpha() and first.lower() == second.lower() and first != second


def _case_ambiguity_alternatives(alternatives: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """Return same-letter upper/lower alternatives when both are plausible.

    Some letters (e.g. C/c, O/o, S/s, W/w) look nearly identical in isolated
    single-character handwriting with no surrounding word context to hint at
    case. When the model's top-N alternatives contain both cases of the same
    letter and neither is a clear loser (both >= 0.20 confidence), this
    surfaces the pair so the UI/consumer can show both plausible readings
    instead of silently picking one.
    """

    for first in alternatives:
        first_label = str(first["label"])
        if not first_label.isalpha():
            continue
        for second in alternatives:
            second_label = str(second["label"])
            if not _same_letter_different_case(first_label, second_label):
                continue
            if min(float(first["confidence"]), float(second["confidence"])) >= 0.20:
                pair = [first, second]
                return sorted(pair, key=lambda item: float(item["confidence"]), reverse=True)
    return []


def _top_alternatives(alternatives: list[dict[str, float | str]], selected_label: str) -> list[dict[str, float | str]]:
    """Return compact top guesses, always keeping the selected label visible.

    Caps the list at 3 unique labels for display purposes. If the label that
    ultimately won (after all the override logic in `predict_characters`)
    isn't among the top alnum-model alternatives, it's force-inserted at the
    front with a 0.0 placeholder confidence — otherwise the UI could show
    "top guesses" that don't even include the actual displayed answer, which
    would look like a bug.
    """

    cleaned: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for item in alternatives:
        label = str(item["label"])
        if label in seen:
            continue
        cleaned.append({"label": label, "confidence": float(item["confidence"])})
        seen.add(label)
        if len(cleaned) >= 3:
            break
    if selected_label not in seen:
        cleaned.insert(0, {"label": selected_label, "confidence": 0.0})
        cleaned = cleaned[:3]
    return cleaned


def _is_uncertain(confidence: float, alternatives: list[dict[str, float | str]]) -> bool:
    """Return true when a prediction should be shown as needing review.

    Flags low absolute confidence (<0.75), but also flags a "close call"
    even when top confidence is otherwise fine: if the top two alternatives
    are within 0.15 of each other, the model is essentially torn between two
    answers and the result deserves a second look.
    """

    if confidence < 0.75:
        return True
    if len(alternatives) >= 2:
        gap = float(alternatives[0]["confidence"]) - float(alternatives[1]["confidence"])
        return gap < 0.15
    return False


def _alnum_should_override(
    current_label: str,
    current_confidence: float,
    alnum_label: str,
    alnum_confidence: float,
    letter_confidence: float,
) -> bool:
    """Decide when the trained alphanumeric model may replace the current label.

    This is the central arbitration point between the older curated
    character model and the newer, generally more accurate combined
    alphanumeric model. The rules are intentionally asymmetric and
    conservative rather than "highest confidence wins":
    - Below 0.55 confidence the alnum model is too unsure to trust at all.
    - If the alnum model says "digit" but the dedicated letter model is very
      confident (>=0.9) it's a letter, don't let a shaky digit guess win.
    - For upper/lower case flips of the same letter, require a bigger margin
      (0.18 for flipping to lowercase, 0.10 for uppercase) since case errors
      are visually subtle and worth being more skeptical about.
    - If the curated model was already unsure (<0.86) or produced something
      that isn't even alphanumeric, defer to the alnum model outright.
    - Otherwise, only override within the same character class (digit vs.
      digit, letter vs. letter) and only if the alnum model isn't
      meaningfully worse (allowing a small 0.05 confidence handicap).
    """

    if alnum_confidence < 0.55:
        return False
    if alnum_label.isdigit() and letter_confidence >= 0.9:
        return False
    if _same_letter_different_case(current_label, alnum_label):
        margin = 0.18 if current_label.islower() else 0.10
        return alnum_confidence >= current_confidence + margin
    if current_confidence < 0.86 or not current_label.isalnum():
        return True
    if current_label.isdigit() and alnum_label.isdigit():
        return alnum_confidence >= current_confidence - 0.05
    if current_label.isalpha() and alnum_label.isalpha():
        return alnum_confidence >= current_confidence - 0.05
    return False


def _load_digit_model_for_fallback(device: torch.device) -> nn.Module | None:
    """Lazily load the digit CNN for ambiguous numeric-looking regions.

    Loaded on first use (not at import time) since most predictions never
    need it — only regions that look digit-like but got a shaky answer from
    the other models trigger this path. See `_DIGIT_MODEL_LOCK` for the
    thread-safety rationale.
    """

    global _DIGIT_MODEL
    if _DIGIT_MODEL is not None:
        return _DIGIT_MODEL
    with _DIGIT_MODEL_LOCK:
        if _DIGIT_MODEL is not None:
            return _DIGIT_MODEL
        if not DIGIT_WEIGHTS_PATH.exists():
            return None
        _DIGIT_MODEL = load_model(device=device)
        return _DIGIT_MODEL


def _digit_fallback_prediction(region: DigitRegion, device: torch.device) -> tuple[str, float] | None:
    """Ask the MNIST digit model to rescue uncertain digit-like regions.

    The size floor (width >= 20, height >= 34, area >= 900) filters out
    small punctuation-like marks before bothering to run the digit model,
    since MNIST was trained on reasonably large centered digits and would
    give meaningless answers on tiny fragments.
    """

    x0, y0, x1, y1 = region.box
    width = x1 - x0
    height = y1 - y0
    if width < 20 or height < 34 or width * height < 900:
        return None
    digit_model = _load_digit_model_for_fallback(device)
    if digit_model is None:
        return None
    digit, confidence = _predict_digit_image(digit_model, region.image, device)
    return str(digit), confidence


def _digit_beats_ambiguous_letter(
    digit_label: str,
    digit_confidence: float,
    current_label: str,
    current_confidence: float,
) -> bool:
    """Return true for known digit/letter pairs where MNIST should win.

    Hand-picked exceptions for shape pairs that are genuinely ambiguous even
    to a human (2/Z, 4/Y, 5/J depending on handwriting style), where
    experience showed the specialist MNIST digit model outperforms the
    general letter/alnum models. Thresholds are set very high (0.94-0.985)
    so this only kicks in when MNIST is nearly certain, and only overrides
    when the letter guess itself wasn't already highly confident.
    """

    pair_thresholds = {
        ("2", "Z"): (0.985, 0.97),
        ("4", "Y"): (0.950, 0.995),
        ("5", "J"): (0.950, 0.995),
        ("1", "I"): (0.995, 0.98),
        ("1", "L"): (0.995, 0.98),
        ("1", "l"): (0.995, 0.98),
        ("0", "O"): (0.995, 0.97),
        ("0", "o"): (0.995, 0.97),
        ("8", "B"): (0.995, 0.97),
        ("5", "S"): (0.995, 0.97),
    }
    thresholds = pair_thresholds.get((digit_label, current_label))
    if thresholds is None:
        return False
    digit_floor, letter_ceiling = thresholds
    return digit_confidence >= digit_floor and current_confidence < letter_ceiling


def _letter_should_override(
    current_label: str,
    current_confidence: float,
    letter_confidence: float,
    digit_was_used: bool,
) -> bool:
    """Decide when the letter-only model is strong enough to replace a label.

    If the digit fallback already claimed this region, the letter model is
    never allowed to override it — digit fallback only fires when it beat
    strong evidence already, so it shouldn't be second-guessed here. When
    the current label isn't alphanumeric at all (e.g. punctuation), a modest
    0.55 confidence is enough since there's little to lose. When it's
    already a letter, only a same-or-better (within 0.03) letter guess wins.
    When it's a digit, the bar is much higher (0.92, and must beat the
    current confidence by 0.08) since replacing a digit with a letter is a
    bigger, riskier change.
    """

    if digit_was_used:
        return False
    if not current_label.isalnum():
        return letter_confidence >= 0.55
    if current_label.isalpha():
        return letter_confidence >= 0.70 and letter_confidence >= current_confidence - 0.03
    return letter_confidence >= 0.92 and letter_confidence >= current_confidence + 0.08


def _mask_span(mask: np.ndarray) -> float:
    """Measure how much horizontal space a foreground mask occupies.

    Returns the ink's horizontal extent as a fraction of the mask's width —
    used by the shape heuristics below (e.g. `_looks_like_seven`) as a cheap
    stand-in for "how wide is the stroke in this horizontal band" without
    needing full contour analysis.
    """

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    return float(xs.max() - xs.min() + 1) / max(float(mask.shape[1]), 1.0)


def _looks_like_seven(region: DigitRegion) -> bool:
    """Recognize the tall handwritten 7 shape that models confuse with 1.

    A "7" has a wide horizontal stroke across the top and a narrow single
    stroke below, whereas a "1" is narrow all the way down. This checks the
    top 25% of the box for wide ink coverage and the bottom 45% for narrow
    coverage, combined with an aspect ratio in the range typical of a
    handwritten digit (not so narrow it's clearly a "1", not so wide it's
    something else). These geometric checks run *after* the neural models
    specifically because 1/7 confusion is common and hard for the CNNs to
    fix on their own without seeing the distinctive top bar.
    """

    mask = _foreground_from_image(region.image) > 0.18
    height, width = mask.shape
    if height <= 0 or width <= 0:
        return False
    top_span = _mask_span(mask[: max(1, int(height * 0.25)), :])
    bottom_span = _mask_span(mask[int(height * 0.55) :, :])
    aspect_ratio = width / max(height, 1)
    return 0.24 <= aspect_ratio <= 0.62 and top_span >= 0.48 and bottom_span <= 0.22


def _looks_like_one(region: DigitRegion) -> bool:
    """Recognize a plain vertical 1 that models confuse with L.

    The mirror image of `_looks_like_seven`'s logic: requires a narrow
    aspect ratio and narrow ink coverage at both the top and bottom of the
    box (no wide top bar, no wide base), which is what plain vertical
    strokes look like but a serif'd "L" or flared "1" would not.
    """

    mask = _foreground_from_image(region.image) > 0.18
    height, width = mask.shape
    if height <= 0 or width <= 0:
        return False
    top_span = _mask_span(mask[: max(1, int(height * 0.25)), :])
    bottom_span = _mask_span(mask[int(height * 0.55) :, :])
    aspect_ratio = width / max(height, 1)
    return aspect_ratio <= 0.34 and top_span <= 0.22 and bottom_span <= 0.24


def _looks_like_four(region: DigitRegion) -> bool:
    """Recognize open-top handwritten 4s that letter models confuse with A.

    A handwritten "4" typically has: a wide horizontal crossbar around the
    middle (`middle_span`), solid ink running down the right side (the
    vertical stroke, `right_vertical_coverage`), very little ink in the
    lower-left quadrant (the open triangular gap unique to "4", as opposed
    to "A" which closes at the bottom), and at least a little ink in the
    upper-left (the diagonal entering the crossbar). All four conditions
    must hold together since any one alone could also describe other glyphs.
    """

    mask = _foreground_from_image(region.image) > 0.18
    height, width = mask.shape
    if height <= 0 or width <= 0:
        return False
    aspect_ratio = width / max(height, 1)
    if not 0.35 <= aspect_ratio <= 0.85:
        return False

    middle = mask[int(height * 0.42) : max(int(height * 0.62), int(height * 0.42) + 1), :]
    right = mask[:, int(width * 0.58) :]
    lower_left = mask[int(height * 0.62) :, : max(1, int(width * 0.38))]
    top_left = mask[: max(1, int(height * 0.45)), : max(1, int(width * 0.45))]
    middle_span = _mask_span(middle)
    right_vertical_coverage = float(np.count_nonzero(np.any(right, axis=1))) / max(height, 1)
    return (
        middle_span >= 0.45
        and right_vertical_coverage >= 0.55
        and float(lower_left.mean()) <= 0.10
        and float(top_left.mean()) >= 0.015
    )


def _ink_center_x(mask: np.ndarray) -> float | None:
    """Return the normalized horizontal center of a mask slice."""

    _, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()) / max(float(mask.shape[1]), 1.0)


def _parenthesis_shape_label(region: DigitRegion) -> str | None:
    """Recognize single-stroke parentheses before model arbitration.

    Parentheses are drawn as one thin curved stroke, which none of the
    trained classifiers see much of (they're mostly trained on letters and
    digits). Detected purely from geometry rather than any model: a narrow,
    sparse (low average ink density) box, split into five horizontal bands
    whose ink centroid traces a curve. If the middle band's centroid is
    shifted left of both the top and bottom bands' centroids, the stroke
    bulges left, i.e. a "(" ; shifted right (with a shallower top, since ")"
    tends to start further right) means a ")" .
    """

    mask = _foreground_from_image(region.image) > 0.18
    height, width = mask.shape
    if height <= 0 or width <= 0:
        return None
    aspect_ratio = width / max(height, 1)
    if not 0.32 <= aspect_ratio <= 0.72 or float(mask.mean()) > 0.08:
        return None

    top_span = _mask_span(mask[: max(1, int(height * 0.25)), :])
    middle_span = _mask_span(mask[int(height * 0.35) : max(int(height * 0.65), int(height * 0.35) + 1), :])
    bottom_span = _mask_span(mask[int(height * 0.75) :, :])
    if middle_span > 0.36 or bottom_span > 0.38:
        return None

    centers = []
    for start, stop in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)):
        section = mask[int(height * start) : max(int(height * stop), int(height * start) + 1), :]
        center = _ink_center_x(section)
        if center is None:
            return None
        centers.append(center)

    if centers[2] <= centers[0] - 0.15 and centers[2] <= centers[4] - 0.10:
        return "("
    if top_span <= 0.58 and centers[2] >= centers[0] + 0.15 and centers[2] >= centers[4] + 0.10:
        return ")"
    return None


def _punctuation_shape_label(region: DigitRegion) -> str | None:
    """Detect punctuation whose geometry is clearer than model logits.

    None of the classifiers were trained with much (or any) coverage of "i"
    dots, "!" marks, or ":" as their own standalone characters, so multi-part
    punctuation is recognized directly from connected-component geometry:
    one "main" (largest) component plus one or more small satellite
    components, disambiguated by whether the small piece sits above, below,
    or straddles the main stroke.
    """

    parenthesis_label = _parenthesis_shape_label(region)
    if parenthesis_label is not None:
        return parenthesis_label

    array = np.asarray(region.image, dtype=np.float32) / 255.0
    mask = array > 0.18
    if mask.mean() > 0.5:
        # This region image is already a foreground-normalized grayscale
        # crop (not raw photo data), so a mask that's more than half "on"
        # signals the polarity looks inverted for this particular crop;
        # re-threshold from the bright side instead of the dark side.
        mask = array < 0.82
    labeled, count = ndimage.label(mask)
    components: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count + 1):
        ys, xs = np.where(labeled == index)
        if len(xs) < 6:
            continue
        components.append((len(xs), int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    if len(components) < 2:
        return None

    components = sorted(components, reverse=True)
    main_area, main_x0, main_y0, main_x1, main_y1 = components[0]
    main_width = main_x1 - main_x0
    main_height = main_y1 - main_y0
    main_center_x = (main_x0 + main_x1) / 2.0
    small_components = components[1:]

    if main_height >= 28 and main_height / max(main_width, 1) >= 3.0:
        for _, x0, y0, x1, y1 in small_components:
            width = x1 - x0
            height = y1 - y0
            center_x = (x0 + x1) / 2.0
            if width <= 24 and height <= 24 and abs(center_x - main_center_x) <= max(10, main_width * 1.5):
                if y1 <= main_y0:
                    return "i"
                if y0 >= main_y1:
                    return "!"

    if len(components) == 2:
        first, second = sorted(components, key=lambda item: item[2])
        _, ax0, ay0, ax1, ay1 = first
        _, bx0, by0, bx1, by1 = second
        first_width = ax1 - ax0
        first_height = ay1 - ay0
        second_width = bx1 - bx0
        second_height = by1 - by0
        center_gap = abs(((ax0 + ax1) / 2.0) - ((bx0 + bx1) / 2.0))
        if (
            first_width <= 24
            and first_height <= 24
            and second_width <= 24
            and second_height <= 24
            and center_gap <= 10
            and by0 > ay1
        ):
            return ":"
    return None


# The functions below operate on final *predictions* (after each region has
# already been classified by the models), unlike `_punctuation_shape_label`
# above which works on raw segmentation geometry before classification.
# Segmentation sometimes keeps a stem and its dot as two separate regions
# even with mark-merging enabled (e.g. when they're far enough apart), so
# this second pass looks across the whole finished prediction list for
# adjacent stem+dot pairs to fuse into "i" or "!" after the fact.


def _is_i_stem(prediction: dict[str, float | int | str]) -> bool:
    """Return true when a prediction could be the stem of lowercase i."""

    label = str(prediction["label"])
    width = int(prediction["width"])
    height = int(prediction["height"])
    return label in {"1", "I", "L", "l", "|", "!"} and height >= 28 and height / max(width, 1) >= 2.35


def _is_detached_i_dot(prediction: dict[str, float | int | str]) -> bool:
    """Return true when a small prediction could be an i dot."""

    label = str(prediction["label"])
    width = int(prediction["width"])
    height = int(prediction["height"])
    return label in {":", ".", "'", "`", "!", "i"} and width <= 36 and height <= 36


def _is_detached_exclamation_dot(prediction: dict[str, float | int | str]) -> bool:
    """Return true when a small prediction could be an exclamation dot."""

    label = str(prediction["label"])
    width = int(prediction["width"])
    height = int(prediction["height"])
    return label in {":", ".", "'", "`", "!", "i", "0", "O", "Q"} and width <= 36 and height <= 36


def _is_detached_colon_dot(prediction: dict[str, float | int | str]) -> bool:
    """Return true when a small prediction could be one colon dot."""

    label = str(prediction["label"])
    width = int(prediction["width"])
    height = int(prediction["height"])
    return label in {":", ".", "'", "`", "Q", "O", "0"} and width <= 36 and height <= 36


def _is_dot_above_stem(
    dot: dict[str, float | int | str],
    stem: dict[str, float | int | str],
) -> bool:
    """Check whether a dot sits above and aligned with an i-like stem.

    Horizontal tolerance scales with the stem's own width (never below
    22px) since a wider stroke can plausibly have a slightly more offset
    dot. Vertical tolerance similarly scales with the stem's height (never
    below 58px) to accommodate both small and large handwriting.
    """

    dot_center_x = float(dot["x"]) + float(dot["width"]) / 2.0
    stem_center_x = float(stem["x"]) + float(stem["width"]) / 2.0
    stem_top = float(stem["y"])
    stem_width = float(stem["width"])
    stem_height = float(stem["height"])
    horizontal_slop = max(22.0, stem_width * 1.5)
    return (
        float(dot["y"]) + float(dot["height"]) <= stem_top + stem_height * 0.18
        and abs(dot_center_x - stem_center_x) <= horizontal_slop
        and stem_top - (float(dot["y"]) + float(dot["height"])) <= max(58.0, stem_height * 0.55)
    )


def _is_dot_below_stem(
    dot: dict[str, float | int | str],
    stem: dict[str, float | int | str],
) -> bool:
    """Check whether a dot sits below and aligned with an exclamation stem."""

    dot_center_x = float(dot["x"]) + float(dot["width"]) / 2.0
    stem_center_x = float(stem["x"]) + float(stem["width"]) / 2.0
    stem_bottom = float(stem["y"]) + float(stem["height"])
    stem_width = float(stem["width"])
    stem_height = float(stem["height"])
    horizontal_slop = max(22.0, stem_width * 1.5)
    vertical_gap = float(dot["y"]) - stem_bottom
    return 0 <= vertical_gap <= max(58.0, stem_height * 0.55) and abs(dot_center_x - stem_center_x) <= horizontal_slop


def _merge_bounds(
    first: dict[str, float | int | str],
    second: dict[str, float | int | str],
) -> tuple[int, int, int, int]:
    """Return the union bounding box for two predictions."""

    x0 = min(int(first["x"]), int(second["x"]))
    y0 = min(int(first["y"]), int(second["y"]))
    x1 = max(int(first["x"]) + int(first["width"]), int(second["x"]) + int(second["width"]))
    y1 = max(int(first["y"]) + int(first["height"]), int(second["y"]) + int(second["height"]))
    return x0, y0, x1, y1


def _postprocess_colons(predictions: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    """Merge two vertically stacked dot predictions into one colon."""

    merged_indexes: set[int] = set()
    replacements: dict[int, dict[str, float | int | str]] = {}

    for first_index, first in enumerate(predictions):
        if first_index in merged_indexes or not _is_detached_colon_dot(first):
            continue
        first_center_x = float(first["x"]) + float(first["width"]) / 2.0
        first_bottom = float(first["y"]) + float(first["height"])
        for second_index, second in enumerate(predictions[first_index + 1 :], start=first_index + 1):
            if second_index in merged_indexes or not _is_detached_colon_dot(second):
                continue
            if int(first["row"]) != int(second["row"]):
                continue
            second_center_x = float(second["x"]) + float(second["width"]) / 2.0
            vertical_gap = float(second["y"]) - first_bottom
            if abs(second_center_x - first_center_x) <= 14 and 8 <= vertical_gap <= 80:
                x0, y0, x1, y1 = _merge_bounds(first, second)
                merged_indexes.add(second_index)
                replacements[first_index] = {
                    **first,
                    "label": ":",
                    "confidence": max(float(first["confidence"]), float(second["confidence"]), 0.9),
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "row": min(int(first["row"]), int(second["row"])),
                }
                break

    merged: list[dict[str, float | int | str]] = []
    for index, prediction in enumerate(predictions):
        if index in merged_indexes:
            continue
        merged.append(replacements.get(index, prediction))
    return sorted(merged, key=lambda item: (int(item["row"]), int(item["x"])))


def _postprocess_lowercase_i(predictions: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    """Merge a detached dot and skinny stem into lowercase i."""

    used_dot_indexes: set[int] = set()
    replacements: dict[int, dict[str, float | int | str]] = {}

    for stem_index, prediction in enumerate(predictions):
        if not _is_i_stem(prediction):
            continue

        dot_index = next(
            (
                index
                for index, candidate in enumerate(predictions)
                if index != stem_index
                and index not in used_dot_indexes
                and int(candidate["row"]) == int(prediction["row"])
                and _is_detached_i_dot(candidate)
                and _is_dot_above_stem(candidate, prediction)
            ),
            None,
        )
        if dot_index is None:
            continue

        dot = predictions[dot_index]
        x0, y0, x1, y1 = _merge_bounds(prediction, dot)
        used_dot_indexes.add(dot_index)
        replacements[stem_index] = {
            **prediction,
            "label": "i",
            "confidence": max(float(prediction["confidence"]), float(dot["confidence"]), 0.9),
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
            "row": min(int(prediction["row"]), int(dot["row"])),
        }

    merged: list[dict[str, float | int | str]] = []
    for index, prediction in enumerate(predictions):
        if index in used_dot_indexes:
            continue
        merged.append(replacements.get(index, prediction))

    return sorted(merged, key=lambda item: (int(item["row"]), int(item["x"])))


def _postprocess_exclamations(predictions: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    """Merge a detached lower dot and skinny stem into exclamation mark."""

    used_dot_indexes: set[int] = set()
    replacements: dict[int, dict[str, float | int | str]] = {}

    for stem_index, prediction in enumerate(predictions):
        if not _is_i_stem(prediction):
            continue

        dot_index = next(
            (
                index
                for index, candidate in enumerate(predictions)
                if index != stem_index
                and index not in used_dot_indexes
                and int(candidate["row"]) == int(prediction["row"])
                and _is_detached_exclamation_dot(candidate)
                and _is_dot_below_stem(candidate, prediction)
            ),
            None,
        )
        if dot_index is None:
            continue

        dot = predictions[dot_index]
        x0, y0, x1, y1 = _merge_bounds(prediction, dot)
        used_dot_indexes.add(dot_index)
        replacements[stem_index] = {
            **prediction,
            "label": "!",
            "confidence": max(float(prediction["confidence"]), float(dot["confidence"]), 0.92),
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
            "row": min(int(prediction["row"]), int(dot["row"])),
        }

    merged: list[dict[str, float | int | str]] = []
    for index, prediction in enumerate(predictions):
        if index in used_dot_indexes:
            continue
        merged.append(replacements.get(index, prediction))
    return sorted(merged, key=lambda item: (int(item["row"]), int(item["x"])))


def _split_box(mask: np.ndarray, left: int, right: int) -> tuple[int, int, int, int] | None:
    """Return the foreground bounds inside a horizontal slice."""

    submask = mask[:, left:right]
    ys, xs = np.where(submask)
    if len(xs) < 20:
        return None
    return (left + int(xs.min()), int(ys.min()), left + int(xs.max()) + 1, int(ys.max()) + 1)


def _blank_seam_cut(smoothed_projection: np.ndarray) -> int | None:
    """Find a split column from a truly blank-ish internal seam.

    Looks for a run of near-zero-ink columns (<=8% of the projection's peak,
    with a 1.0 floor to tolerate single stray pixels) within the middle 30%
    to 72% of the region — i.e. away from the outer edges, where a genuine
    gap between two touching characters would fall. Requires the blank run
    to be at least 5 columns wide so a single thin anti-aliasing gap inside
    one character's stroke doesn't get mistaken for a real character
    boundary. The cut point is the run's average column, not its edge.
    """

    width = len(smoothed_projection)
    start = int(width * 0.30)
    stop = int(width * 0.72)
    low_columns = np.where(
        smoothed_projection[start:stop] <= max(1.0, float(smoothed_projection.max()) * 0.08)
    )[0] + start
    if len(low_columns) < 5:
        return None

    groups = np.split(low_columns, np.where(np.diff(low_columns) > 1)[0] + 1)
    group = max(groups, key=len)
    if len(group) < 5:
        return None
    return int(round(float(group.mean())))


def _thin_bridge_cut(mask: np.ndarray, smoothed_projection: np.ndarray) -> int | None:
    """Find a split column where two glyphs touch through a small bridge.

    Unlike `_blank_seam_cut`, this handles characters that touch via a thin
    connecting stroke rather than a clean gap (e.g. cursive-ish joins, or two
    digits whose strokes just barely overlap). Only attempted on wide boxes
    (aspect ratio >= 1.15, width >= 110) since a normal single character
    wouldn't be this wide. The column with the least ink in the middle
    search window is a candidate cut, but it's only accepted if: it's a real
    local valley relative to its neighboring peaks (not just uniformly low
    ink throughout), the bridge itself is thin (little ink in the few
    columns around the cut), and both resulting halves are wide enough (or
    tall+narrow enough to be a legitimate narrow mark like "1" or "l") to
    plausibly be their own characters — this avoids slicing off a sliver
    that's too thin to be meaningful.
    """

    height, width = mask.shape
    if width < 110 or width / max(height, 1) < 1.15:
        return None

    start = int(width * 0.28)
    stop = int(width * 0.74)
    if stop <= start:
        return None

    window = smoothed_projection[start:stop]
    cut = start + int(np.argmin(window))
    left_peak = float(smoothed_projection[max(0, cut - 28) : max(0, cut - 4)].max(initial=0.0))
    right_peak = float(smoothed_projection[min(width, cut + 5) : min(width, cut + 29)].max(initial=0.0))
    neighbor_peak = min(left_peak, right_peak)
    valley = float(smoothed_projection[cut])
    if neighbor_peak <= 0.0 or valley > max(2.5, neighbor_peak * 0.34):
        return None

    bridge_ink = int(mask[:, max(0, cut - 2) : min(width, cut + 3)].sum())
    if bridge_ink > max(10, int(height * 0.22)):
        return None

    left_box = _split_box(mask, 0, cut)
    right_box = _split_box(mask, cut, width)
    if left_box is None or right_box is None:
        return None
    left_width = left_box[2] - left_box[0]
    right_width = right_box[2] - right_box[0]
    left_height = left_box[3] - left_box[1]
    right_height = right_box[3] - right_box[1]
    left_is_narrow_mark = left_width >= 8 and left_height >= height * 0.45
    right_is_narrow_mark = right_width >= 8 and right_height >= height * 0.45
    if (left_width < 24 and not left_is_narrow_mark) or (right_width < 24 and not right_is_narrow_mark):
        return None
    return cut


def _split_one_touching_character_region(region: DigitRegion) -> list[DigitRegion]:
    """Split one wide region when it has a safe vertical cut."""

    mask = _foreground_from_image(region.image) > 0.18
    height, width = mask.shape
    if height <= 0 or width <= 0 or width / max(height, 1) < 0.85 or width < 48:
        return [region]

    projection = mask.sum(axis=0).astype(np.float32)
    smoothed = np.convolve(projection, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    cut = _blank_seam_cut(smoothed)
    if cut is None:
        cut = _thin_bridge_cut(mask, smoothed)
    if cut is None:
        return [region]

    local_boxes = [_split_box(mask, 0, cut), _split_box(mask, cut, width)]
    if any(box is None for box in local_boxes):
        return [region]

    x0, y0, _, _ = region.box
    split_regions: list[DigitRegion] = []
    for lx0, ly0, lx1, ly1 in local_boxes:
        split_regions.append(
            DigitRegion(
                image=region.image.crop((lx0, ly0, lx1, ly1)),
                box=(x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1),
                row=region.row,
            )
        )
    return split_regions


def _split_touching_character_regions(regions: list[DigitRegion]) -> list[DigitRegion]:
    """Split wide regions that contain either a seam or a thin touching bridge.

    Runs breadth-first with a depth cap of 2 so a region can be split at
    most twice (into up to ~4 pieces): after each split, both resulting
    pieces are re-queued for another split attempt, since a wide region can
    occasionally contain more than two touching characters. The depth cap
    exists purely as a safety net against runaway recursive splitting on
    unusual input.
    """

    split_regions: list[DigitRegion] = []
    pending: list[tuple[DigitRegion, int]] = [(region, 0) for region in regions]
    while pending:
        region, depth = pending.pop(0)
        pieces = _split_one_touching_character_region(region)
        if len(pieces) == 1 or depth >= 2:
            split_regions.extend(pieces)
            continue
        pending.extend((piece, depth + 1) for piece in pieces)

    return sorted(split_regions, key=lambda item: (item.row, item.box[0]))


def predict_characters(
    model: nn.Module,
    labels: list[str],
    image: Image.Image,
    device: torch.device | None = None,
    letter_model: nn.Module | None = None,
    letter_labels: list[str] | None = None,
    alnum_model: nn.Module | None = None,
    alnum_labels: list[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Segment an image and predict letters, digits, and punctuation.

    This is the top-level prediction pipeline that layers several signals on
    top of each other, in order: the curated character model's raw softmax
    output, then a punctuation shape override, then a nearest-exemplar
    fallback, then (optionally) the alphabet-only and combined alphanumeric
    models, then a narrow MNIST digit-model rescue, then geometric
    shape-based overrides for specific known confusions (1/7, 4/A). Once
    every region has an individual label, a final pass merges detached dots
    into "i"/"!"/":" across the whole prediction list. The ordering matters:
    later stages are only consulted when earlier ones leave real ambiguity,
    to keep already-confident predictions from being second-guessed.
    """

    selected_device = device or next(model.parameters()).device
    regions = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)
    regions = _split_touching_character_regions(regions)
    predictions: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for region in regions:
            tensor = tensor_from_character_region(region, selected_device)
            probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
            confidence, label_index = torch.max(probabilities, dim=0)
            label = labels[int(label_index.item())]
            confidence_value = float(confidence.item())
            # Punctuation shapes are geometry-only and considered authoritative
            # over the neural model, which has little/no training signal for
            # marks like parentheses; once set, punctuation_label gates out
            # every later override stage below (they all check `is None`).
            punctuation_label = _punctuation_shape_label(region)
            if punctuation_label is not None:
                label = punctuation_label
                confidence_value = max(confidence_value, 0.92)
            # When the curated model itself is unsure, fall back to whichever
            # cached training exemplar is pixel-closest (nearest neighbor)
            # rather than trusting a low-confidence softmax guess. Distance
            # is converted to a confidence-like score by an empirically
            # chosen linear scale (32.0), clamped to a modest [0.35, 0.9]
            # range since this is a much cruder signal than the model itself.
            exemplar_match = _nearest_exemplar_label(model, tensor)
            if punctuation_label is None and exemplar_match is not None and confidence_value < 0.82:
                exemplar_index, distance = exemplar_match
                label = labels[exemplar_index]
                confidence_value = max(0.35, min(0.9, 1.0 - distance / 32.0))

            # Score letters before alphanumerics so a confident H is not stolen
            # by digit-like model guesses such as 4.
            letter_match = None
            if (
                punctuation_label is None
                and letter_model is not None
                and letter_labels is not None
                and _looks_like_letter_region(region, label, confidence_value)
            ):
                letter_match = _letter_prediction(letter_model, letter_labels, region, selected_device)

            # The combined model is the primary trained 0-9/A-Z classifier, but
            # it only overrides when it agrees with the current character type
            # or when the old curated model was unsure.
            alnum_was_used = False
            alnum_alternatives: list[dict[str, float | str]] = []
            case_alternatives: list[dict[str, float | str]] = []
            if punctuation_label is None and alnum_model is not None and alnum_labels is not None:
                alnum_match = _alnum_prediction(alnum_model, alnum_labels, region, selected_device)
                alnum_label, alnum_confidence, alnum_alternatives = alnum_match
                case_alternatives = _case_ambiguity_alternatives(alnum_alternatives)
                letter_confidence = letter_match[1] if letter_match is not None else 0.0
                if _alnum_should_override(str(label), confidence_value, alnum_label, alnum_confidence, letter_confidence):
                    label = alnum_label
                    confidence_value = max(confidence_value, alnum_confidence)
                    alnum_was_used = True

            # The digit-only model is kept as a narrow rescue path for cases like
            # messy 2/7 drawings, where MNIST is still stronger than EMNIST.
            # Only invoked when there's a real reason to doubt the current
            # label: it's not alphabetic at all, confidence is mediocre, or
            # it's one of the specific letters known to be confused with
            # digits — see _digit_beats_ambiguous_letter.
            digit_match = None
            digit_was_used = False
            if punctuation_label is None and (
                not str(label).isalpha() or confidence_value < 0.86 or str(label) in _DIGIT_AMBIGUOUS_LABELS
            ):
                digit_match = _digit_fallback_prediction(region, selected_device)
                if digit_match is not None:
                    digit_label, digit_confidence = digit_match
                    letter_confidence = letter_match[1] if letter_match is not None else 0.0
                    digit_beats_ambiguous_letter = _digit_beats_ambiguous_letter(
                        digit_label,
                        digit_confidence,
                        str(label),
                        confidence_value,
                    )
                    if (
                        digit_confidence >= 0.80
                        and (letter_confidence < 0.9 or digit_beats_ambiguous_letter)
                        and (digit_beats_ambiguous_letter or not str(label).isalpha() or confidence_value < 0.82)
                    ):
                        label = digit_label
                        confidence_value = max(confidence_value, digit_confidence)
                        digit_was_used = True

            if letter_match is not None and not digit_was_used:
                letter_label, letter_confidence = letter_match
                if _letter_should_override(str(label), confidence_value, letter_confidence, digit_was_used):
                    label = letter_label
                    confidence_value = max(confidence_value, letter_confidence)
            # Final geometric tie-breakers for the handful of shape confusions
            # that models still get wrong even after every other stage above;
            # these only fire when the current label is already one of the
            # specific candidates being disambiguated (e.g. won't relabel a
            # confident "9" just because it vaguely resembles a "4" shape).
            if punctuation_label is None and _looks_like_four(region) and str(label) in {"4", "A", "Y"}:
                label = "4"
                confidence_value = max(confidence_value, 0.92)
            elif punctuation_label is None and _looks_like_seven(region) and str(label) in {"1", "7", "I", "L", "l"}:
                label = "7"
                confidence_value = max(confidence_value, 0.92)
            elif punctuation_label is None and _looks_like_one(region) and str(label) in {"1", "I", "L", "l"}:
                label = "1"
                confidence_value = max(confidence_value, 0.92)
            if case_alternatives:
                # If the label that ultimately won is one of the two
                # case-ambiguous alternatives, show its own alternative
                # confidence rather than whatever confidence value survived
                # the override cascade above, so the displayed number stays
                # consistent with the case-ambiguity note.
                matching_case = next((item for item in case_alternatives if str(item["label"]) == str(label)), None)
                if matching_case is not None:
                    confidence_value = float(matching_case["confidence"])
            top_alternatives = _top_alternatives(alnum_alternatives, str(label)) if alnum_alternatives else []
            x0, y0, x1, y1 = region.box
            prediction: dict[str, float | int | str | list[dict[str, float | str]]] = {
                    "label": label,
                    "confidence": confidence_value,
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "row": region.row,
                }
            if top_alternatives:
                prediction["alternatives"] = top_alternatives
            predictions.append(prediction)
    predictions = _postprocess_exclamations(predictions)
    predictions = _postprocess_lowercase_i(predictions)
    return _postprocess_colons(predictions)


def main() -> None:
    """CLI entrypoint for curated character-model training."""

    parser = argparse.ArgumentParser(description="Train the expanded handwriting character recognizer.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-accuracy", type=float, default=75.0)
    parser.add_argument("--model", choices=sorted(CHARACTER_MODEL_TYPES), default="cnn")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", action="store_true")
    args = parser.parse_args()
    train_character_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_accuracy=args.min_accuracy,
        model_type=args.model,
        device_name=args.device,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        warm_start=args.warm_start,
    )


if __name__ == "__main__":
    main()
