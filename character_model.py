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
_DIGIT_MODEL: nn.Module | None = None


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
    """Small MLP-style classifier for curated character glyphs."""

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
    """Center and scale one character crop into a 28x28 foreground image."""

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
    """Save one exemplar tensor per curated sample for nearest-neighbor fallback."""

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
) -> list[CharacterEpochMetrics]:
    """Train the curated character model and save weights/labels/exemplars."""

    if not dataset_root.exists():
        raise RuntimeError(f"Missing dataset at {dataset_root}")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = get_device()
    if device.type == "mps":
        device = torch.device("cpu")
    train_loader, validation_loader, labels = make_loaders(dataset_root, batch_size)
    model = CharacterCNN(num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[CharacterEpochMetrics] = []
    best_accuracy = 0.0
    best_state = None

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
            "image_size": IMAGE_SIZE,
            "normalization": {"mean": CHAR_MEAN, "std": CHAR_STD},
        },
        WEIGHTS_PATH,
    )
    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    build_character_exemplars(dataset_root)
    METRICS_PATH.write_text(json.dumps([asdict(item) for item in history], indent=2), encoding="utf-8")
    return history


def load_character_model(weights_path: Path = WEIGHTS_PATH, device: torch.device | None = None) -> tuple[CharacterCNN, list[str]]:
    """Load the curated character classifier and optional exemplar bank."""

    selected_device = device or get_device()
    checkpoint = torch.load(weights_path, map_location=selected_device, weights_only=True)
    labels = list(checkpoint["labels"])
    model = CharacterCNN(num_classes=len(labels)).to(selected_device)
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
    """Find the closest curated exemplar for low-confidence predictions."""

    exemplars = getattr(model, "character_exemplars", None)
    targets = getattr(model, "character_exemplar_targets", None)
    if exemplars is None or targets is None:
        return None
    vector = tensor.detach().cpu().flatten().float().unsqueeze(0)
    distances = torch.cdist(vector, exemplars.reshape(exemplars.size(0), -1)).squeeze(0)
    distance, exemplar_index = torch.min(distances, dim=0)
    return int(targets[int(exemplar_index)].item()), float(distance.item())


def _looks_like_letter_region(region: DigitRegion, label: str, confidence: float) -> bool:
    """Decide whether a region is worth sending to the alphabet model."""

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
) -> tuple[str, float]:
    """Predict one region with the combined digit+letter model."""

    tensor = tensor_from_alnum_region(region, device)
    probabilities = torch.softmax(alnum_model(tensor), dim=1).squeeze(0)
    confidence, label_index = torch.max(probabilities, dim=0)
    return alnum_labels[int(label_index.item())], float(confidence.item())


def _load_digit_model_for_fallback(device: torch.device) -> nn.Module | None:
    """Lazily load the digit CNN for ambiguous numeric-looking regions."""

    global _DIGIT_MODEL
    if _DIGIT_MODEL is not None:
        return _DIGIT_MODEL
    if not DIGIT_WEIGHTS_PATH.exists():
        return None
    _DIGIT_MODEL = load_model(device=device)
    return _DIGIT_MODEL


def _digit_fallback_prediction(region: DigitRegion, device: torch.device) -> tuple[str, float] | None:
    """Ask the MNIST digit model to rescue uncertain digit-like regions."""

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
    """Return true for known digit/letter pairs where MNIST should win."""

    return (
        digit_confidence >= 0.985
        and digit_label == "2"
        and current_label == "Z"
        and current_confidence < 0.96
    ) or (
        digit_confidence >= 0.94
        and (
            (digit_label == "4" and current_label == "Y")
            or (digit_label == "5" and current_label == "J")
        )
    )


def _letter_should_override(
    current_label: str,
    current_confidence: float,
    letter_confidence: float,
    digit_was_used: bool,
) -> bool:
    """Decide when the letter-only model is strong enough to replace a label."""

    if digit_was_used:
        return False
    if not current_label.isalnum():
        return letter_confidence >= 0.55
    if current_label.isalpha():
        return letter_confidence >= 0.70 and letter_confidence >= current_confidence - 0.03
    return letter_confidence >= 0.92 and letter_confidence >= current_confidence + 0.08


def _mask_span(mask: np.ndarray) -> float:
    """Measure how much horizontal space a foreground mask occupies."""

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    return float(xs.max() - xs.min() + 1) / max(float(mask.shape[1]), 1.0)


def _looks_like_seven(region: DigitRegion) -> bool:
    """Recognize the tall handwritten 7 shape that models confuse with 1."""

    mask = _foreground_from_image(region.image) > 0.18
    height, width = mask.shape
    if height <= 0 or width <= 0:
        return False
    top_span = _mask_span(mask[: max(1, int(height * 0.25)), :])
    bottom_span = _mask_span(mask[int(height * 0.55) :, :])
    aspect_ratio = width / max(height, 1)
    return 0.24 <= aspect_ratio <= 0.62 and top_span >= 0.48 and bottom_span <= 0.22


def _looks_like_one(region: DigitRegion) -> bool:
    """Recognize a plain vertical 1 that models confuse with L."""

    mask = _foreground_from_image(region.image) > 0.18
    height, width = mask.shape
    if height <= 0 or width <= 0:
        return False
    top_span = _mask_span(mask[: max(1, int(height * 0.25)), :])
    bottom_span = _mask_span(mask[int(height * 0.55) :, :])
    aspect_ratio = width / max(height, 1)
    return aspect_ratio <= 0.34 and top_span <= 0.22 and bottom_span <= 0.24


def _looks_like_four(region: DigitRegion) -> bool:
    """Recognize open-top handwritten 4s that letter models confuse with A."""

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


def _punctuation_shape_label(region: DigitRegion) -> str | None:
    """Detect punctuation whose geometry is clearer than model logits."""

    array = np.asarray(region.image, dtype=np.float32) / 255.0
    mask = array > 0.18
    if mask.mean() > 0.5:
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
    """Check whether a dot sits above and aligned with an i-like stem."""

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
    """Segment an image and predict letters, digits, and punctuation."""

    selected_device = device or next(model.parameters()).device
    regions = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)
    predictions: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for region in regions:
            tensor = tensor_from_character_region(region, selected_device)
            probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
            confidence, label_index = torch.max(probabilities, dim=0)
            label = labels[int(label_index.item())]
            confidence_value = float(confidence.item())
            punctuation_label = _punctuation_shape_label(region)
            if punctuation_label is not None:
                label = punctuation_label
                confidence_value = max(confidence_value, 0.92)
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
            if punctuation_label is None and alnum_model is not None and alnum_labels is not None:
                alnum_match = _alnum_prediction(alnum_model, alnum_labels, region, selected_device)
                alnum_label, alnum_confidence = alnum_match
                letter_confidence = letter_match[1] if letter_match is not None else 0.0
                if (
                    alnum_confidence >= 0.55
                    and not (alnum_label.isdigit() and letter_confidence >= 0.9)
                    and (
                        confidence_value < 0.86
                        or not str(label).isalnum()
                        or (
                            str(label).isdigit()
                            and alnum_label.isdigit()
                            and alnum_confidence >= confidence_value - 0.05
                        )
                        or (
                            str(label).isalpha()
                            and alnum_label.isalpha()
                            and alnum_confidence >= confidence_value - 0.05
                        )
                    )
                ):
                    label = alnum_label
                    confidence_value = max(confidence_value, alnum_confidence)
                    alnum_was_used = True

            # The digit-only model is kept as a narrow rescue path for cases like
            # messy 2/7 drawings, where MNIST is still stronger than EMNIST.
            digit_match = None
            digit_was_used = False
            if punctuation_label is None and (
                not str(label).isalpha() or confidence_value < 0.86 or str(label) in {"J", "Y", "Z"}
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
            if punctuation_label is None and _looks_like_four(region) and str(label) in {"4", "A", "Y"}:
                label = "4"
                confidence_value = max(confidence_value, 0.92)
            elif punctuation_label is None and _looks_like_seven(region) and str(label) in {"1", "7", "I", "L", "l"}:
                label = "7"
                confidence_value = max(confidence_value, 0.92)
            elif punctuation_label is None and _looks_like_one(region) and str(label) in {"1", "I", "L", "l"}:
                label = "1"
                confidence_value = max(confidence_value, 0.92)
            x0, y0, x1, y1 = region.box
            predictions.append(
                {
                    "label": label,
                    "confidence": confidence_value,
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "row": region.row,
                }
            )
    predictions = _postprocess_exclamations(predictions)
    predictions = _postprocess_lowercase_i(predictions)
    return _postprocess_colons(predictions)


def main() -> None:
    """CLI entrypoint for curated character-model training."""

    parser = argparse.ArgumentParser(description="Train the expanded handwriting character recognizer.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-accuracy", type=float, default=75.0)
    args = parser.parse_args()
    train_character_model(epochs=args.epochs, batch_size=args.batch_size, min_accuracy=args.min_accuracy)


if __name__ == "__main__":
    main()
