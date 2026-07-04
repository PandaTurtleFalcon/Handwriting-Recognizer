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
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, WeightedRandomSampler
from torchvision import transforms

from mnist_model import DigitRegion, get_device, segment_digit_regions


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


@dataclass(frozen=True)
class CharacterEpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    seconds: float


class CharacterDataset(Dataset):
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
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((CHAR_MEAN,), (CHAR_STD,)),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((CHAR_MEAN,), (CHAR_STD,)),
        ]
    )


def normalize_character_image(image: Image.Image) -> Image.Image:
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
    normalized = normalize_character_image(image)
    array = np.asarray(normalized, dtype=np.float32) / 255.0
    array = (array - CHAR_MEAN) / CHAR_STD
    return torch.from_numpy(array).unsqueeze(0)


def split_dataset(dataset: CharacterDataset, validation_size: float = 0.15) -> tuple[list[int], list[int]]:
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


def tensor_from_character_region(region: DigitRegion, device: torch.device) -> torch.Tensor:
    return character_tensor_from_image(region.image).unsqueeze(0).to(device)


def _nearest_exemplar_label(model: nn.Module, tensor: torch.Tensor) -> tuple[int, float] | None:
    exemplars = getattr(model, "character_exemplars", None)
    targets = getattr(model, "character_exemplar_targets", None)
    if exemplars is None or targets is None:
        return None
    vector = tensor.detach().cpu().flatten().float().unsqueeze(0)
    distances = torch.cdist(vector, exemplars.reshape(exemplars.size(0), -1)).squeeze(0)
    distance, exemplar_index = torch.min(distances, dim=0)
    return int(targets[int(exemplar_index)].item()), float(distance.item())


def predict_characters(
    model: nn.Module,
    labels: list[str],
    image: Image.Image,
    device: torch.device | None = None,
) -> list[dict[str, float | int | str]]:
    selected_device = device or next(model.parameters()).device
    regions = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)
    predictions: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for region in regions:
            tensor = tensor_from_character_region(region, selected_device)
            probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
            confidence, label_index = torch.max(probabilities, dim=0)
            label = labels[int(label_index.item())]
            exemplar_match = _nearest_exemplar_label(model, tensor)
            if exemplar_match is not None and float(confidence.item()) < 0.82:
                exemplar_index, distance = exemplar_match
                label = labels[exemplar_index]
                confidence = torch.tensor(max(0.35, min(0.9, 1.0 - distance / 32.0)))
            x0, y0, x1, y1 = region.box
            predictions.append(
                {
                    "label": label,
                    "confidence": float(confidence.item()),
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "row": region.row,
                }
            )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the expanded handwriting character recognizer.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-accuracy", type=float, default=75.0)
    args = parser.parse_args()
    train_character_model(epochs=args.epochs, batch_size=args.batch_size, min_accuracy=args.min_accuracy)


if __name__ == "__main__":
    main()
