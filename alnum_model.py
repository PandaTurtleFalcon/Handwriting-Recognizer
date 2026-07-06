"""Train and load the combined digit + alphabet recognizer.

This module joins MNIST digits with EMNIST letters into one 36-class dataset
(`0-9` and `A-Z`). The website still keeps older specialist models around for
hard edge cases, but this checkpoint is the main high-accuracy alphanumeric
model shown in the UI badge.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import datasets, transforms

from emnist_experiment import DATA_ROOT as EMNIST_DATA_ROOT
from emnist_experiment import EMNIST_MEAN, EMNIST_STD, EmnistCNN, EmnistMLP, TinyEmnistCNN, WideEmnistCNN, build_or_load_emnist_cache
from mnist_model import get_device


PROJECT_DIR = Path(__file__).resolve().parent
MNIST_DATA_ROOT = PROJECT_DIR / "data" / "mnist"
WEIGHTS_PATH = PROJECT_DIR / "alnum_cnn.pt"
METRICS_PATH = PROJECT_DIR / "alnum_training_metrics.json"
LABELS = [str(index) for index in range(10)] + [chr(ord("A") + index) for index in range(26)]
MODEL_CLASSES = {
    "mlp": EmnistMLP,
    "tinycnn": TinyEmnistCNN,
    "cnn": EmnistCNN,
    "widecnn": WideEmnistCNN,
}


@dataclass(frozen=True)
class AlnumEpochMetrics:
    """Metrics captured at the end of each combined training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    digit_test_accuracy: float
    letter_test_accuracy: float
    seconds: float
    overfit_gap: float


def mnist_transform(augment: bool = False) -> transforms.Compose:
    """Return the MNIST transform using the same normalization as EMNIST."""

    steps: list[object] = []
    if augment:
        steps.append(
            transforms.RandomAffine(
                degrees=8,
                translate=(0.06, 0.06),
                scale=(0.92, 1.08),
                shear=5,
                fill=0,
            )
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,)),
        ]
    )
    return transforms.Compose(steps)


def build_or_load_mnist_cache(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached MNIST tensors, or download and cache them on first use."""

    cache_path = MNIST_DATA_ROOT / f"cache_mnist_{'train' if train else 'test'}.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cache["images"], cache["targets"]

    dataset = datasets.MNIST(
        root=str(MNIST_DATA_ROOT),
        train=train,
        download=True,
        transform=mnist_transform(augment=False),
    )
    images = []
    targets = []
    for image, target in dataset:
        images.append(image)
        targets.append(int(target))
    image_tensor = torch.stack(images)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": image_tensor, "targets": target_tensor}, cache_path)
    return image_tensor, target_tensor


def _limit_per_class(
    images: torch.Tensor,
    targets: torch.Tensor,
    samples_per_class: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic class-balanced subset for faster experiments."""

    if samples_per_class is None:
        return images, targets
    generator = torch.Generator().manual_seed(seed)
    selected_indices = []
    for label in sorted(set(int(item) for item in targets.tolist())):
        label_indices = torch.where(targets == label)[0]
        if len(label_indices) > samples_per_class:
            order = torch.randperm(len(label_indices), generator=generator)[:samples_per_class]
            label_indices = label_indices[order]
        selected_indices.append(label_indices)
    merged_indices = torch.cat(selected_indices)
    order = torch.randperm(len(merged_indices), generator=generator)
    merged_indices = merged_indices[order]
    return images[merged_indices], targets[merged_indices]


def make_cached_loaders(
    batch_size: int,
    samples_per_class: int | None,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Build loaders from cached tensors for repeatable high-speed training."""

    mnist_train_images, mnist_train_targets = build_or_load_mnist_cache(train=True)
    mnist_test_images, mnist_test_targets = build_or_load_mnist_cache(train=False)
    letter_train_images, letter_train_targets, _ = build_or_load_emnist_cache("letters", train=True)
    letter_test_images, letter_test_targets, _ = build_or_load_emnist_cache("letters", train=False)

    letter_train_targets = letter_train_targets + 10
    letter_test_targets = letter_test_targets + 10
    mnist_train_images, mnist_train_targets = _limit_per_class(
        mnist_train_images,
        mnist_train_targets,
        samples_per_class,
        seed,
    )
    letter_train_images, letter_train_targets = _limit_per_class(
        letter_train_images,
        letter_train_targets,
        samples_per_class,
        seed + 1,
    )

    train_dataset = ConcatDataset(
        [
            TensorDataset(mnist_train_images, mnist_train_targets),
            TensorDataset(letter_train_images, letter_train_targets),
        ]
    )
    test_dataset = ConcatDataset(
        [
            TensorDataset(mnist_test_images, mnist_test_targets),
            TensorDataset(letter_test_images, letter_test_targets),
        ]
    )
    train_targets = torch.cat([mnist_train_targets, letter_train_targets]).numpy()
    class_counts = np.bincount(train_targets, minlength=len(LABELS))
    sample_weights = [1.0 / max(class_counts[int(target)], 1) for target in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_targets), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    digit_test_loader = DataLoader(TensorDataset(mnist_test_images, mnist_test_targets), batch_size=batch_size)
    letter_test_loader = DataLoader(TensorDataset(letter_test_images, letter_test_targets), batch_size=batch_size)
    return train_loader, test_loader, digit_test_loader, letter_test_loader


def make_augmented_loaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Build loaders with live MNIST augmentation and cached EMNIST letters."""

    mnist_train = datasets.MNIST(
        root=str(MNIST_DATA_ROOT),
        train=True,
        download=True,
        transform=mnist_transform(augment=True),
    )
    mnist_test = datasets.MNIST(
        root=str(MNIST_DATA_ROOT),
        train=False,
        download=True,
        transform=mnist_transform(augment=False),
    )
    letter_train_images, letter_train_targets, _ = build_or_load_emnist_cache("letters", train=True)
    letter_test_images, letter_test_targets, _ = build_or_load_emnist_cache("letters", train=False)
    letter_train_targets = letter_train_targets + 10
    letter_test_targets = letter_test_targets + 10
    letter_train = TensorDataset(letter_train_images, letter_train_targets)
    letter_test = TensorDataset(letter_test_images, letter_test_targets)

    train_dataset = ConcatDataset([mnist_train, letter_train])
    test_dataset = ConcatDataset([mnist_test, letter_test])
    train_targets = np.concatenate(
        [
            np.asarray(mnist_train.targets, dtype=np.int64),
            letter_train_targets.numpy(),
        ]
    )
    class_counts = np.bincount(train_targets, minlength=len(LABELS))
    sample_weights = [1.0 / max(class_counts[int(target)], 1) for target in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_targets), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    digit_test_loader = DataLoader(mnist_test, batch_size=batch_size)
    letter_test_loader = DataLoader(letter_test, batch_size=batch_size)
    return train_loader, test_loader, digit_test_loader, letter_test_loader


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Evaluate a model and return average loss plus accuracy percentage."""

    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, targets).item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
    return loss_total / max(len(loader), 1), 100.0 * correct / max(total, 1)


def save_checkpoint(
    history: list[AlnumEpochMetrics],
    best_state: dict[str, torch.Tensor] | None,
    best_accuracy: float,
    model_type: str,
    augment: bool,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> None:
    """Persist the best model weights and the full metrics history."""

    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "labels": LABELS,
                "test_accuracy": best_accuracy,
                "model_type": model_type,
                "augment": augment,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": str(device),
                "normalization": {"mean": EMNIST_MEAN, "std": EMNIST_STD},
            },
            WEIGHTS_PATH,
        )
    METRICS_PATH.write_text(
        json.dumps(
            {
                "labels": LABELS,
                "model_type": model_type,
                "augment": augment,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": str(device),
                "history": [asdict(item) for item in history],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_alnum_model(
    weights_path: Path = WEIGHTS_PATH,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[str]] | tuple[None, None]:
    """Load the trained combined recognizer if its checkpoint exists."""

    if not weights_path.exists():
        return None, None
    selected_device = device or get_device()
    checkpoint = torch.load(weights_path, map_location=selected_device, weights_only=True)
    labels = [str(label) for label in checkpoint["labels"]]
    model_type = str(checkpoint.get("model_type", "cnn"))
    model_class = MODEL_CLASSES.get(model_type, EmnistCNN)
    model = model_class(num_classes=len(labels)).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, labels


def train(
    epochs: int,
    batch_size: int,
    min_accuracy: float,
    learning_rate: float,
    seed: int,
    augment: bool,
    model_type: str,
    samples_per_class: int | None,
    device_name: str,
) -> list[AlnumEpochMetrics]:
    """Train a 36-class recognizer until it clears the requested accuracy."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device_name == "cpu":
        device = torch.device("cpu")
    elif device_name == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        device = torch.device("mps")
    else:
        device = get_device()

    loaders = make_augmented_loaders(batch_size) if augment else make_cached_loaders(batch_size, samples_per_class, seed)
    train_loader, test_loader, digit_test_loader, letter_test_loader = loaders
    model = MODEL_CLASSES[model_type](num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[AlnumEpochMetrics] = []
    best_accuracy = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            train_total += targets.size(0)
        scheduler.step()

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        _, digit_accuracy = evaluate(model, digit_test_loader, criterion, device)
        _, letter_accuracy = evaluate(model, letter_test_loader, criterion, device)
        metrics = AlnumEpochMetrics(
            epoch=epoch,
            train_loss=train_loss_total / max(len(train_loader), 1),
            train_accuracy=100.0 * train_correct / max(train_total, 1),
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            digit_test_accuracy=digit_accuracy,
            letter_test_accuracy=letter_accuracy,
            seconds=time.time() - start,
            overfit_gap=100.0 * train_correct / max(train_total, 1) - test_accuracy,
        )
        history.append(metrics)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        save_checkpoint(history, best_state, best_accuracy, model_type, augment, learning_rate, seed, device)
        print(
            f"Epoch {epoch}/{epochs} train_acc={metrics.train_accuracy:.2f}% "
            f"test_acc={metrics.test_accuracy:.2f}% digits={digit_accuracy:.2f}% "
            f"letters={letter_accuracy:.2f}% gap={metrics.overfit_gap:.2f}%",
            flush=True,
        )
        if best_accuracy >= min_accuracy and epoch >= 8:
            break

    if best_accuracy < min_accuracy:
        raise RuntimeError(f"Best combined test accuracy was {best_accuracy:.2f}%, below {min_accuracy:.2f}%.")
    return history


def main() -> None:
    """CLI entrypoint for training the combined recognizer."""

    parser = argparse.ArgumentParser(description="Train a combined MNIST + EMNIST alphanumeric recognizer.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--min-accuracy", type=float, default=95.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--model", choices=["mlp", "tinycnn", "cnn", "widecnn"], default="cnn")
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_accuracy=args.min_accuracy,
        learning_rate=args.learning_rate,
        seed=args.seed,
        augment=args.augment,
        model_type=args.model,
        samples_per_class=args.samples_per_class,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
