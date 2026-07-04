from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import ImageOps
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_model import get_device


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / "data" / "emnist"
METRICS_PATH = PROJECT_DIR / "emnist_experiment_metrics.json"
WEIGHTS_PATH = PROJECT_DIR / "emnist_experiment.pt"
EMNIST_MEAN = 0.1736
EMNIST_STD = 0.3248


@dataclass(frozen=True)
class ExperimentMetrics:
    epoch: int
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float
    seconds: float


class EmnistMLP(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(384, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class EmnistCNN(nn.Module):
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
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def emnist_transform(augment: bool = False) -> transforms.Compose:
    steps = [transforms.Lambda(lambda image: ImageOps.mirror(image.rotate(-90, expand=True)))]
    if augment:
        steps.append(
            transforms.RandomAffine(
                degrees=8,
                translate=(0.06, 0.06),
                scale=(0.92, 1.08),
                shear=6,
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


def make_loaders(batch_size: int, split: str, augment: bool) -> tuple[DataLoader, DataLoader, list[str]]:
    train_transform = emnist_transform(augment=augment)
    test_transform = emnist_transform(augment=False)
    target_transform = (lambda label: label - 1) if split == "letters" else None
    train_dataset = datasets.EMNIST(
        DATA_ROOT,
        split=split,
        train=True,
        download=True,
        transform=train_transform,
        target_transform=target_transform,
    )
    test_dataset = datasets.EMNIST(
        DATA_ROOT,
        split=split,
        train=False,
        download=True,
        transform=test_transform,
        target_transform=target_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    labels = list(train_dataset.classes[1:]) if split == "letters" else list(train_dataset.classes)
    return train_loader, test_loader, labels


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
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_total / max(len(loader), 1), 100.0 * correct / max(total, 1)


def train(epochs: int, batch_size: int, split: str, model_type: str, augment: bool) -> list[ExperimentMetrics]:
    device = get_device()
    if device.type == "mps":
        device = torch.device("cpu")
    torch.manual_seed(42)

    train_loader, test_loader, labels = make_loaders(batch_size, split, augment)
    model_class = EmnistCNN if model_type == "cnn" else EmnistMLP
    model = model_class(num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[ExperimentMetrics] = []
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
            train_correct += (outputs.argmax(dim=1) == labels_tensor).sum().item()
            train_total += labels_tensor.size(0)
        scheduler.step()
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        metrics = ExperimentMetrics(
            epoch=epoch,
            train_accuracy=100.0 * train_correct / max(train_total, 1),
            test_accuracy=test_accuracy,
            train_loss=train_loss_total / max(len(train_loader), 1),
            test_loss=test_loss,
            seconds=time.time() - start,
        )
        history.append(metrics)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"Epoch {epoch}/{epochs} train_acc={metrics.train_accuracy:.2f}% "
            f"test_acc={metrics.test_accuracy:.2f}%",
            flush=True,
        )

    torch.save(
        {
            "model_state_dict": best_state,
            "labels": labels,
            "test_accuracy": best_accuracy,
            "split": split,
            "model_type": model_type,
            "augment": augment,
        },
        WEIGHTS_PATH,
    )
    payload = {
        "split": split,
        "model_type": model_type,
        "augment": augment,
        "history": [asdict(item) for item in history],
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EMNIST balanced alphabet/digit baseline.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--split", choices=["balanced", "letters"], default="balanced")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        split=args.split,
        model_type=args.model,
        augment=args.augment,
    )


if __name__ == "__main__":
    main()
