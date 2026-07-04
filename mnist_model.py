from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_DIR / "mnist_cnn.pt"
METRICS_PATH = PROJECT_DIR / "training_metrics.json"
DEFAULT_DATA_ROOTS = [
    PROJECT_DIR / "data",
    Path.home() / "Downloads" / "data",
]

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    seconds: float
    overfit_gap: float


@dataclass(frozen=True)
class DigitRegion:
    image: Image.Image
    box: tuple[int, int, int, int]
    row: int


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_data_root() -> Path:
    for root in DEFAULT_DATA_ROOTS:
        if (root / "MNIST" / "raw").exists():
            return root
    return DEFAULT_DATA_ROOTS[0]


def build_loaders(data_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.92, 1.08),
                shear=5,
            ),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=str(data_root),
        train=True,
        download=not (data_root / "MNIST" / "raw").exists(),
        transform=train_transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=not (data_root / "MNIST" / "raw").exists(),
        transform=test_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


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


def train_model(
    epochs: int = 6,
    batch_size: int = 128,
    min_accuracy: float = 95.0,
    data_root: Path | None = None,
    weights_path: Path = WEIGHTS_PATH,
    metrics_path: Path = METRICS_PATH,
) -> list[EpochMetrics]:
    torch.manual_seed(42)
    np.random.seed(42)

    root = data_root or find_data_root()
    device = get_device()
    train_loader, test_loader = build_loaders(root, batch_size=batch_size)
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[EpochMetrics] = []
    best_accuracy = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss = train_loss_total / max(len(train_loader), 1)
        train_accuracy = 100.0 * train_correct / max(train_total, 1)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            seconds=time.time() - start,
            overfit_gap=train_accuracy - test_accuracy,
        )
        history.append(metrics)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch}/{epochs} "
            f"train_acc={train_accuracy:.2f}% test_acc={test_accuracy:.2f}% "
            f"gap={metrics.overfit_gap:.2f}%"
        )

    if best_state is None or best_accuracy < min_accuracy:
        raise RuntimeError(
            f"Best test accuracy was {best_accuracy:.2f}%, below the {min_accuracy:.2f}% target."
        )

    torch.save(
        {
            "model_state_dict": best_state,
            "test_accuracy": best_accuracy,
            "architecture": "DigitCNN",
            "normalization": {"mean": MNIST_MEAN, "std": MNIST_STD},
        },
        weights_path,
    )
    metrics_path.write_text(json.dumps([asdict(item) for item in history], indent=2), encoding="utf-8")
    return history


def load_model(weights_path: Path = WEIGHTS_PATH, device: torch.device | None = None) -> DigitCNN:
    selected_device = device or get_device()
    checkpoint = torch.load(weights_path, map_location=selected_device, weights_only=True)
    model = DigitCNN().to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _foreground_from_image(image: Image.Image) -> np.ndarray:
    grayscale = ImageOps.grayscale(image)
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    border = np.concatenate((array[0, :], array[-1, :], array[:, 0], array[:, -1]))
    if float(np.median(border)) > 0.5:
        array = 1.0 - array
    array[array < 0.18] = 0.0
    return array


def _group_boxes_by_reading_order(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    if not boxes:
        return []
    if len(boxes) == 1:
        return [boxes]

    heights = [y1 - y0 for _, y0, _, y1 in boxes]
    row_tolerance = max(8.0, float(np.median(heights)) * 0.7)
    rows: list[list[tuple[int, int, int, int]]] = []

    for box in sorted(boxes, key=lambda item: ((item[1] + item[3]) / 2, item[0])):
        center_y = (box[1] + box[3]) / 2
        for row in rows:
            row_center = np.mean([(item[1] + item[3]) / 2 for item in row])
            if abs(center_y - row_center) <= row_tolerance:
                row.append(box)
                break
        else:
            rows.append([box])

    sorted_rows = sorted(rows, key=lambda row: min(item[1] for item in row))
    return [sorted(row, key=lambda item: item[0]) for row in sorted_rows]


def _sort_boxes_reading_order(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return [box for row in _group_boxes_by_reading_order(boxes) for box in row]


def _best_vertical_cut(mask: np.ndarray, left: int, right: int) -> int:
    left = max(0, left)
    right = min(mask.shape[1] - 1, right)
    if right <= left + 2:
        return (left + right) // 2

    projection = mask[:, left:right].sum(axis=0).astype(np.float32)
    if projection.size >= 7:
        projection = np.convolve(projection, np.ones(7, dtype=np.float32) / 7.0, mode="same")
    min_value = float(projection.min())
    candidates = np.where(projection <= min_value + 0.5)[0]
    midpoint = (right - left) / 2.0
    selected = int(candidates[np.argmin(np.abs(candidates - midpoint))])
    return left + selected


def _trim_overlapping_boxes(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes

    adjusted = [list(box) for box in sorted(boxes, key=lambda item: item[0])]
    for index in range(len(adjusted) - 1):
        current = adjusted[index]
        following = adjusted[index + 1]
        overlap = current[2] - following[0]
        if overlap <= 0:
            continue

        current_width = current[2] - current[0]
        following_width = following[2] - following[0]
        overlap_ratio = overlap / max(1, min(current_width, following_width))
        if overlap_ratio < 0.12:
            continue

        cut = _best_vertical_cut(mask, following[0], current[2])
        if overlap_ratio >= 0.35:
            cut = min(cut, following[0] + int(overlap * 0.35))
        min_width = max(8, int(np.median([current_width, following_width]) * 0.18))
        if cut - current[0] >= min_width and following[2] - cut >= min_width:
            current[2] = cut
            following[0] = cut

    return [tuple(box) for box in adjusted if box[2] > box[0] and box[3] > box[1]]


def _split_wide_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    if height <= 20 or width / height < 1.45:
        return [box]

    estimated_digits = int(round(width / max(1.0, height * 0.62)))
    if estimated_digits < 2:
        return [box]
    estimated_digits = min(4, estimated_digits)

    cuts: list[int] = []
    segment_left = x0
    for part in range(1, estimated_digits):
        expected = x0 + int(width * part / estimated_digits)
        search_radius = max(12, int(width / estimated_digits * 0.35))
        cut = _best_vertical_cut(mask[y0:y1, :], expected - search_radius, expected + search_radius)
        if cut - segment_left > 8 and x1 - cut > 8:
            cuts.append(cut)
            segment_left = cut

    if not cuts:
        return [box]

    edges = [x0, *cuts, x1]
    split_boxes = []
    for left, right in zip(edges, edges[1:]):
        submask = mask[y0:y1, left:right]
        ys, xs = np.where(submask)
        if len(xs) < 12:
            continue
        split_boxes.append((left + int(xs.min()), y0 + int(ys.min()), left + int(xs.max()) + 1, y0 + int(ys.max()) + 1))
    return split_boxes or [box]


def _refine_digit_boxes(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    split_boxes: list[tuple[int, int, int, int]] = []
    for box in boxes:
        split_boxes.extend(_split_wide_box(mask, box))
    return _trim_overlapping_boxes(mask, split_boxes)


def _box_ink_area(mask: np.ndarray, box: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box
    return int(mask[y0:y1, x0:x1].sum())


def _gap_between_boxes(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    first_x0, first_y0, first_x1, first_y1 = first
    second_x0, second_y0, second_x1, second_y1 = second
    horizontal_gap = max(first_x0 - second_x1, second_x0 - first_x1, 0)
    vertical_gap = max(first_y0 - second_y1, second_y0 - first_y1, 0)
    return float(np.hypot(horizontal_gap, vertical_gap))


def _merge_small_mark_boxes(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes

    areas = [_box_ink_area(mask, box) for box in boxes]
    median_area = float(np.median(areas))
    small_area_limit = max(18.0, median_area * 0.35)
    merged_boxes: list[tuple[int, int, int, int]] = []
    consumed: set[int] = set()

    for index, box in enumerate(boxes):
        if index in consumed:
            continue
        x0, y0, x1, y1 = box
        area = areas[index]
        width = x1 - x0
        height = y1 - y0
        is_small = area <= small_area_limit or min(width, height) <= 7
        if is_small:
            continue

        merged = box
        center_x = (x0 + x1) / 2.0
        for mark_index, mark_box in enumerate(boxes):
            if mark_index == index or mark_index in consumed:
                continue
            mark_area = areas[mark_index]
            mx0, my0, mx1, my1 = mark_box
            mark_width = mx1 - mx0
            mark_height = my1 - my0
            mark_is_small = mark_area <= small_area_limit or min(mark_width, mark_height) <= 7
            if not mark_is_small:
                continue
            mark_center_x = (mx0 + mx1) / 2.0
            horizontal_limit = max(width, mark_width, 12) * 0.75
            vertical_limit = max(height, mark_height, 14) * 0.85
            if abs(center_x - mark_center_x) > horizontal_limit:
                continue
            if _gap_between_boxes(box, mark_box) > vertical_limit:
                continue
            merged = (min(merged[0], mx0), min(merged[1], my0), max(merged[2], mx1), max(merged[3], my1))
            consumed.add(mark_index)

        consumed.add(index)
        merged_boxes.append(merged)

    for index, box in enumerate(boxes):
        if index not in consumed:
            merged_boxes.append(box)
    return _sort_boxes_reading_order(merged_boxes)


def segment_digit_regions(
    image: Image.Image,
    split_wide: bool = True,
    min_component_pixels: int = 12,
    merge_marks: bool = False,
) -> list[DigitRegion]:
    foreground = _foreground_from_image(image)
    mask = foreground > 0.18
    if int(mask.sum()) < 20:
        return []

    label_mask = ndimage.binary_dilation(mask, iterations=2)
    labeled, count = ndimage.label(label_mask)
    boxes: list[tuple[int, int, int, int]] = []

    for label in range(1, count + 1):
        ys, xs = np.where(labeled == label)
        if len(xs) < min_component_pixels:
            continue
        dx0, dx1 = int(xs.min()), int(xs.max()) + 1
        dy0, dy1 = int(ys.min()), int(ys.max()) + 1
        original_component = mask[dy0:dy1, dx0:dx1] & (labeled[dy0:dy1, dx0:dx1] == label)
        original_ys, original_xs = np.where(original_component)
        if len(original_xs) < min_component_pixels:
            continue
        x0, x1 = dx0 + int(original_xs.min()), dx0 + int(original_xs.max()) + 1
        y0, y1 = dy0 + int(original_ys.min()), dy0 + int(original_ys.max()) + 1
        width = x1 - x0
        height = y1 - y0
        if width * height < 20:
            continue
        boxes.append((x0, y0, x1, y1))

    if split_wide:
        boxes = _refine_digit_boxes(mask, boxes)
    if merge_marks:
        boxes = _merge_small_mark_boxes(mask, boxes)
    rows = _group_boxes_by_reading_order(boxes)
    digits: list[DigitRegion] = []
    for row_index, row in enumerate(rows, start=1):
        for box_index, (x0, y0, x1, y1) in enumerate(row):
            pad = max(4, int(max(x1 - x0, y1 - y0) * 0.15))
            left_limit = 0
            right_limit = foreground.shape[1]
            if box_index > 0:
                left_limit = row[box_index - 1][2]
            if box_index < len(row) - 1:
                right_limit = row[box_index + 1][0]

            x0 = max(left_limit, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(right_limit, x1 + pad)
            y1 = min(foreground.shape[0], y1 + pad)
            crop = (foreground[y0:y1, x0:x1] * 255).astype(np.uint8)
            digits.append(DigitRegion(image=Image.fromarray(crop, mode="L"), box=(x0, y0, x1, y1), row=row_index))
    return digits


def segment_digits(image: Image.Image) -> list[Image.Image]:
    return [region.image for region in segment_digit_regions(image)]


def mnist_normalize_image(image: Image.Image) -> Image.Image:
    array = _foreground_from_image(image)
    ys, xs = np.where(array > 0.18)
    if len(xs) > 0:
        array = array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    height, width = array.shape
    scale = 20.0 / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    digit = Image.fromarray((array * 255).astype(np.uint8), mode="L").resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (28, 28), 0)
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2
    canvas.paste(digit, (left, top))

    canvas_array = np.asarray(canvas, dtype=np.float32)
    center = ndimage.center_of_mass(canvas_array)
    if not any(np.isnan(center)):
        shift_y = int(round(14 - center[0]))
        shift_x = int(round(14 - center[1]))
        shifted = ndimage.shift(canvas_array, shift=(shift_y, shift_x), order=1, mode="constant", cval=0)
        canvas = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8), mode="L")
    return canvas


def tensor_from_digit(image: Image.Image, device: torch.device, thicken: bool = False) -> torch.Tensor:
    normalized = mnist_normalize_image(image)
    array = np.asarray(normalized, dtype=np.float32) / 255.0
    if thicken:
        array = ndimage.maximum_filter(array, size=2)
    array = (array - MNIST_MEAN) / MNIST_STD
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def _predict_digit_tensor(model: nn.Module, tensor: torch.Tensor) -> tuple[int, float]:
    probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    confidence, label = torch.max(probabilities, dim=0)
    return int(label.item()), float(confidence.item())


def _low_confidence_digit_variants(image: Image.Image) -> list[Image.Image]:
    variants = [image]
    width, height = image.size
    if height > 0 and width / height >= 0.85:
        for ratio in (0.88, 0.78):
            right = max(1, int(width * ratio))
            variants.append(image.crop((0, 0, right, height)))
    return variants


def _predict_digit_image(model: nn.Module, image: Image.Image, device: torch.device) -> tuple[int, float]:
    label, confidence = _predict_digit_tensor(model, tensor_from_digit(image, device))
    if confidence >= 0.85:
        return label, confidence

    best_label = label
    best_confidence = confidence
    for variant in _low_confidence_digit_variants(image):
        for thicken in (False, True):
            candidate_label, candidate_confidence = _predict_digit_tensor(
                model,
                tensor_from_digit(variant, device, thicken=thicken),
            )
            if candidate_confidence > best_confidence:
                best_label = candidate_label
                best_confidence = candidate_confidence
    return best_label, best_confidence


def predict_digits(model: nn.Module, image: Image.Image, device: torch.device | None = None) -> list[dict[str, float | int]]:
    selected_device = device or next(model.parameters()).device
    digit_regions = segment_digit_regions(image)
    predictions: list[dict[str, float | int]] = []
    with torch.no_grad():
        for digit_region in digit_regions:
            label, confidence = _predict_digit_image(model, digit_region.image, selected_device)
            x0, y0, x1, y1 = digit_region.box
            predictions.append(
                {
                    "digit": label,
                    "confidence": confidence,
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "row": digit_region.row,
                }
            )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the MNIST digit CNN.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-accuracy", type=float, default=95.0)
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, min_accuracy=args.min_accuracy)


if __name__ == "__main__":
    main()
