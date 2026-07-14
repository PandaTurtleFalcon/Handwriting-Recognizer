"""Train, load, segment, and predict handwritten MNIST-style digits.

This file owns the digit-only CNN and the image preprocessing pipeline shared
by the website and the character recognizer. The segmentation helpers turn an
uploaded page into ordered bounding boxes before any neural model is called.
"""

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

# The canonical MNIST per-pixel mean/std (computed over the training set).
# Every tensor fed to DigitCNN must be normalized with these exact constants
# because that's what the network was trained on.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics captured at the end of one MNIST training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    seconds: float
    overfit_gap: float


@dataclass(frozen=True)
class DigitRegion:
    """A cropped handwriting region plus its original page location."""

    image: Image.Image
    box: tuple[int, int, int, int]
    row: int


class DigitCNN(nn.Module):
    """Convolutional network trained on MNIST digits.

    Two conv blocks (32 then 64 filters), each followed by a 2x2 max-pool,
    take the 28x28 input down to a 7x7x64 feature map before the classifier
    head. Dropout increases with depth (0.10 -> 0.20 -> 0.35) since later,
    more abstract layers are more prone to overfitting on a small 10-class
    problem.
    """

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
    """Choose the fastest locally available PyTorch device.

    Preference order is Apple Silicon MPS, then CUDA, then CPU. `mps` is
    guarded with getattr because older torch builds don't expose
    `torch.backends.mps` at all, which would otherwise raise AttributeError.
    """

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_data_root() -> Path:
    """Find an existing MNIST data folder before falling back to project data."""

    for root in DEFAULT_DATA_ROOTS:
        if (root / "MNIST" / "raw").exists():
            return root
    return DEFAULT_DATA_ROOTS[0]


def build_loaders(data_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Create augmented training and plain test loaders for MNIST."""

    # Mild random affine jitter (rotation/translate/scale/shear) approximates
    # the natural variation in real handwriting so the model generalizes
    # beyond MNIST's fairly uniform, pre-centered digit style. The ranges are
    # deliberately small — MNIST digits are already well-formed, so
    # aggressive augmentation would create unrealistic training examples.
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
    """Evaluate the digit model and return average loss plus accuracy."""

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
    """Train the MNIST CNN and save the best checkpoint above the target."""

    # Fixed seed keeps training runs reproducible for comparing experiments.
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

        # Track the best-performing epoch's weights (not necessarily the
        # final epoch's) since cosine annealing and stochastic minibatches
        # mean test accuracy can dip after its peak before training ends.
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
    """Load the trained digit CNN from disk."""

    selected_device = device or get_device()
    checkpoint = torch.load(weights_path, map_location=selected_device, weights_only=True)
    model = DigitCNN().to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _foreground_from_image(image: Image.Image) -> np.ndarray:
    """Convert an image into a foreground-ink mask, handling light/dark ink.

    Uploaded photos can be dark ink on a light page or (less commonly) light
    chalk/pen on a dark background. We sample the image border pixels as a
    proxy for the background color: if the border is mostly bright, assume
    light background and invert so "ink" is always represented as high
    values. 0.18 is an empirical noise floor that discards faint scanner/
    camera artifacts without erasing genuine light pencil strokes.
    """

    grayscale = ImageOps.grayscale(image)
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
        return np.zeros((1, 1), dtype=np.float32)
    border = np.concatenate((array[0, :], array[-1, :], array[:, 0], array[:, -1]))
    if float(np.median(border)) > 0.5:
        array = 1.0 - array
    array[array < 0.18] = 0.0
    return array


def _group_boxes_by_reading_order(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Group boxes into rows so multi-line uploads read top-to-bottom.

    Rows aren't detected by fixed pixel bands (handwriting is rarely level)
    but by clustering boxes whose vertical centers are close to each other.
    The tolerance scales with the median glyph height (with an 8px floor for
    very small crops) so both tiny and large handwriting cluster sensibly.
    A box joins the first row whose running average center it's close
    enough to; this is a greedy single-pass clustering, not globally optimal,
    but is fast and good enough for the row counts typical of this app.
    """

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
    """Sort bounding boxes in human reading order."""

    return [box for row in _group_boxes_by_reading_order(boxes) for box in row]


def _best_vertical_cut(mask: np.ndarray, left: int, right: int) -> int:
    """Find the lowest-ink vertical seam for splitting connected digits.

    Sums ink pixels column-by-column (a vertical projection) across the
    search window; the column with the least ink is where two touching
    digits are most likely to meet. The projection is smoothed with a 7-wide
    moving average first so a single stray dark pixel doesn't create a false
    minimum. When several columns tie near the minimum (within 0.5 ink
    "units"), the one closest to the window's midpoint is preferred, which
    biases toward an even split rather than an edge cut.
    """

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
    """Reduce overlap after splitting so boxes map cleanly to characters.

    Splitting a wide box (see `_split_wide_box`) can leave adjacent boxes
    overlapping slightly. Small overlaps (<12% of the narrower box's width)
    are left alone since they're usually harmless bounding-box slop. Larger
    overlaps get a fresh ink-seam cut between them; for very large overlaps
    (>=35%) the cut is additionally capped near the left box's edge so we
    don't carve away too much of the right character's leading stroke. The
    `min_width` guard rejects cuts that would leave either side too thin to
    plausibly be its own character.
    """

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
    """Split boxes that are much wider than a normal single digit.

    Connected-component detection merges touching digits (e.g. "11" or "42"
    drawn without a gap) into one box. A box is only considered for splitting
    once its aspect ratio passes 1.45 (tall single digits are usually
    narrower than they are tall) to avoid mangling legitimately wide single
    glyphs. `0.62` approximates a typical single digit's width-to-height
    ratio, so `width / (height * 0.62)` estimates how many digits are packed
    into the box; this is then capped at 4 to keep runaway detections (e.g.
    from a long underline) from exploding into many tiny boxes. Cut points
    are searched near evenly-spaced expected positions (not blindly at the
    minimum ink column) so an accidental gap doesn't split off a sliver.
    """

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
            # Too little ink in this slice to be a real character; drop it
            # rather than emit a near-empty box downstream.
            continue
        split_boxes.append((left + int(xs.min()), y0 + int(ys.min()), left + int(xs.max()) + 1, y0 + int(ys.max()) + 1))
    return split_boxes or [box]


def _refine_digit_boxes(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Apply digit-specific box cleanup after connected-component detection."""

    split_boxes: list[tuple[int, int, int, int]] = []
    for box in boxes:
        split_boxes.extend(_split_wide_box(mask, box))
    return _trim_overlapping_boxes(mask, split_boxes)


def _box_ink_area(mask: np.ndarray, box: tuple[int, int, int, int]) -> int:
    """Count foreground pixels inside a bounding box.

    Used as a cheap proxy for "how substantial is this mark" when deciding
    whether a component is a real character stroke versus stray noise or a
    small mark like a dot.
    """

    x0, y0, x1, y1 = box
    return int(mask[y0:y1, x0:x1].sum())


def _gap_between_boxes(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    """Return the horizontal gap between two boxes."""

    first_x0, first_y0, first_x1, first_y1 = first
    second_x0, second_y0, second_x1, second_y1 = second
    horizontal_gap = max(first_x0 - second_x1, second_x0 - first_x1, 0)
    vertical_gap = max(first_y0 - second_y1, second_y0 - first_y1, 0)
    return float(np.hypot(horizontal_gap, vertical_gap))


def _merge_small_mark_boxes(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Attach dots and punctuation marks to the closest parent character.

    Connected-component detection treats an "i" dot, an apostrophe, or a
    detached exclamation-point dot as its own separate component. Left
    alone, these would become spurious extra predictions. This groups every
    "small" component (area below a threshold relative to the row's median
    character size, or physically tiny) into whichever nearby "large"
    component it's horizontally aligned with and vertically close to, so it
    rides along as part of that character's bounding box instead of being
    predicted on its own.
    """

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
        # A thin tall stroke (like the stem of an "i" or "l") can have a
        # small ink area but must never be treated as a mark itself, or it
        # would get absorbed into some other character instead of anchoring
        # its own dot.
        is_tall_stroke = height >= 28 and height / max(width, 1) >= 3.0
        is_small = area <= small_area_limit and not is_tall_stroke
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
            mark_is_small = (
                mark_area <= small_area_limit
                or min(mark_width, mark_height) <= 7
                or (mark_width <= 24 and mark_height <= 24)
            )
            if not mark_is_small:
                continue
            mark_center_x = (mx0 + mx1) / 2.0
            # A mark is only absorbed if it's roughly centered over the
            # parent character (horizontal_limit) and close enough not to
            # belong to a neighboring character instead (vertical_limit).
            # Both limits scale with the parent's own size so tiny dots near
            # tiny letters aren't held to the same absolute pixel tolerance
            # as marks near large characters.
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


def _merge_disconnected_character_parts(
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Join substantial overlapping parts of one hand-drawn character.

    Some single characters get drawn (or scanned) as multiple disconnected
    ink blobs that connected-component labeling can't join automatically —
    e.g. a pen lift mid-stroke, or a crossed "t"/"+" where the crossbar
    doesn't quite touch the stem. Two adjacent boxes are candidates for
    merging either when they're side-by-side with a small horizontal gap and
    strong vertical overlap ("horizontally_near_parts"), or stacked with a
    small vertical gap and strong horizontal overlap
    ("vertically_stacked_parts"). The `current_area >= 45` checks avoid
    pulling in tiny noise specks (those are handled separately by
    `_merge_small_mark_boxes`), and the merged aspect-ratio cap (<=1.25)
    prevents this from silently fusing two genuinely separate characters
    into one implausibly wide box.
    """

    if len(boxes) <= 1:
        return boxes

    merged_rows: list[tuple[int, int, int, int]] = []
    for row_boxes in _group_boxes_by_reading_order(boxes):
        merged_rows.extend(_merge_disconnected_character_parts_in_row(mask, sorted(row_boxes, key=lambda item: item[0])))
    return _sort_boxes_reading_order(_merge_vertically_stacked_character_parts(mask, merged_rows))


def _merge_vertically_stacked_character_parts(
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Join vertically stacked parts that row clustering separated."""

    if len(boxes) <= 1:
        return boxes

    ordered = sorted(boxes, key=lambda item: (item[1], item[0]))
    consumed: set[int] = set()
    merged: list[tuple[int, int, int, int]] = []
    for index, current in enumerate(ordered):
        if index in consumed:
            continue
        x0, y0, x1, y1 = current
        width = x1 - x0
        height = y1 - y0
        current_area = _box_ink_area(mask, current)
        for following_index in range(index + 1, len(ordered)):
            if following_index in consumed:
                continue
            following = ordered[following_index]
            nx0, ny0, nx1, ny1 = following
            next_width = nx1 - nx0
            next_height = ny1 - ny0
            vertical_gap = ny0 - y1
            close_vertical_gap = vertical_gap <= max(16, int(min(height, next_height) * 0.12))
            horizontal_overlap = min(x1, nx1) - max(x0, nx0)
            horizontal_overlap_ratio = horizontal_overlap / max(1, min(width, next_width))
            merged_width = max(x1, nx1) - min(x0, nx0)
            merged_height = max(y1, ny1) - min(y0, ny0)
            if (
                close_vertical_gap
                and horizontal_overlap_ratio >= 0.35
                and current_area >= 45
                and _box_ink_area(mask, following) >= 45
                and merged_width / max(merged_height, 1) <= 1.25
                and merged_height / max(height, next_height, 1) <= 1.65
            ):
                current = (min(x0, nx0), min(y0, ny0), max(x1, nx1), max(y1, ny1))
                consumed.add(following_index)
                x0, y0, x1, y1 = current
                width = x1 - x0
                height = y1 - y0
                current_area = _box_ink_area(mask, current)
        consumed.add(index)
        merged.append(current)
    return merged


def _merge_disconnected_character_parts_in_row(
    mask: np.ndarray,
    ordered: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Join substantial disconnected character parts within one visual row."""

    if len(ordered) <= 1:
        return ordered

    merged: list[tuple[int, int, int, int]] = []
    index = 0
    while index < len(ordered):
        current = ordered[index]
        while index + 1 < len(ordered):
            following = ordered[index + 1]
            x0, y0, x1, y1 = current
            nx0, ny0, nx1, ny1 = following
            width = x1 - x0
            height = y1 - y0
            next_width = nx1 - nx0
            next_height = ny1 - ny0
            horizontal_gap = nx0 - x1
            vertical_overlap = min(y1, ny1) - max(y0, ny0)
            overlap_ratio = vertical_overlap / max(1, min(height, next_height))
            horizontal_overlap = min(x1, nx1) - max(x0, nx0)
            horizontal_overlap_ratio = horizontal_overlap / max(1, min(width, next_width))
            vertical_gap = ny0 - y1
            merged_width = max(x1, nx1) - min(x0, nx0)
            merged_height = max(y1, ny1) - min(y0, ny0)
            current_area = _box_ink_area(mask, current)
            next_area = _box_ink_area(mask, following)
            gap_limit = max(10, int(min(width, next_width) * 0.4), int(min(height, next_height) * 0.28))

            horizontally_near_parts = (
                0 <= horizontal_gap <= gap_limit
                and overlap_ratio >= 0.55
                and merged_width / max(width, next_width, 1) <= 1.8
            )
            vertically_stacked_parts = (
                vertical_gap <= max(12, int(min(height, next_height) * 0.35))
                and horizontal_overlap_ratio >= 0.35
            )
            should_merge = (
                (horizontally_near_parts or vertically_stacked_parts)
                and current_area >= 45
                and next_area >= 45
                and merged_width / max(merged_height, 1) <= 1.25
            )
            if not should_merge:
                break
            current = (min(x0, nx0), min(y0, ny0), max(x1, nx1), max(y1, ny1))
            index += 1
        merged.append(current)
        index += 1
    return merged


def segment_digit_regions(
    image: Image.Image,
    split_wide: bool = True,
    min_component_pixels: int = 12,
    merge_marks: bool = False,
) -> list[DigitRegion]:
    """Segment an uploaded image into ordered handwriting regions.

    This is the entry point for turning one uploaded photo/scan into a list
    of per-character crops with page coordinates and row numbers, which the
    digit and character models then classify independently.
    """

    foreground = _foreground_from_image(image)
    mask = foreground > 0.18
    if int(mask.sum()) < 20:
        # Not enough ink anywhere in the image to be handwriting; bail out
        # rather than running expensive labeling on a near-blank page.
        return []

    # Dilating before labeling closes small gaps within a single character's
    # strokes (e.g. a loosely drawn "8") so connected-component analysis
    # doesn't fracture one glyph into multiple components. The *original*
    # (non-dilated) mask is still used below to compute tight bounding boxes,
    # so dilation only affects which pixels get grouped together, not the
    # box extents.
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
        boxes = _merge_disconnected_character_parts(mask, boxes)
        boxes = _merge_small_mark_boxes(mask, boxes)
    rows = _group_boxes_by_reading_order(boxes)
    digits: list[DigitRegion] = []
    for row_index, row in enumerate(rows, start=1):
        for box_index, (x0, y0, x1, y1) in enumerate(row):
            # Pad each box outward a bit (15% of its longer side) so the
            # crop includes a small ink-free margin, which the MNIST-style
            # normalization step expects. The pad is clamped against the
            # neighboring box in the same row so padding never eats into an
            # adjacent character.
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
            if x1 <= x0 or y1 <= y0:
                continue
            crop = (foreground[y0:y1, x0:x1] * 255).astype(np.uint8)
            digits.append(DigitRegion(image=Image.fromarray(crop, mode="L"), box=(x0, y0, x1, y1), row=row_index))
    return digits


def segment_digits(image: Image.Image) -> list[Image.Image]:
    """Return just the cropped digit images for legacy digit-only callers."""

    return [region.image for region in segment_digit_regions(image)]


def mnist_normalize_image(image: Image.Image) -> Image.Image:
    """Normalize one cropped digit to the centered 28x28 MNIST format.

    Mirrors the canonical MNIST preprocessing recipe: crop tightly to ink,
    scale so the longer side is 20px (leaving a ~4px border on all sides
    within the 28x28 canvas, matching the original dataset's convention),
    then recenter using the pixel-mass centroid rather than the bounding-box
    center. Centroid centering matters because a digit like "7" has most of
    its mass in the top stroke, so a plain bbox-center would leave it visibly
    off-center compared to how MNIST digits were actually prepared — and the
    model was trained on centroid-centered digits.
    """

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
    if canvas_array.sum() == 0:
        return canvas
    center = ndimage.center_of_mass(canvas_array)
    if not any(np.isnan(center)):
        # center_of_mass returns NaN when the canvas is entirely blank
        # (nothing to shift); skip the recenter in that edge case.
        shift_y = int(round(14 - center[0]))
        shift_x = int(round(14 - center[1]))
        shifted = ndimage.shift(canvas_array, shift=(shift_y, shift_x), order=1, mode="constant", cval=0)
        canvas = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8), mode="L")
    return canvas


def tensor_from_digit(image: Image.Image, device: torch.device, thicken: bool = False) -> torch.Tensor:
    """Convert a cropped digit image into a normalized model tensor.

    `thicken` runs a 2x2 max filter to bulk up thin/faint strokes before
    normalization — used as a retry variant for low-confidence predictions
    (see `_predict_digit_image`) since some handwriting is drawn with very
    light pressure that thins out during preprocessing.
    """

    normalized = mnist_normalize_image(image)
    array = np.asarray(normalized, dtype=np.float32) / 255.0
    if thicken:
        array = ndimage.maximum_filter(array, size=2)
    array = (array - MNIST_MEAN) / MNIST_STD
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def _predict_digit_tensor(model: nn.Module, tensor: torch.Tensor) -> tuple[int, float]:
    """Predict one normalized digit tensor."""

    probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    confidence, label = torch.max(probabilities, dim=0)
    return int(label.item()), float(confidence.item())


def _low_confidence_digit_variants(image: Image.Image) -> list[Image.Image]:
    """Generate simple crop variants for uncertain digit predictions.

    A common segmentation mistake is including a sliver of a neighboring
    digit's stroke at the right edge, which can confuse the classifier (e.g.
    "1" + a stray mark from the next digit looking like "4"). Only crops with
    aspect ratio >= 0.85 (i.e. not already narrow/tall) are trimmed, since
    narrow digits like "1" are unlikely to have this problem and trimming
    them further would cut into real ink.
    """

    variants = [image]
    width, height = image.size
    if height > 0 and width / height >= 0.85:
        for ratio in (0.88, 0.78):
            right = max(1, int(width * ratio))
            variants.append(image.crop((0, 0, right, height)))
    return variants


def _predict_digit_image(model: nn.Module, image: Image.Image, device: torch.device) -> tuple[int, float]:
    """Predict one cropped digit image, retrying variants when confidence is low.

    0.85 is treated as "confident enough" to skip the more expensive retry
    loop. Below that, every crop variant is tried both normal and
    "thickened" (see `tensor_from_digit`), and the single highest-confidence
    result across all of them wins — this is a brute-force ensemble over
    cheap preprocessing variations rather than a smarter model.
    """

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
    """Segment and predict every digit in an uploaded image."""

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
    """CLI entrypoint for MNIST digit training."""

    parser = argparse.ArgumentParser(description="Train and evaluate the MNIST digit CNN.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-accuracy", type=float, default=95.0)
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, min_accuracy=args.min_accuracy)


if __name__ == "__main__":
    main()
