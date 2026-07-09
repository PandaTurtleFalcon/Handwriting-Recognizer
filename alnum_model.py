"""Train and load the combined digit + alphabet recognizer.

This module joins MNIST digits with EMNIST letters into one 36-class dataset
(`0-9` and `A-Z`). The website still keeps older specialist models around for
hard edge cases, but this checkpoint is the main high-accuracy alphanumeric
model shown in the UI badge.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import datasets, transforms

from emnist_experiment import DATA_ROOT as EMNIST_DATA_ROOT
from emnist_experiment import EMNIST_MEAN, EMNIST_STD, EmnistCNN, EmnistMLP, TinyEmnistCNN, WideEmnistCNN, build_or_load_emnist_cache
from extra_alnum_datasets import load_labeled_image_folder
from mnist_model import get_device


PROJECT_DIR = Path(__file__).resolve().parent
MNIST_DATA_ROOT = PROJECT_DIR / "data" / "mnist"
USPS_DATA_ROOT = PROJECT_DIR / "data" / "usps"
NIST_SD19_DATA_ROOT = PROJECT_DIR / "data" / "nist_sd19"
NIST_SD19_BY_CLASS_URL = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
CORRECTIONS_PATH = PROJECT_DIR / "data" / "corrections" / "corrections.jsonl"
CORRECTION_UPLOAD_DIR = PROJECT_DIR / "data" / "corrections" / "uploads"
CHARS74K_DATA_ROOT = PROJECT_DIR / "data" / "chars74k"
CHARS74K_ENGLISH_HND_URL = "https://drive.google.com/uc?export=download&id=1FCH2jjo9Z1HbPVDWAEfvE3ZKaPpaXogg"
WEIGHTS_PATH = PROJECT_DIR / "alnum_cnn.pt"
METRICS_PATH = PROJECT_DIR / "alnum_training_metrics.json"
# Class index 0-9 are digits, 10-35 are A-Z. Case is folded to uppercase in
# this 36-class model; MIXEDCASE_LABELS below is the separate 62-class
# variant that keeps upper/lower distinct.
LABELS = [str(index) for index in range(10)] + [chr(ord("A") + index) for index in range(26)]
MIXEDCASE_WEIGHTS_PATH = PROJECT_DIR / "mixedcase_cnn.pt"
MIXEDCASE_METRICS_PATH = PROJECT_DIR / "mixedcase_training_metrics.json"
MIXEDCASE_LABELS = (
    [str(index) for index in range(10)]
    + [chr(ord("A") + index) for index in range(26)]
    + [chr(ord("a") + index) for index in range(26)]
)
MODEL_CLASSES = {
    "mlp": EmnistMLP,
    "tinycnn": TinyEmnistCNN,
    "cnn": EmnistCNN,
    "widecnn": WideEmnistCNN,
}


@dataclass(frozen=True)
class AlnumEpochMetrics:
    """Metrics captured at the end of each combined training epoch.

    digit_test_accuracy and letter_test_accuracy are tracked separately from
    the overall test_accuracy because a model can look good on aggregate
    while quietly underperforming on one of the two source domains (e.g.
    digits being much easier than letters).
    """

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


def _foreground_tensor_from_image(image: Image.Image) -> torch.Tensor:
    """Convert black-on-white or white-on-black glyphs to normalized tensors.

    Same border-sampling trick as `mnist_model._foreground_from_image`: the
    median of the border pixels stands in for the background color, and the
    image is inverted if that background reads as bright, so "ink" is always
    the high-value foreground regardless of the original scan's polarity.
    """

    grayscale = ImageOps.grayscale(image)
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    border = np.concatenate((array[0, :], array[-1, :], array[:, 0], array[:, -1]))
    if float(np.median(border)) > 0.5:
        array = 1.0 - array
    array[array < 0.18] = 0.0
    ys, xs = np.where(array > 0.18)
    if len(xs) > 0:
        array = array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # Scaled to 22px (vs. 20px in mnist_model's normalize) to match how this
    # dataset mix (MNIST + EMNIST + extras) was empirically found to align
    # best inside the 28x28 canvas across all source datasets.
    height, width = array.shape
    scale = 22.0 / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    glyph = Image.fromarray((array * 255).astype(np.uint8), mode="L").resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(glyph, ((28 - new_width) // 2, (28 - new_height) // 2))
    tensor_array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor_array = (tensor_array - EMNIST_MEAN) / EMNIST_STD
    return torch.from_numpy(tensor_array).unsqueeze(0)


def image_folder_transform(image: Image.Image) -> torch.Tensor:
    """Normalize local image-folder datasets like the website normalizes uploads."""

    return _foreground_tensor_from_image(image)


def build_or_load_usps_cache(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Load USPS digit tensors as an optional MNIST-compatible supplement."""

    cache_path = USPS_DATA_ROOT / f"cache_usps_{'train' if train else 'test'}.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cache["images"], cache["targets"]

    dataset = datasets.USPS(
        root=str(USPS_DATA_ROOT),
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,)),
            ]
        ),
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


def ensure_nist_sd19_by_class() -> Path:
    """Download the raw NIST SD19 by_class archive if it is missing."""

    archive_path = NIST_SD19_DATA_ROOT / "by_class.zip"
    if archive_path.exists():
        return archive_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(NIST_SD19_BY_CLASS_URL, timeout=300) as response:
        archive_path.write_bytes(response.read())
    return archive_path


def _nist_sd19_label_from_hex(hex_label: str) -> int | None:
    """Map a NIST SD19 hex ASCII folder name to MIXEDCASE_LABELS index."""

    try:
        label = chr(int(hex_label, 16))
    except ValueError:
        return None
    if label.isdigit():
        return int(label)
    if "A" <= label <= "Z":
        return 10 + ord(label) - ord("A")
    if "a" <= label <= "z":
        return 36 + ord(label) - ord("a")
    return None


def build_or_load_nist_sd19_mixedcase_cache(
    samples_per_class: int = 1200,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a balanced 62-class sample from raw NIST SD19 by_class.zip.

    The archive contains more than 1.5 million files, so this reads directly
    from the zip and caches only a deterministic balanced subset instead of
    extracting everything into the workspace.
    """

    cache_path = NIST_SD19_DATA_ROOT / f"cache_mixedcase_62_{samples_per_class}_seed{seed}.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cache["images"], cache["targets"]

    archive_path = ensure_nist_sd19_by_class()
    grouped: dict[int, list[str]] = {index: [] for index in range(len(MIXEDCASE_LABELS))}
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".png"):
                continue
            parts = name.split("/")
            if len(parts) < 4 or parts[0] != "by_class":
                continue
            target = _nist_sd19_label_from_hex(parts[1])
            if target is not None:
                grouped[target].append(name)

        generator = np.random.default_rng(seed)
        images = []
        targets = []
        for target, names in grouped.items():
            if not names:
                continue
            selected = generator.choice(
                np.array(names, dtype=object),
                size=min(samples_per_class, len(names)),
                replace=False,
            )
            for name in selected.tolist():
                with Image.open(io.BytesIO(archive.read(str(name)))) as image:
                    images.append(_foreground_tensor_from_image(image))
                targets.append(target)

    if not images:
        raise RuntimeError("No NIST SD19 mixed-case images were loaded.")

    image_tensor = torch.stack(images)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": image_tensor, "targets": target_tensor}, cache_path)
    return image_tensor, target_tensor


def load_correction_cache(
    labels: list[str],
    corrections_path: Path = CORRECTIONS_PATH,
    upload_dir: Path = CORRECTION_UPLOAD_DIR,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load saved per-character user corrections as training tensors.

    Only future character-level corrections with an `image_id` can train the
    model. Older records and whole-sequence corrections are skipped because
    they do not contain enough information to crop a specific glyph safely.
    """

    if not corrections_path.exists():
        return None
    label_to_index = {label: index for index, label in enumerate(labels)}
    images = []
    targets = []
    for line in corrections_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("correction_kind") != "character":
            continue
        corrected_label = str(record.get("corrected_label", ""))
        if len(corrected_label) != 1 or corrected_label not in label_to_index:
            continue
        image_id = str(record.get("image_id", ""))
        if not image_id:
            continue
        image_path = upload_dir / f"{image_id}.png"
        if not image_path.exists():
            continue
        bbox = record.get("bbox", {})
        if not isinstance(bbox, dict):
            continue
        try:
            x0 = max(0, int(round(float(bbox.get("x", 0)))))
            y0 = max(0, int(round(float(bbox.get("y", 0)))))
            width = max(1, int(round(float(bbox.get("width", 0)))))
            height = max(1, int(round(float(bbox.get("height", 0)))))
        except (TypeError, ValueError):
            continue
        with Image.open(image_path) as image:
            crop = ImageOps.exif_transpose(image).convert("RGB").crop((x0, y0, x0 + width, y0 + height))
            images.append(_foreground_tensor_from_image(crop))
        targets.append(label_to_index[corrected_label])

    if not images:
        return None
    return torch.stack(images), torch.tensor(targets, dtype=torch.long)


def _safe_extract_tgz(archive_path: Path, target_dir: Path) -> None:
    """Extract a tgz archive while blocking path traversal members.

    Guards against the classic "tarbomb" attack where a malicious archive
    member's name (e.g. "../../etc/passwd") would resolve outside
    `target_dir` during extraction. Every member's resolved destination is
    checked before any extraction happens, since standard `extractall` does
    not validate this on its own.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            destination = (target_dir / member.name).resolve()
            if not str(destination).startswith(str(target_dir.resolve())):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        archive.extractall(target_dir)


def ensure_chars74k_english_hnd() -> Path:
    """Download/extract Chars74K EnglishHnd and return its image root."""

    image_root = CHARS74K_DATA_ROOT / "English" / "Hnd" / "Img"
    if image_root.exists():
        return image_root

    archive_path = CHARS74K_DATA_ROOT / "EnglishHnd.tgz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        request = urllib.request.Request(CHARS74K_ENGLISH_HND_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            archive_path.write_bytes(response.read())
    _safe_extract_tgz(archive_path, CHARS74K_DATA_ROOT)
    if not image_root.exists():
        raise RuntimeError("Chars74K EnglishHnd archive did not contain the expected image folder.")
    return image_root


def _chars74k_sample_label(sample_dir: Path) -> int | None:
    """Map Chars74K Sample001..062 folders into existing 0-9/A-Z labels.

    Chars74K's EnglishHnd layout uses 62 numbered sample folders in a fixed
    order: 001-010 are digits 0-9, 011-036 are uppercase A-Z, and 037-062 are
    lowercase a-z. Since this model folds case (36 classes, not 62), both the
    uppercase and lowercase ranges map onto the same 10-35 target indices —
    Sample011 and Sample037 (A and a) both become label index 10.
    """

    try:
        sample_number = int(sample_dir.name.replace("Sample", ""))
    except ValueError:
        return None
    if 1 <= sample_number <= 10:
        return sample_number - 1
    if 11 <= sample_number <= 36:
        return 10 + sample_number - 11
    if 37 <= sample_number <= 62:
        return 10 + sample_number - 37
    return None


def build_or_load_chars74k_cache() -> tuple[torch.Tensor, torch.Tensor]:
    """Load Chars74K EnglishHnd tensors folded into the 36 model classes."""

    cache_path = CHARS74K_DATA_ROOT / "cache_english_hnd_36.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cache["images"], cache["targets"]

    image_root = ensure_chars74k_english_hnd()
    images = []
    targets = []
    for sample_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        target = _chars74k_sample_label(sample_dir)
        if target is None:
            continue
        for image_path in sorted(sample_dir.glob("*.png")):
            with Image.open(image_path) as image:
                images.append(_foreground_tensor_from_image(image))
            targets.append(target)
    if not images:
        raise RuntimeError("No Chars74K images were loaded.")

    image_tensor = torch.stack(images)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": image_tensor, "targets": target_tensor}, cache_path)
    return image_tensor, target_tensor


def _chars74k_sample_mixedcase_label(sample_dir: Path) -> int | None:
    """Map Chars74K Sample001..062 folders into 0-9/A-Z/a-z labels."""

    try:
        sample_number = int(sample_dir.name.replace("Sample", ""))
    except ValueError:
        return None
    if 1 <= sample_number <= 10:
        return sample_number - 1
    if 11 <= sample_number <= 36:
        return 10 + sample_number - 11
    if 37 <= sample_number <= 62:
        return 36 + sample_number - 37
    return None


def build_or_load_chars74k_mixedcase_cache() -> tuple[torch.Tensor, torch.Tensor]:
    """Load Chars74K EnglishHnd tensors as distinct digit/upper/lower labels."""

    cache_path = CHARS74K_DATA_ROOT / "cache_english_hnd_62.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cache["images"], cache["targets"]

    image_root = ensure_chars74k_english_hnd()
    images = []
    targets = []
    for sample_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        target = _chars74k_sample_mixedcase_label(sample_dir)
        if target is None:
            continue
        for image_path in sorted(sample_dir.glob("*.png")):
            with Image.open(image_path) as image:
                images.append(_foreground_tensor_from_image(image))
            targets.append(target)
    if not images:
        raise RuntimeError("No Chars74K mixed-case images were loaded.")

    image_tensor = torch.stack(images)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": image_tensor, "targets": target_tensor}, cache_path)
    return image_tensor, target_tensor


def build_or_load_emnist_byclass_folded_cache(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Load EMNIST ByClass, folding lowercase samples into uppercase labels.

    ByClass has 62 raw classes (digits + upper + lower); this collapses case
    so 'a' and 'A' samples both train the same 36-class label used by the
    combined digit/letter model, giving it more real-world lowercase
    handwriting examples without needing a 62-class output layer.
    """

    cache_path = EMNIST_DATA_ROOT / f"cache_byclass_folded_36_{'train' if train else 'test'}.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cache["images"], cache["targets"]

    images, raw_targets, raw_labels = build_or_load_emnist_cache("byclass", train=train)
    folded_targets = []
    kept_indices = []
    for index, raw_target in enumerate(raw_targets.tolist()):
        label = raw_labels[int(raw_target)]
        if label.isdigit():
            folded_targets.append(int(label))
            kept_indices.append(index)
        elif len(label) == 1 and label.isalpha():
            folded_targets.append(10 + ord(label.upper()) - ord("A"))
            kept_indices.append(index)

    if not kept_indices:
        raise RuntimeError("EMNIST ByClass cache did not contain alphanumeric labels.")

    index_tensor = torch.tensor(kept_indices, dtype=torch.long)
    image_tensor = images[index_tensor]
    target_tensor = torch.tensor(folded_targets, dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": image_tensor, "targets": target_tensor}, cache_path)
    return image_tensor, target_tensor


def build_or_load_emnist_byclass_mixedcase_cache(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Load EMNIST ByClass as 0-9/A-Z/a-z without folding case."""

    cache_path = EMNIST_DATA_ROOT / f"cache_byclass_mixedcase_62_{'train' if train else 'test'}.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cache["images"], cache["targets"]

    images, raw_targets, raw_labels = build_or_load_emnist_cache("byclass", train=train)
    # ByClass's own label ordering happens to already start with digits then
    # uppercase then lowercase, matching MIXEDCASE_LABELS exactly. This check
    # fails loudly if torchvision ever changes that ordering, since silently
    # training on mismatched labels would produce a model that looks fine
    # metrically but predicts the wrong character.
    if raw_labels[: len(MIXEDCASE_LABELS)] != MIXEDCASE_LABELS:
        raise RuntimeError("Unexpected EMNIST ByClass label order; refusing to train mixed-case model.")
    mask = raw_targets < len(MIXEDCASE_LABELS)
    image_tensor = images[mask]
    target_tensor = raw_targets[mask].long()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": image_tensor, "targets": target_tensor}, cache_path)
    return image_tensor, target_tensor


def _limit_per_class(
    images: torch.Tensor,
    targets: torch.Tensor,
    samples_per_class: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic class-balanced subset for faster experiments.

    Capping samples per class (rather than a flat overall subset) keeps rare
    classes from being starved out, which would happen with a naive random
    subsample when class sizes are imbalanced. A seeded generator keeps the
    selection reproducible across runs with the same seed.
    """

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
    extra_train_dir: Path | None = None,
    extra_test_dir: Path | None = None,
    include_emnist_byclass: bool = False,
    include_chars74k: bool = False,
    include_usps: bool = False,
    include_corrections: bool = False,
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

    train_parts = [
        TensorDataset(mnist_train_images, mnist_train_targets),
        TensorDataset(letter_train_images, letter_train_targets),
    ]
    test_parts = [
        TensorDataset(mnist_test_images, mnist_test_targets),
        TensorDataset(letter_test_images, letter_test_targets),
    ]
    train_target_parts = [mnist_train_targets, letter_train_targets]
    if include_emnist_byclass:
        byclass_train_images, byclass_train_targets = build_or_load_emnist_byclass_folded_cache(train=True)
        byclass_train_images, byclass_train_targets = _limit_per_class(
            byclass_train_images,
            byclass_train_targets,
            samples_per_class,
            seed + 2,
        )
        train_parts.append(TensorDataset(byclass_train_images, byclass_train_targets))
        train_target_parts.append(byclass_train_targets)
    if include_chars74k:
        chars_images, chars_targets = build_or_load_chars74k_cache()
        train_parts.append(TensorDataset(chars_images, chars_targets))
        train_target_parts.append(chars_targets)
    if include_usps:
        usps_train_images, usps_train_targets = build_or_load_usps_cache(train=True)
        train_parts.append(TensorDataset(usps_train_images, usps_train_targets))
        train_target_parts.append(usps_train_targets)
    if include_corrections:
        corrections = load_correction_cache(LABELS)
        if corrections is not None:
            correction_images, correction_targets = corrections
            train_parts.append(TensorDataset(correction_images, correction_targets))
            train_target_parts.append(correction_targets)
    if extra_train_dir is not None:
        extra_train_images, extra_train_targets = load_labeled_image_folder(extra_train_dir, LABELS, image_folder_transform)
        train_parts.append(TensorDataset(extra_train_images, extra_train_targets))
        train_target_parts.append(extra_train_targets)
    if extra_test_dir is not None:
        extra_test_images, extra_test_targets = load_labeled_image_folder(extra_test_dir, LABELS, image_folder_transform)
        test_parts.append(TensorDataset(extra_test_images, extra_test_targets))

    train_dataset = ConcatDataset(train_parts)
    test_dataset = ConcatDataset(test_parts)
    train_targets = torch.cat(train_target_parts).numpy()
    # Mixing several source datasets (MNIST, EMNIST letters, optionally
    # ByClass/Chars74K/USPS/extra folders) means class counts are naturally
    # unbalanced. Inverse-frequency sample weights make the WeightedRandomSampler
    # draw rare classes proportionally more often so training doesn't end up
    # biased toward whichever dataset happens to be largest.
    class_counts = np.bincount(train_targets, minlength=len(LABELS))
    sample_weights = [1.0 / max(class_counts[int(target)], 1) for target in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_targets), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    digit_test_loader = DataLoader(TensorDataset(mnist_test_images, mnist_test_targets), batch_size=batch_size)
    letter_test_loader = DataLoader(TensorDataset(letter_test_images, letter_test_targets), batch_size=batch_size)
    return train_loader, test_loader, digit_test_loader, letter_test_loader


def make_augmented_loaders(
    batch_size: int,
    extra_train_dir: Path | None = None,
    extra_test_dir: Path | None = None,
    include_emnist_byclass: bool = False,
    include_chars74k: bool = False,
    include_usps: bool = False,
    include_corrections: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
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

    train_parts = [mnist_train, letter_train]
    test_parts = [mnist_test, letter_test]
    train_target_parts = [
        torch.as_tensor(mnist_train.targets, dtype=torch.long),
        letter_train_targets,
    ]
    if include_emnist_byclass:
        byclass_train_images, byclass_train_targets = build_or_load_emnist_byclass_folded_cache(train=True)
        train_parts.append(TensorDataset(byclass_train_images, byclass_train_targets))
        train_target_parts.append(byclass_train_targets)
    if include_chars74k:
        chars_images, chars_targets = build_or_load_chars74k_cache()
        train_parts.append(TensorDataset(chars_images, chars_targets))
        train_target_parts.append(chars_targets)
    if include_usps:
        usps_train_images, usps_train_targets = build_or_load_usps_cache(train=True)
        train_parts.append(TensorDataset(usps_train_images, usps_train_targets))
        train_target_parts.append(usps_train_targets)
    if include_corrections:
        corrections = load_correction_cache(LABELS)
        if corrections is not None:
            correction_images, correction_targets = corrections
            train_parts.append(TensorDataset(correction_images, correction_targets))
            train_target_parts.append(correction_targets)
    if extra_train_dir is not None:
        extra_train_images, extra_train_targets = load_labeled_image_folder(
            extra_train_dir,
            LABELS,
            image_folder_transform,
        )
        train_parts.append(TensorDataset(extra_train_images, extra_train_targets))
        train_target_parts.append(extra_train_targets)
    if extra_test_dir is not None:
        extra_test_images, extra_test_targets = load_labeled_image_folder(extra_test_dir, LABELS, image_folder_transform)
        test_parts.append(TensorDataset(extra_test_images, extra_test_targets))

    train_dataset = ConcatDataset(train_parts)
    test_dataset = ConcatDataset(test_parts)
    train_targets = torch.cat(train_target_parts).numpy()
    class_counts = np.bincount(train_targets, minlength=len(LABELS))
    sample_weights = [1.0 / max(class_counts[int(target)], 1) for target in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_targets), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    digit_test_loader = DataLoader(mnist_test, batch_size=batch_size)
    letter_test_loader = DataLoader(letter_test, batch_size=batch_size)
    return train_loader, test_loader, digit_test_loader, letter_test_loader


def _target_range_loader(images: torch.Tensor, targets: torch.Tensor, start: int, stop: int, batch_size: int) -> DataLoader:
    """Build a loader for a contiguous target range.

    Used to carve out just the upper-case (10-35) or lower-case (36-61)
    slice of MIXEDCASE_LABELS so per-case accuracy can be tracked separately
    during training, since overall accuracy alone can hide one case being
    much worse than the other.
    """

    mask = (targets >= start) & (targets < stop)
    return DataLoader(TensorDataset(images[mask], targets[mask]), batch_size=batch_size)


def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    labels: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Return per-label accuracy percentages for a loader."""

    model.eval()
    correct = torch.zeros(len(labels), dtype=torch.long)
    total = torch.zeros(len(labels), dtype=torch.long)
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()
            targets_cpu = targets.cpu()
            for label_index in range(len(labels)):
                mask = targets_cpu == label_index
                if not bool(mask.any()):
                    continue
                total[label_index] += int(mask.sum().item())
                correct[label_index] += int((predictions[mask] == targets_cpu[mask]).sum().item())
    return {
        label: 100.0 * int(correct[index].item()) / max(int(total[index].item()), 1)
        for index, label in enumerate(labels)
        if int(total[index].item()) > 0
    }


def make_mixedcase_loaders(
    batch_size: int,
    samples_per_class: int | None,
    seed: int,
    include_chars74k: bool = False,
    include_usps: bool = False,
    include_nist_sd19: bool = False,
    nist_samples_per_class: int = 1200,
    include_corrections: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """Build loaders for the 62-class digit/upper/lower recognizer."""

    mnist_train_images, mnist_train_targets = build_or_load_mnist_cache(train=True)
    mnist_test_images, mnist_test_targets = build_or_load_mnist_cache(train=False)
    byclass_train_images, byclass_train_targets = build_or_load_emnist_byclass_mixedcase_cache(train=True)
    byclass_test_images, byclass_test_targets = build_or_load_emnist_byclass_mixedcase_cache(train=False)

    mnist_train_images, mnist_train_targets = _limit_per_class(
        mnist_train_images,
        mnist_train_targets,
        samples_per_class,
        seed,
    )
    byclass_train_images, byclass_train_targets = _limit_per_class(
        byclass_train_images,
        byclass_train_targets,
        samples_per_class,
        seed + 1,
    )

    train_parts = [
        TensorDataset(mnist_train_images, mnist_train_targets),
        TensorDataset(byclass_train_images, byclass_train_targets),
    ]
    train_target_parts = [mnist_train_targets, byclass_train_targets]
    if include_chars74k:
        chars_images, chars_targets = build_or_load_chars74k_mixedcase_cache()
        train_parts.append(TensorDataset(chars_images, chars_targets))
        train_target_parts.append(chars_targets)
    if include_usps:
        usps_train_images, usps_train_targets = build_or_load_usps_cache(train=True)
        train_parts.append(TensorDataset(usps_train_images, usps_train_targets))
        train_target_parts.append(usps_train_targets)
    if include_nist_sd19:
        nist_images, nist_targets = build_or_load_nist_sd19_mixedcase_cache(nist_samples_per_class, seed)
        train_parts.append(TensorDataset(nist_images, nist_targets))
        train_target_parts.append(nist_targets)
    if include_corrections:
        corrections = load_correction_cache(list(MIXEDCASE_LABELS))
        if corrections is not None:
            correction_images, correction_targets = corrections
            train_parts.append(TensorDataset(correction_images, correction_targets))
            train_target_parts.append(correction_targets)

    train_dataset = ConcatDataset(train_parts)
    test_dataset = ConcatDataset(
        [
            TensorDataset(mnist_test_images, mnist_test_targets),
            TensorDataset(byclass_test_images, byclass_test_targets),
        ]
    )
    train_targets = torch.cat(train_target_parts).numpy()
    class_counts = np.bincount(train_targets, minlength=len(MIXEDCASE_LABELS))
    sample_weights = [1.0 / max(class_counts[int(target)], 1) for target in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_targets), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    digit_test_loader = DataLoader(TensorDataset(mnist_test_images, mnist_test_targets), batch_size=batch_size)
    upper_test_loader = _target_range_loader(byclass_test_images, byclass_test_targets, 10, 36, batch_size)
    lower_test_loader = _target_range_loader(byclass_test_images, byclass_test_targets, 36, 62, batch_size)
    return train_loader, test_loader, digit_test_loader, upper_test_loader, lower_test_loader


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
    extra_train_dir: Path | None = None,
    extra_test_dir: Path | None = None,
    include_emnist_byclass: bool = False,
    include_chars74k: bool = False,
    include_usps: bool = False,
    include_corrections: bool = False,
    warm_start: bool = False,
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
                "extra_train_dir": str(extra_train_dir) if extra_train_dir is not None else None,
                "extra_test_dir": str(extra_test_dir) if extra_test_dir is not None else None,
                "include_emnist_byclass": include_emnist_byclass,
                "include_chars74k": include_chars74k,
                "include_usps": include_usps,
                "include_corrections": include_corrections,
                "warm_start": warm_start,
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
                "extra_train_dir": str(extra_train_dir) if extra_train_dir is not None else None,
                "extra_test_dir": str(extra_test_dir) if extra_test_dir is not None else None,
                "include_emnist_byclass": include_emnist_byclass,
                "include_chars74k": include_chars74k,
                "include_usps": include_usps,
                "include_corrections": include_corrections,
                "warm_start": warm_start,
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


def load_mixedcase_model(
    weights_path: Path = MIXEDCASE_WEIGHTS_PATH,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[str]] | tuple[None, None]:
    """Load the trained mixed-case recognizer if its checkpoint exists."""

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


def save_mixedcase_checkpoint(
    history: list[dict[str, float | int]],
    best_state: dict[str, torch.Tensor] | None,
    best_accuracy: float,
    best_metrics: dict[str, float | str] | None,
    model_type: str,
    learning_rate: float,
    seed: int,
    device: torch.device,
    samples_per_class: int | None,
    include_chars74k: bool = False,
    include_usps: bool = False,
    include_nist_sd19: bool = False,
    nist_samples_per_class: int = 1200,
    include_corrections: bool = False,
    warm_start: bool = False,
    per_class_accuracy: dict[str, float] | None = None,
) -> None:
    """Persist the best mixed-case weights and metrics."""

    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "labels": MIXEDCASE_LABELS,
                "test_accuracy": best_accuracy,
                "model_type": model_type,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": str(device),
                "samples_per_class": samples_per_class,
                "include_chars74k": include_chars74k,
                "include_usps": include_usps,
                "include_nist_sd19": include_nist_sd19,
                "nist_samples_per_class": nist_samples_per_class,
                "include_corrections": include_corrections,
                "warm_start": warm_start,
                "normalization": {"mean": EMNIST_MEAN, "std": EMNIST_STD},
            },
            MIXEDCASE_WEIGHTS_PATH,
        )
    MIXEDCASE_METRICS_PATH.write_text(
        json.dumps(
            {
                "labels": MIXEDCASE_LABELS,
                "model_type": model_type,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": str(device),
                "samples_per_class": samples_per_class,
                "include_chars74k": include_chars74k,
                "include_usps": include_usps,
                "include_nist_sd19": include_nist_sd19,
                "nist_samples_per_class": nist_samples_per_class,
                "include_corrections": include_corrections,
                "warm_start": warm_start,
                "per_class_accuracy": per_class_accuracy or {},
                "best_checkpoint": best_metrics or {"test_accuracy": best_accuracy},
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def train_mixedcase(
    epochs: int,
    batch_size: int,
    min_accuracy: float,
    learning_rate: float,
    seed: int,
    model_type: str,
    samples_per_class: int | None,
    device_name: str,
    include_chars74k: bool = False,
    include_usps: bool = False,
    include_nist_sd19: bool = False,
    nist_samples_per_class: int = 1200,
    include_corrections: bool = False,
    warm_start: bool = False,
) -> list[dict[str, float | int]]:
    """Train a 62-class recognizer that distinguishes uppercase and lowercase."""

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

    train_loader, test_loader, digit_test_loader, upper_test_loader, lower_test_loader = make_mixedcase_loaders(
        batch_size,
        samples_per_class,
        seed,
        include_chars74k,
        include_usps,
        include_nist_sd19,
        nist_samples_per_class,
        include_corrections,
    )
    model = MODEL_CLASSES[model_type](num_classes=len(MIXEDCASE_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    if warm_start and MIXEDCASE_WEIGHTS_PATH.exists():
        checkpoint = torch.load(MIXEDCASE_WEIGHTS_PATH, map_location=device, weights_only=True)
        if list(checkpoint.get("labels", [])) == list(MIXEDCASE_LABELS) and checkpoint.get("model_type", "cnn") == model_type:
            model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float | int]] = []
    best_accuracy = 0.0
    best_state = None
    best_per_class_accuracy: dict[str, float] = {}
    best_metrics: dict[str, float | str] | None = None
    if warm_start:
        best_loss, best_accuracy = evaluate(model, test_loader, criterion, device)
        _, best_digit_accuracy = evaluate(model, digit_test_loader, criterion, device)
        _, best_upper_accuracy = evaluate(model, upper_test_loader, criterion, device)
        _, best_lower_accuracy = evaluate(model, lower_test_loader, criterion, device)
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        best_per_class_accuracy = evaluate_per_class(model, test_loader, list(MIXEDCASE_LABELS), device)
        best_metrics = {
            "test_loss": best_loss,
            "test_accuracy": best_accuracy,
            "digit_test_accuracy": best_digit_accuracy,
            "upper_test_accuracy": best_upper_accuracy,
            "lower_test_accuracy": best_lower_accuracy,
            "source": "warm_start_seed",
        }

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
        _, upper_accuracy = evaluate(model, upper_test_loader, criterion, device)
        _, lower_accuracy = evaluate(model, lower_test_loader, criterion, device)
        train_accuracy = 100.0 * train_correct / max(train_total, 1)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss_total / max(len(train_loader), 1),
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "digit_test_accuracy": digit_accuracy,
            "upper_test_accuracy": upper_accuracy,
            "lower_test_accuracy": lower_accuracy,
            "seconds": time.time() - start,
            "overfit_gap": train_accuracy - test_accuracy,
        }
        history.append(metrics)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_per_class_accuracy = evaluate_per_class(model, test_loader, list(MIXEDCASE_LABELS), device)
            best_metrics = {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "digit_test_accuracy": digit_accuracy,
                "upper_test_accuracy": upper_accuracy,
                "lower_test_accuracy": lower_accuracy,
                "source": f"epoch_{epoch}",
            }
        save_mixedcase_checkpoint(
            history,
            best_state,
            best_accuracy,
            best_metrics,
            model_type,
            learning_rate,
            seed,
            device,
            samples_per_class,
            include_chars74k,
            include_usps,
            include_nist_sd19,
            nist_samples_per_class,
            include_corrections,
            warm_start,
            best_per_class_accuracy,
        )
        print(
            f"Epoch {epoch}/{epochs} train_acc={train_accuracy:.2f}% "
            f"test_acc={test_accuracy:.2f}% digits={digit_accuracy:.2f}% "
            f"upper={upper_accuracy:.2f}% lower={lower_accuracy:.2f}% "
            f"gap={metrics['overfit_gap']:.2f}%",
            flush=True,
        )
        # Early-stop once the target accuracy is hit, but only after epoch 8
        # so an early lucky epoch (before the cosine schedule has annealed
        # the learning rate down) doesn't cut training short prematurely.
        if best_accuracy >= min_accuracy and epoch >= 8:
            break

    if best_accuracy < min_accuracy:
        raise RuntimeError(f"Best mixed-case test accuracy was {best_accuracy:.2f}%, below {min_accuracy:.2f}%.")
    return history


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
    extra_train_dir: Path | None = None,
    extra_test_dir: Path | None = None,
    include_emnist_byclass: bool = False,
    include_chars74k: bool = False,
    include_usps: bool = False,
    include_corrections: bool = False,
    warm_start: bool = False,
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

    loaders = (
        make_augmented_loaders(
            batch_size,
            extra_train_dir,
            extra_test_dir,
            include_emnist_byclass,
            include_chars74k,
            include_usps,
            include_corrections,
        )
        if augment
        else make_cached_loaders(
            batch_size,
            samples_per_class,
            seed,
            extra_train_dir,
            extra_test_dir,
            include_emnist_byclass,
            include_chars74k,
            include_usps,
            include_corrections,
        )
    )
    train_loader, test_loader, digit_test_loader, letter_test_loader = loaders
    model = MODEL_CLASSES[model_type](num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    # Warm-starting continues training from the existing checkpoint instead
    # of random init, useful for fine-tuning on new data (e.g. after adding
    # --include-chars74k) without losing prior accuracy. Only applied if the
    # checkpoint's label set and architecture match exactly, since loading
    # mismatched state_dicts would silently corrupt the model.
    if warm_start and WEIGHTS_PATH.exists():
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        if checkpoint.get("labels") == LABELS and checkpoint.get("model_type", "cnn") == model_type:
            model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: list[AlnumEpochMetrics] = []
    best_accuracy = 0.0
    best_state = None
    if warm_start:
        # Seed "best" with the warm-started model's own pre-training-loop
        # accuracy so a fresh run of fine-tuning epochs can't overwrite a
        # already-good checkpoint with a worse one if this run regresses.
        _, best_accuracy = evaluate(model, test_loader, criterion, device)
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

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
        save_checkpoint(
            history,
            best_state,
            best_accuracy,
            model_type,
            augment,
            learning_rate,
            seed,
            device,
            extra_train_dir,
            extra_test_dir,
            include_emnist_byclass,
            include_chars74k,
            include_usps,
            include_corrections,
            warm_start,
        )
        print(
            f"Epoch {epoch}/{epochs} train_acc={metrics.train_accuracy:.2f}% "
            f"test_acc={metrics.test_accuracy:.2f}% digits={digit_accuracy:.2f}% "
            f"letters={letter_accuracy:.2f}% gap={metrics.overfit_gap:.2f}%",
            flush=True,
        )
        # Early-stop once the target accuracy is hit, but only after epoch 8
        # so an early lucky epoch (before the cosine schedule has annealed
        # the learning rate down) doesn't cut training short prematurely.
        if best_accuracy >= min_accuracy and epoch >= 8:
            break

    if best_accuracy < min_accuracy:
        raise RuntimeError(f"Best combined test accuracy was {best_accuracy:.2f}%, below {min_accuracy:.2f}%.")
    return history


def main() -> None:
    """CLI entrypoint for training the combined recognizer."""

    parser = argparse.ArgumentParser(description="Train a combined MNIST + EMNIST alphanumeric recognizer.")
    parser.add_argument("--mixed-case", action="store_true", help="Train the 62-class mixed-case recognizer.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--min-accuracy", type=float, default=95.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--model", choices=["mlp", "tinycnn", "cnn", "widecnn"], default="cnn")
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--extra-train-dir",
        type=Path,
        default=None,
        help="Optional image-folder dataset to append to alphanumeric training data.",
    )
    parser.add_argument(
        "--extra-test-dir",
        type=Path,
        default=None,
        help="Optional image-folder dataset to append to alphanumeric test data.",
    )
    parser.add_argument(
        "--include-emnist-byclass",
        action="store_true",
        help="Append folded EMNIST ByClass samples to the training set.",
    )
    parser.add_argument(
        "--include-chars74k",
        action="store_true",
        help="Download and append Chars74K EnglishHnd samples to the training set.",
    )
    parser.add_argument(
        "--include-usps",
        action="store_true",
        help="Append USPS digit samples to the training set.",
    )
    parser.add_argument(
        "--include-corrections",
        action="store_true",
        help="Append saved per-character user corrections from data/corrections.",
    )
    parser.add_argument(
        "--include-nist-sd19",
        action="store_true",
        help="Append a sampled raw NIST SD19 by_class subset to mixed-case training.",
    )
    parser.add_argument(
        "--nist-samples-per-class",
        type=int,
        default=1200,
        help="Number of raw NIST SD19 samples to cache per class when --include-nist-sd19 is used.",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Initialize from the existing alphanumeric checkpoint before training.",
    )
    args = parser.parse_args()
    if args.mixed_case:
        train_mixedcase(
            epochs=args.epochs,
            batch_size=args.batch_size,
            min_accuracy=args.min_accuracy,
            learning_rate=args.learning_rate,
            seed=args.seed,
            model_type=args.model,
            samples_per_class=args.samples_per_class,
            device_name=args.device,
            include_chars74k=args.include_chars74k,
            include_usps=args.include_usps,
            include_nist_sd19=args.include_nist_sd19,
            nist_samples_per_class=args.nist_samples_per_class,
            include_corrections=args.include_corrections,
            warm_start=args.warm_start,
        )
        return
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
        extra_train_dir=args.extra_train_dir,
        extra_test_dir=args.extra_test_dir,
        include_emnist_byclass=args.include_emnist_byclass,
        include_chars74k=args.include_chars74k,
        include_usps=args.include_usps,
        include_corrections=args.include_corrections,
        warm_start=args.warm_start,
    )


if __name__ == "__main__":
    main()
