"""Optional dataset adapters for alphanumeric training experiments.

This module keeps non-core datasets outside the default MNIST+EMNIST training
path. The current adapter accepts a local image-folder dataset whose immediate
child folders are class labels from the combined 36-class label set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_labeled_image_folder(
    root: Path,
    labels: list[str],
    transform: Callable[[Image.Image], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a local image-folder dataset into tensors matching model labels.

    The expected layout is:

    data_root/
      0/*.png
      1/*.png
      A/*.png
      Z/*.png

    Folder names are matched case-insensitively against ``labels``. Unknown
    class folders fail fast so accidental data layout issues do not silently
    corrupt target indices.
    """

    if not root.exists():
        raise RuntimeError(f"Extra dataset directory does not exist: {root}")
    if not root.is_dir():
        raise RuntimeError(f"Extra dataset path is not a directory: {root}")

    # Case-insensitive lookup so folders can be named "a" or "A" and still map
    # onto the model's canonical (usually uppercase) label list.
    label_to_index = {label.upper(): index for index, label in enumerate(labels)}
    image_paths: list[tuple[Path, int]] = []
    unknown_classes: list[str] = []

    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label_key = class_dir.name.upper()
        if label_key not in label_to_index:
            # Collect rather than raise immediately so the error message can
            # report every bad folder in one pass instead of just the first.
            unknown_classes.append(class_dir.name)
            continue
        target = label_to_index[label_key]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append((image_path, target))

    if unknown_classes:
        formatted = ", ".join(sorted(unknown_classes))
        raise RuntimeError(f"Extra dataset contains unsupported class folders: {formatted}")
    if not image_paths:
        raise RuntimeError(f"Extra dataset contains no supported images: {root}")

    images = []
    targets = []
    for image_path, target in image_paths:
        with Image.open(image_path) as image:
            # Convert to single-channel grayscale before the caller-supplied
            # transform so this matches how uploaded images are preprocessed.
            images.append(transform(image.convert("L")))
        targets.append(target)

    return torch.stack(images), torch.tensor(targets, dtype=torch.long)
