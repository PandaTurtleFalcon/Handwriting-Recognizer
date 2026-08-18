"""Prepare T-H-E Dataset Version IV as a mixed-case tensor cache.

The upstream English CSV stores one sample per row as:

    class_id, pixel_0, pixel_1, ... pixel_783

Class ids follow the dataset paper/README: lower-case English letters are
1-26 and upper-case English letters are 40-65. This script maps those ids into
the project's 62-class mixed-case label order and writes an ignored `.pt`
cache that `alnum_model.load_mixedcase_extra_cache()` can consume.
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path
from collections.abc import Callable, Iterable

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import MIXEDCASE_LABELS, _foreground_tensor_from_image  # noqa: E402


DEFAULT_SOURCE_URL = "https://media.githubusercontent.com/media/bartosgaye/thedataset/master/version4.csv"
DEFAULT_RAW_PATH = PROJECT_DIR / "data" / "the_dataset" / "version4.csv"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "data" / "the_dataset" / "the_version4_mixedcase.pt"
ORIENTATION_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "transpose": lambda pixels: pixels.T,
    "raw": lambda pixels: pixels,
    "flipud": np.flipud,
    "fliplr": np.fliplr,
    "rot90": lambda pixels: np.rot90(pixels, 1),
    "rot270": lambda pixels: np.rot90(pixels, 3),
}


def the_label_to_mixedcase_index(class_id: int) -> int | None:
    """Map a T-H-E English class id to the project's mixed-case label index."""

    if 1 <= class_id <= 26:
        label = chr(ord("a") + class_id - 1)
    elif 40 <= class_id <= 65:
        label = chr(ord("A") + class_id - 40)
    else:
        return None
    return MIXEDCASE_LABELS.index(label)


def parse_the_rows(
    rows: Iterable[list[str]],
    limit: int | None = None,
    orientation: str = "transpose",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert CSV rows into normalized image and target tensors."""

    if orientation not in ORIENTATION_TRANSFORMS:
        choices = ", ".join(sorted(ORIENTATION_TRANSFORMS))
        raise ValueError(f"Unsupported orientation {orientation!r}; choose one of: {choices}")
    transform = ORIENTATION_TRANSFORMS[orientation]
    images: list[torch.Tensor] = []
    targets: list[int] = []
    for row_number, row in enumerate(rows, start=1):
        if limit is not None and len(targets) >= limit:
            break
        if len(row) != 785:
            raise ValueError(f"Row {row_number} should have 785 columns, found {len(row)}")
        class_id = int(row[0])
        target = the_label_to_mixedcase_index(class_id)
        if target is None:
            continue
        pixels = np.asarray([float(value) for value in row[1:]], dtype=np.float32).reshape(28, 28)
        pixels = transform(pixels)
        if float(pixels.max(initial=0.0)) <= 1.0:
            pixels = pixels * 255.0
        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")
        images.append(_foreground_tensor_from_image(image))
        targets.append(target)
    if not images:
        raise ValueError("No supported English mixed-case rows were found.")
    return torch.stack(images).float(), torch.tensor(targets, dtype=torch.long)


def convert_the_csv(
    csv_path: Path,
    output_path: Path,
    limit: int | None = None,
    orientation: str = "transpose",
) -> dict[str, object]:
    """Read Version IV CSV and write an `images`/`targets` tensor cache."""

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        images, targets = parse_the_rows(csv.reader(handle), limit=limit, orientation=orientation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "targets": targets}, output_path)
    counts = torch.bincount(targets, minlength=len(MIXEDCASE_LABELS))
    present = {MIXEDCASE_LABELS[index]: int(count) for index, count in enumerate(counts.tolist()) if count}
    return {
        "images": int(images.shape[0]),
        "classes": len(present),
        "orientation": orientation,
        "per_class": present,
        "output": str(output_path),
    }


def download_if_needed(source_url: str, csv_path: Path) -> bool:
    """Download the Git LFS-backed CSV unless a local copy already exists."""

    if csv_path.exists() and csv_path.stat().st_size > 1024:
        return False
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(source_url, csv_path)
    return True


def main() -> int:
    """CLI entry point for dataset download/conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_PATH, help="Path to the Version IV CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output `.pt` tensor cache path.")
    parser.add_argument("--download", action="store_true", help="Download Version IV from GitHub media first.")
    parser.add_argument("--url", default=DEFAULT_SOURCE_URL, help="Download URL for Version IV CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests/probes.")
    parser.add_argument(
        "--orientation",
        choices=sorted(ORIENTATION_TRANSFORMS),
        default="transpose",
        help="Pixel orientation transform. Version IV rows render upright with transpose.",
    )
    args = parser.parse_args()

    downloaded = download_if_needed(args.url, args.csv) if args.download else False
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}. Run with --download or provide --csv.")
    report = convert_the_csv(args.csv, args.output, limit=args.limit, orientation=args.orientation)
    action = "downloaded and converted" if downloaded else "converted"
    print(f"{action}: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
