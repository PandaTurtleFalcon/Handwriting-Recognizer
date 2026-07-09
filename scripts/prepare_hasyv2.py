"""Prepare HASYv2 subsets for this project's training CLIs.

HASYv2 stores labels as LaTeX strings. This script copies the simple labels
that overlap with the app into image-folder layouts:

* ``alnum_image_folder``: 0-9 and A-Z, with lowercase folded to uppercase.
* ``character_ascii``: ASCII codepoint folders, matching ``character_labels``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


ALNUM_LABELS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ASCII_LABELS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-/<>[]|")


def alnum_label(label: str) -> str | None:
    """Map a HASY label into the 36-class alnum label set."""

    if len(label) != 1:
        return None
    folded = label.upper()
    return folded if folded in ALNUM_LABELS else None


def ascii_label(label: str) -> str | None:
    """Map a HASY label into the broader ASCII character label set."""

    if len(label) == 1 and label in ASCII_LABELS:
        return str(ord(label))
    return None


def prepare_hasy_subset(extracted_root: Path, output_root: Path, mode: str) -> Counter[str]:
    """Copy matching HASYv2 images into a training image-folder layout."""

    labels_path = extracted_root / "hasy-data-labels.csv"
    if not labels_path.exists():
        raise RuntimeError(f"Missing HASY labels CSV: {labels_path}")
    if mode not in {"alnum", "ascii"}:
        raise ValueError(f"Unsupported mode: {mode}")

    mapper = alnum_label if mode == "alnum" else ascii_label
    counts: Counter[str] = Counter()
    output_root.mkdir(parents=True, exist_ok=True)

    with labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = mapper(row["latex"])
            if label is None:
                continue
            source = extracted_root / row["path"]
            if not source.exists():
                raise RuntimeError(f"Missing HASY image: {source}")
            target_dir = output_root / label
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name
            if not target.exists():
                shutil.copy2(source, target)
            counts[label] += 1
    return counts


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Prepare HASYv2 image-folder subsets.")
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=Path("data/extra_hasyv2/extracted"),
        help="Path containing HASYv2 hasy-data-labels.csv and hasy-data/.",
    )
    parser.add_argument(
        "--alnum-output",
        type=Path,
        default=Path("data/extra_hasyv2/alnum_image_folder"),
        help="Output folder for 36-class alnum training.",
    )
    parser.add_argument(
        "--ascii-output",
        type=Path,
        default=Path("data/extra_hasyv2/character_ascii"),
        help="Output folder for ASCII-codepoint character training.",
    )
    parser.add_argument("--skip-alnum", action="store_true")
    parser.add_argument("--skip-ascii", action="store_true")
    args = parser.parse_args()

    if not args.skip_alnum:
        counts = prepare_hasy_subset(args.extracted_root, args.alnum_output, "alnum")
        print(f"alnum classes={len(counts)} images={sum(counts.values())}")
    if not args.skip_ascii:
        counts = prepare_hasy_subset(args.extracted_root, args.ascii_output, "ascii")
        print(f"ascii classes={len(counts)} images={sum(counts.values())}")


if __name__ == "__main__":
    main()
