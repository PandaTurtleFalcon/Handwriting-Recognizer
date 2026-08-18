"""Prepare local CVL word boxes as approximate mixed-case letter crops.

CVL distributes word-level XML boxes, not character boxes. This importer keeps
the raw CVL download local under ignored `data/`, crops each labeled word, and
splits the word crop into approximate character windows by cumulative ink mass.
The output cache uses the existing `images`/`targets` format accepted by
`alnum_model.load_mixedcase_extra_cache()`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import MIXEDCASE_LABELS, _foreground_tensor_from_image  # noqa: E402
from mnist_model import _foreground_from_image  # noqa: E402


DEFAULT_SOURCE_ROOT = PROJECT_DIR / "data" / "cvl"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "data" / "cvl_letters_twin_subset.pt"
DEFAULT_LABELS = "Oo0Il1isS5CcUuvPpZz2"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TEXT_ATTRS = ("text", "transcription", "word", "label", "value", "contents")
ALNUM_RE = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class CvlWord:
    """One labeled CVL word box."""

    text: str
    box: tuple[int, int, int, int]
    xml_path: Path


def _attr_float(attrs: dict[str, str], *names: str) -> float | None:
    """Return the first parseable float attribute from a list of aliases."""

    lowered = {key.lower(): value for key, value in attrs.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _box_from_attrs(attrs: dict[str, str]) -> tuple[int, int, int, int] | None:
    """Parse common word-box attribute layouts into x0/y0/x1/y1."""

    x = _attr_float(attrs, "x", "left", "x0", "xmin", "xMin")
    y = _attr_float(attrs, "y", "top", "y0", "ymin", "yMin")
    width = _attr_float(attrs, "w", "width")
    height = _attr_float(attrs, "h", "height")
    if x is not None and y is not None and width is not None and height is not None:
        return _clean_box(x, y, x + width, y + height)

    x0 = _attr_float(attrs, "x0", "x1", "left", "xmin", "xMin")
    y0 = _attr_float(attrs, "y0", "y1", "top", "ymin", "yMin")
    x1 = _attr_float(attrs, "x1", "x2", "right", "xmax", "xMax")
    y1 = _attr_float(attrs, "y1", "y2", "bottom", "ymax", "yMax")
    if None not in (x0, y0, x1, y1):
        return _clean_box(float(x0), float(y0), float(x1), float(y1))
    return None


def _clean_box(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int] | None:
    """Round and validate a positive-area bounding box."""

    left, right = sorted((int(round(x0)), int(round(x1))))
    top, bottom = sorted((int(round(y0)), int(round(y1))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _text_from_element(element: ElementTree.Element) -> str:
    """Read the word text from common XML attributes or element content."""

    for name in TEXT_ATTRS:
        value = element.attrib.get(name)
        if value:
            return value.strip()
        value = element.attrib.get(name.upper())
        if value:
            return value.strip()
    return "".join(element.itertext()).strip()


def parse_cvl_words(xml_path: Path) -> list[CvlWord]:
    """Parse labeled word boxes from one CVL XML file."""

    try:
        root = ElementTree.parse(xml_path).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"Could not parse CVL XML: {xml_path}") from exc
    words = []
    for element in root.iter():
        box = _box_from_attrs(dict(element.attrib))
        text = _text_from_element(element)
        if box is None or not ALNUM_RE.search(text):
            continue
        words.append(CvlWord(text=text, box=box, xml_path=xml_path))
    return words


def build_image_index(root: Path) -> dict[str, Path]:
    """Map image stems to local CVL image paths."""

    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem, path)
    return index


def matching_image_for_xml(xml_path: Path, image_index: dict[str, Path]) -> Path | None:
    """Find the image whose stem matches a CVL XML file."""

    if xml_path.stem in image_index:
        return image_index[xml_path.stem]
    for stem, image_path in image_index.items():
        if stem.startswith(xml_path.stem) or xml_path.stem.startswith(stem):
            return image_path
    return None


def _ink_bounds(mask: np.ndarray, left: int, right: int) -> tuple[int, int]:
    """Tighten one horizontal segment to the ink columns it contains."""

    segment = mask[:, left:right]
    columns = np.where(segment.any(axis=0))[0]
    if columns.size == 0:
        return left, right
    return left + int(columns.min()), left + int(columns.max()) + 1


def split_word_by_ink(word_image: Image.Image, labels: str) -> list[tuple[str, Image.Image]]:
    """Split a word crop into approximate per-character crops."""

    if not labels:
        return []
    foreground = _foreground_from_image(word_image)
    mask = foreground > 0.18
    if not bool(mask.any()):
        return []
    column_mass = foreground.sum(axis=0)
    total_mass = float(column_mass.sum())
    if total_mass <= 0.0:
        return []
    cumulative = np.cumsum(column_mass)
    boundaries = [0]
    for index in range(1, len(labels)):
        boundary = int(np.searchsorted(cumulative, total_mass * index / len(labels)))
        boundaries.append(max(boundaries[-1] + 1, min(boundary, foreground.shape[1] - 1)))
    boundaries.append(foreground.shape[1])

    crops = []
    for label, left, right in zip(labels, boundaries, boundaries[1:]):
        tight_left, tight_right = _ink_bounds(mask, left, right)
        if tight_right <= tight_left:
            continue
        crop = Image.fromarray((foreground[:, tight_left:tight_right] * 255).astype(np.uint8), mode="L")
        crops.append((label, crop))
    return crops


def _filtered_labels(text: str, allowed_labels: set[str]) -> str:
    """Return the trainable labels from a CVL word transcription."""

    return "".join(match.group(0) for match in ALNUM_RE.finditer(text) if match.group(0) in allowed_labels)


def prepare_cvl_letters(
    source_root: Path,
    output_path: Path,
    labels: str = DEFAULT_LABELS,
    limit_per_label: int | None = None,
) -> dict[str, object]:
    """Convert local CVL word boxes into a mixed-case tensor cache."""

    if not source_root.exists():
        raise FileNotFoundError(f"CVL source folder not found: {source_root}")
    label_to_index = {label: index for index, label in enumerate(MIXEDCASE_LABELS)}
    allowed_labels = {label for label in labels if label in label_to_index}
    if not allowed_labels:
        raise ValueError("No requested labels exist in the mixed-case label set.")

    image_index = build_image_index(source_root)
    images: list[torch.Tensor] = []
    targets: list[int] = []
    counts: dict[str, int] = {}
    word_count = 0
    missing_images = 0
    for xml_path in sorted(source_root.rglob("*.xml")):
        image_path = matching_image_for_xml(xml_path, image_index)
        if image_path is None:
            missing_images += 1
            continue
        words = parse_cvl_words(xml_path)
        if not words:
            continue
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            for word in words:
                trainable = _filtered_labels(word.text, allowed_labels)
                if not trainable:
                    continue
                word_count += 1
                crop = rgb_image.crop(word.box)
                for label, label_crop in split_word_by_ink(crop, trainable):
                    count = counts.get(label, 0)
                    if limit_per_label is not None and count >= limit_per_label:
                        continue
                    images.append(_foreground_tensor_from_image(label_crop))
                    targets.append(label_to_index[label])
                    counts[label] = count + 1
    if not images:
        raise ValueError(f"No CVL letter crops were created from {source_root}")

    image_tensor = torch.stack(images).float()
    target_tensor = torch.tensor(targets, dtype=torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": image_tensor, "targets": target_tensor}, output_path)
    return {
        "images": int(image_tensor.shape[0]),
        "classes": len(counts),
        "words_used": word_count,
        "missing_image_xml_files": missing_images,
        "per_class": dict(sorted(counts.items())),
        "output": str(output_path),
    }


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--limit-per-label", type=int, default=None)
    args = parser.parse_args()
    report = prepare_cvl_letters(
        source_root=args.source_root,
        output_path=args.output,
        labels=args.labels,
        limit_per_label=args.limit_per_label,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
