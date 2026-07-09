"""Rasterize UJI Pen Characters v2 strokes into ASCII-code image folders."""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


ASCII_LABELS = {chr(code) for code in range(33, 127)}
IMAGE_SIZE = 96
PADDING = 10


@dataclass(frozen=True)
class UjiSample:
    """One isolated UJI character sample."""

    label: str
    session: str
    strokes: tuple[tuple[tuple[int, int], ...], ...]


def _read_text(input_path: Path) -> str:
    """Read either the raw UJI text file or the official zip archive."""

    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            return archive.read("ujipenchars2.txt").decode("utf-8")
    return input_path.read_text(encoding="utf-8")


def parse_uji_samples(text: str, allowed_labels: set[str] | None = None) -> list[UjiSample]:
    """Parse UJI's line-based stroke format into samples."""

    labels = allowed_labels or ASCII_LABELS
    lines = text.splitlines()
    samples: list[UjiSample] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("//"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3 or parts[0] != "WORD":
            continue
        label, session = parts[1], parts[2]
        if index >= len(lines):
            break
        stroke_header = lines[index].strip().split()
        index += 1
        if len(stroke_header) != 2 or stroke_header[0] != "NUMSTROKES":
            continue
        stroke_count = int(stroke_header[1])
        strokes = []
        for _ in range(stroke_count):
            if index >= len(lines):
                break
            stroke_line = lines[index].strip().split()
            index += 1
            if len(stroke_line) < 4 or stroke_line[0] != "POINTS" or stroke_line[2] != "#":
                continue
            point_count = int(stroke_line[1])
            values = [int(value) for value in stroke_line[3:]]
            points = tuple((values[offset], values[offset + 1]) for offset in range(0, min(len(values), point_count * 2), 2))
            if points:
                strokes.append(points)
        if len(label) == 1 and label in labels and strokes:
            samples.append(UjiSample(label=label, session=session, strokes=tuple(strokes)))
    return samples


def rasterize_sample(sample: UjiSample, image_size: int = IMAGE_SIZE, padding: int = PADDING) -> Image.Image:
    """Render a UJI online sample into a black-on-white grayscale image."""

    points = [point for stroke in sample.strokes for point in stroke]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = max(x_max - x_min, 1)
    height = max(y_max - y_min, 1)
    drawable = max(image_size - padding * 2, 1)
    scale = drawable / max(width, height)
    x_offset = (image_size - width * scale) / 2.0
    y_offset = (image_size - height * scale) / 2.0

    image = Image.new("L", (image_size, image_size), 255)
    draw = ImageDraw.Draw(image)
    line_width = max(2, int(round(image_size / 32)))
    for stroke in sample.strokes:
        scaled = [
            (
                int(round((x - x_min) * scale + x_offset)),
                int(round((y - y_min) * scale + y_offset)),
            )
            for x, y in stroke
        ]
        if len(scaled) == 1:
            x, y = scaled[0]
            radius = max(1, line_width)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)
        else:
            draw.line(scaled, fill=0, width=line_width, joint="curve")
    return image


def convert_uji_dataset(input_path: Path, output_root: Path, limit_per_label: int | None = None) -> dict[str, int]:
    """Convert the official UJI data file into the trainer's folder layout."""

    samples = parse_uji_samples(_read_text(input_path))
    output_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for sample in samples:
        count = counts.get(sample.label, 0)
        if limit_per_label is not None and count >= limit_per_label:
            continue
        counts[sample.label] = count + 1
        target_dir = output_root / str(ord(sample.label))
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_session = sample.session.replace("/", "_")
        rasterize_sample(sample).save(target_dir / f"uji-{safe_session}-{count + 1:03d}.png")
    return counts


def main() -> None:
    """CLI entrypoint for preparing UJI Pen Characters v2."""

    parser = argparse.ArgumentParser(description="Prepare UJI Pen Characters v2 as ASCII-code PNG folders.")
    parser.add_argument("--input", type=Path, default=Path("data/uji_pen_v2/raw/uji_pen_characters_v2.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/uji_pen_v2/character_ascii"))
    parser.add_argument("--limit-per-label", type=int, default=None)
    args = parser.parse_args()
    counts = convert_uji_dataset(args.input, args.output, args.limit_per_label)
    print(f"classes={len(counts)} images={sum(counts.values())}")


if __name__ == "__main__":
    main()
