"""Generate a small synthetic punctuation ASCII-folder dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

TARGET_LABELS = [".", "'", "-", "_", "+", "%", "!", "/", "|", ";", ":"]


def _new_canvas(rng: random.Random) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    size = rng.randint(42, 68)
    image = Image.new("L", (size, size), 255)
    return image, ImageDraw.Draw(image)


def _draw_dot(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)


def _draw_label(draw: ImageDraw.ImageDraw, label: str, size: int, rng: random.Random) -> None:
    thickness = rng.randint(3, 7)
    margin = rng.randint(7, 13)
    mid_x = size // 2 + rng.randint(-4, 4)
    mid_y = size // 2 + rng.randint(-4, 4)
    if label == ".":
        _draw_dot(draw, mid_x, size - margin, rng.randint(3, 6))
    elif label == "'":
        x = mid_x + rng.randint(-2, 2)
        y0 = margin
        y1 = margin + rng.randint(9, 18)
        draw.line((x, y0, x + rng.randint(-2, 2), y1), fill=0, width=thickness)
    elif label == "-":
        y = mid_y
        draw.line((margin, y, size - margin, y + rng.randint(-1, 1)), fill=0, width=thickness)
    elif label == "_":
        y = size - margin
        draw.line((margin, y, size - margin, y), fill=0, width=thickness)
    elif label == "+":
        draw.line((margin, mid_y, size - margin, mid_y), fill=0, width=thickness)
        draw.line((mid_x, margin, mid_x, size - margin), fill=0, width=thickness)
    elif label == "%":
        _draw_dot(draw, margin + 6, margin + 6, rng.randint(3, 5))
        _draw_dot(draw, size - margin - 6, size - margin - 6, rng.randint(3, 5))
        draw.line((size - margin, margin, margin, size - margin), fill=0, width=max(2, thickness - 2))
    elif label == "!":
        draw.line((mid_x, margin, mid_x + rng.randint(-1, 1), size - margin - 11), fill=0, width=thickness)
        _draw_dot(draw, mid_x, size - margin, rng.randint(3, 5))
    elif label == "/":
        draw.line((size - margin, margin, margin, size - margin), fill=0, width=thickness)
    elif label == "|":
        draw.line((mid_x, margin, mid_x + rng.randint(-1, 1), size - margin), fill=0, width=thickness)
    elif label == ";":
        _draw_dot(draw, mid_x, margin + 7, rng.randint(3, 5))
        draw.line((mid_x, size - margin - 12, mid_x - rng.randint(1, 4), size - margin), fill=0, width=thickness)
    elif label == ":":
        _draw_dot(draw, mid_x, margin + 7, rng.randint(3, 5))
        _draw_dot(draw, mid_x, size - margin - 7, rng.randint(3, 5))


def generate_punctuation_variants(output_root: Path, samples_per_label: int, seed: int) -> None:
    """Write generated punctuation samples into ASCII-code folders."""

    rng = random.Random(seed)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    for label in TARGET_LABELS:
        class_dir = output_root / str(ord(label))
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_label):
            image, draw = _new_canvas(rng)
            _draw_label(draw, label, image.size[0], rng)
            angle = rng.uniform(-7.0, 7.0)
            image = image.rotate(angle, fillcolor=255, resample=Image.Resampling.BICUBIC)
            image.save(class_dir / f"{index:04d}.png")


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Generate synthetic punctuation variants.")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/punctuation_variants_ascii"))
    parser.add_argument("--samples-per-label", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_punctuation_variants(args.output_root, args.samples_per_label, args.seed)
    print(f"generated {len(TARGET_LABELS) * args.samples_per_label} samples in {args.output_root}")


if __name__ == "__main__":
    main()
