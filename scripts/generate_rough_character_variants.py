"""Generate rough handwritten ASCII-folder samples for character training."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.evaluate_hardcases import _draw_script_like_character

DEFAULT_TARGET_LABELS = "".join(
    dict.fromkeys("O0o1IlisScCzZ5Yy4gq9Bb8Tt7PpKkFfMmUuVvWwXxHhEe.,'!-_:;()/|+%")
)


def render_rough_character(label: str, seed: int, image_size: int = 128) -> Image.Image:
    """Render one deterministic rough black-on-white glyph image."""

    if len(label) != 1:
        raise ValueError("Each rough character label must be exactly one character.")
    rng = random.Random(seed)
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    scale = rng.uniform(image_size * 0.48, image_size * 0.68)
    x = rng.uniform(image_size * 0.18, image_size * 0.34)
    baseline = rng.uniform(image_size * 0.70, image_size * 0.84)
    _draw_script_like_character(draw, label, x, baseline, scale, rng)
    angle = rng.uniform(-7.0, 7.0)
    image = image.rotate(angle, fillcolor="white", resample=Image.Resampling.BICUBIC)
    return image.convert("L")


def generate_rough_character_variants(
    output_root: Path,
    labels: str = DEFAULT_TARGET_LABELS,
    samples_per_label: int = 80,
    seed: int = 42,
    image_size: int = 128,
) -> None:
    """Write rough generated character samples into ASCII-code folders."""

    if samples_per_label < 1:
        raise ValueError("samples_per_label must be at least 1.")
    unique_labels = list(dict.fromkeys(labels))
    if any(len(label) != 1 for label in unique_labels):
        raise ValueError("labels must be a string of single-character labels.")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    for label_index, label in enumerate(unique_labels):
        class_dir = output_root / str(ord(label))
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample_index in range(samples_per_label):
            sample_seed = seed + label_index * 10_000 + sample_index
            image = render_rough_character(label, seed=sample_seed, image_size=image_size)
            image.save(class_dir / f"{sample_index:04d}.png")


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Generate rough handwritten ASCII-folder character variants.")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/rough_character_variants_ascii"))
    parser.add_argument("--labels", default=DEFAULT_TARGET_LABELS)
    parser.add_argument("--samples-per-label", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    generate_rough_character_variants(
        output_root=args.output_root,
        labels=args.labels,
        samples_per_label=args.samples_per_label,
        seed=args.seed,
        image_size=args.image_size,
    )
    print(f"generated {len(dict.fromkeys(args.labels)) * args.samples_per_label} samples in {args.output_root}")


if __name__ == "__main__":
    main()
