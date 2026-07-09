"""Evaluate live app recognition on generated hard-case handwriting strings."""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import main
from character_model import labels_match_with_ambiguity
from mnist_model import get_device


DEFAULT_CASES = [
    "15",
    "27",
    "Hi",
    "Hi!",
    "Hi.",
    "Test",
    "S5s",
    "Oo0",
    "Il1!",
    "T3s7",
    "(85)",
    "can't",
]
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


@dataclass(frozen=True)
class HardCaseResult:
    """One app-level generated hard-case result."""

    target: str
    prediction: str
    exact: bool
    ambiguity_aware: bool


def load_web_models() -> tuple[object, object]:
    """Initialize the same recognizer stack used by the website."""

    device = get_device()
    main.MnistWebHandler.device = device
    (
        main.MnistWebHandler.model,
        main.MnistWebHandler.labels,
        main.MnistWebHandler.letter_model,
        main.MnistWebHandler.letter_labels,
        main.MnistWebHandler.alnum_model,
        main.MnistWebHandler.alnum_labels,
    ) = main.load_character_recognizer_stack(device)
    main.MnistWebHandler.recognizer_kind = "characters" if main.MnistWebHandler.labels is not None else "digits"
    return main.MnistWebHandler.model, device


def sequence_matches_with_ambiguity(target: str, prediction: str) -> bool:
    """Return true when equal-length strings only differ by visual twins."""

    if target == prediction:
        return True
    if len(target) != len(prediction):
        return False
    return all(labels_match_with_ambiguity(expected, actual) for expected, actual in zip(target, prediction))


def choose_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a handwriting-ish system font, falling back to Pillow's default."""

    for path in FONT_CANDIDATES:
        candidate = Path(path)
        if not candidate.exists():
            continue
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_case(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> bytes:
    """Render one black-on-white text sample as PNG bytes."""

    scratch = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(scratch)
    glyph_boxes = [draw.textbbox((0, 0), character, font=font) for character in text]
    tracking = 18
    glyph_widths = [max(1, box[2] - box[0]) for box in glyph_boxes]
    glyph_heights = [max(1, box[3] - box[1]) for box in glyph_boxes]
    width = max(96, sum(glyph_widths) + tracking * max(len(text) - 1, 0) + 64)
    height = max(96, max(glyph_heights, default=1) + 64)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    x = 32
    for character, bbox, glyph_width in zip(text, glyph_boxes, glyph_widths):
        draw.text((x - bbox[0], 32 - bbox[1]), character, fill="black", font=font)
        x += glyph_width + tracking
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def evaluate_cases(cases: list[str] | None = None) -> dict[str, object]:
    """Run generated hard cases through the app classifier."""

    selected_cases = cases or DEFAULT_CASES
    model, device = load_web_models()
    font = choose_font(72)
    results: list[HardCaseResult] = []
    for target in selected_cases:
        payload = render_case(target, font)
        classified = main.classify_files([(f"{target}.png", payload)], model, device, save_sources=False)[0]
        prediction = str(classified.get("sequence", ""))
        results.append(
            HardCaseResult(
                target=target,
                prediction=prediction,
                exact=prediction == target,
                ambiguity_aware=sequence_matches_with_ambiguity(target, prediction),
            )
        )
    exact = sum(result.exact for result in results)
    ambiguity = sum(result.ambiguity_aware for result in results)
    return {
        "total": len(results),
        "exact_correct": exact,
        "exact_accuracy": 100.0 * exact / max(len(results), 1),
        "ambiguity_aware_correct": ambiguity,
        "ambiguity_aware_accuracy": 100.0 * ambiguity / max(len(results), 1),
        "results": [result.__dict__ for result in results],
    }


def main_cli() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Evaluate generated hard-case strings against the web recognizer.")
    parser.add_argument("--case", action="append", default=[], help="Specific case to evaluate; repeatable.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    report = evaluate_cases(args.case or None)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(
        f"hardcase_exact={report['exact_accuracy']:.2f}% "
        f"({report['exact_correct']}/{report['total']}) "
        f"ambiguity_aware={report['ambiguity_aware_accuracy']:.2f}% "
        f"({report['ambiguity_aware_correct']}/{report['total']})"
    )
    for result in report["results"]:
        status = "ok" if result["exact"] else "miss"
        ambiguity = "amb-ok" if result["ambiguity_aware"] else "amb-miss"
        print(f"{status}/{ambiguity}: target={result['target']!r} prediction={result['prediction']!r}")


if __name__ == "__main__":
    main_cli()
