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
    "Cc",
    "Ff",
    "Mm",
    "Uu",
    "Vv",
    "Ww",
    "Xx",
    "Yy",
    "Zz",
    "Pp",
    "Kk",
    "0Oo",
    "O0o",
    "1Il",
    "I1l",
    "Ss5",
    "5Ss",
    "2Zz",
    "9qg",
    "G6b",
    "B8",
    "Tt7",
    "Hello",
    "HELLO",
    "hello",
    "Cat",
    "USA",
    "abc123",
    "A1b2",
    "Hi5!",
    "look behind",
    "you",
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
    font: str = ""


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


def font_label(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, fallback_index: int = 0) -> str:
    """Return a stable short label for a PIL font object."""

    path = getattr(font, "path", "")
    if path:
        return Path(str(path)).stem
    return f"default-{fallback_index}"


def iter_fonts(size: int) -> list[tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]]:
    """Load available handwriting-ish system fonts for evaluator coverage."""

    fonts: list[tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]] = []
    for path in FONT_CANDIDATES:
        candidate = Path(path)
        if not candidate.exists():
            continue
        try:
            fonts.append((candidate.stem, ImageFont.truetype(str(candidate), size=size)))
        except OSError:
            continue
    if not fonts:
        default_font = ImageFont.load_default()
        fonts.append((font_label(default_font), default_font))
    return fonts


def choose_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load one handwriting-ish system font, falling back to Pillow's default."""

    return iter_fonts(size)[0][1]


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


def evaluate_cases(cases: list[str] | None = None, all_fonts: bool = False) -> dict[str, object]:
    """Run generated hard cases through the app classifier."""

    selected_cases = cases or DEFAULT_CASES
    model, device = load_web_models()
    default_font = choose_font(72)
    fonts = iter_fonts(72) if all_fonts else [(font_label(default_font), default_font)]
    results: list[HardCaseResult] = []
    for font_name, font in fonts:
        for target in selected_cases:
            payload = render_case(target, font)
            classified = main.classify_files([(f"{target}-{font_name}.png", payload)], model, device, save_sources=False)[0]
            prediction = str(classified.get("sequence", ""))
            results.append(
                HardCaseResult(
                    target=target,
                    prediction=prediction,
                    exact=prediction == target,
                    ambiguity_aware=sequence_matches_with_ambiguity(target, prediction),
                    font=font_name,
                )
            )
    exact = sum(result.exact for result in results)
    ambiguity = sum(result.ambiguity_aware for result in results)
    per_font: dict[str, dict[str, object]] = {}
    for font_name, _ in fonts:
        font_results = [result for result in results if result.font == font_name]
        font_exact = sum(result.exact for result in font_results)
        font_ambiguity = sum(result.ambiguity_aware for result in font_results)
        per_font[font_name] = {
            "total": len(font_results),
            "exact_correct": font_exact,
            "exact_accuracy": 100.0 * font_exact / max(len(font_results), 1),
            "ambiguity_aware_correct": font_ambiguity,
            "ambiguity_aware_accuracy": 100.0 * font_ambiguity / max(len(font_results), 1),
        }
    return {
        "total": len(results),
        "exact_correct": exact,
        "exact_accuracy": 100.0 * exact / max(len(results), 1),
        "ambiguity_aware_correct": ambiguity,
        "ambiguity_aware_accuracy": 100.0 * ambiguity / max(len(results), 1),
        "per_font": per_font,
        "results": [result.__dict__ for result in results],
    }


def main_cli() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Evaluate generated hard-case strings against the web recognizer.")
    parser.add_argument("--case", action="append", default=[], help="Specific case to evaluate; repeatable.")
    parser.add_argument("--all-fonts", action="store_true", help="Evaluate every available configured font.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    report = evaluate_cases(args.case or None, all_fonts=args.all_fonts)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(
        f"hardcase_exact={report['exact_accuracy']:.2f}% "
        f"({report['exact_correct']}/{report['total']}) "
        f"ambiguity_aware={report['ambiguity_aware_accuracy']:.2f}% "
        f"({report['ambiguity_aware_correct']}/{report['total']})"
    )
    if args.all_fonts:
        for font_name, font_report in report["per_font"].items():
            print(
                f"font={font_name!r} exact={font_report['exact_accuracy']:.2f}% "
                f"({font_report['exact_correct']}/{font_report['total']}) "
                f"ambiguity_aware={font_report['ambiguity_aware_accuracy']:.2f}% "
                f"({font_report['ambiguity_aware_correct']}/{font_report['total']})"
            )
    for result in report["results"]:
        status = "ok" if result["exact"] else "miss"
        ambiguity = "amb-ok" if result["ambiguity_aware"] else "amb-miss"
        font = f" font={result['font']!r}" if result.get("font") else ""
        print(f"{status}/{ambiguity}:{font} target={result['target']!r} prediction={result['prediction']!r}")


if __name__ == "__main__":
    main_cli()
