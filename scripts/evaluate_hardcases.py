"""Evaluate live app recognition on generated hard-case handwriting strings."""

from __future__ import annotations

import argparse
import io
import json
import math
import random
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
    "look behind you",
    "you",
]
UPLOADED_FIXTURES = [
    {
        "path": PROJECT_DIR / "data" / "app_hardcase_fixtures" / "look_behind_you_reported.png",
        "target": "look behind\nyou",
    }
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
    raw_prediction: str | None = None
    raw_rows: list[str] | None = None
    prediction_count: int | None = None
    raw_exact: bool | None = None
    raw_ambiguity_aware: bool | None = None


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

    if display_matches(target, prediction):
        return True
    if len(target) != len(prediction):
        return False
    return all(labels_match_with_ambiguity(expected, actual) for expected, actual in zip(target, prediction))


def display_matches(target: str, prediction: str) -> bool:
    """Return true when strings match exactly or only differ by whitespace layout."""

    if target == prediction:
        return True
    if not any(character.isspace() for character in target + prediction):
        return False
    return " ".join(target.split()) == " ".join(prediction.split())


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


def _jitter_polyline(points: list[tuple[float, float]], rng: random.Random, jitter: float) -> list[tuple[float, float]]:
    """Return points with deterministic hand-jitter added."""

    return [(x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)) for x, y in points]


def _draw_script_like_character(
    draw: ImageDraw.ImageDraw,
    character: str,
    x: float,
    baseline: float,
    scale: float,
    rng: random.Random,
) -> float:
    """Draw a rough handwritten glyph and return its advance width."""

    width = max(2, int(round(scale * 0.09)))
    jitter = scale * 0.025

    def line(points: list[tuple[float, float]], extra_width: int = 0) -> None:
        draw.line(_jitter_polyline(points, rng, jitter), fill="black", width=max(1, width + extra_width), joint="curve")

    def ellipse(bounds: tuple[float, float, float, float], extra_width: int = 0) -> None:
        draw.ellipse(bounds, outline="black", width=max(1, width + extra_width))

    top = baseline - scale
    mid = baseline - scale * 0.52
    bottom = baseline
    advance = scale * 0.62
    if character == " ":
        return scale * 0.5
    if character in {"l", "1", "I"}:
        line([(x + scale * 0.28, top), (x + scale * 0.30, bottom)])
        if character in {"1", "I"}:
            line([(x + scale * 0.15, top + scale * 0.02), (x + scale * 0.46, top + scale * 0.02)])
        return scale * 0.46
    if character in {"o", "O", "0"}:
        if character == "o":
            ellipse((x + scale * 0.12, top + scale * 0.42, x + scale * 0.52, bottom - scale * 0.08))
            return scale * 0.58
        ellipse((x + scale * 0.06, top + scale * 0.24, x + scale * 0.58, bottom - scale * 0.05))
        if character == "0":
            line([(x + scale * 0.20, bottom - scale * 0.08), (x + scale * 0.50, top + scale * 0.28)])
        return scale * 0.68
    if character in {"c", "C"}:
        c_top = top + scale * 0.50 if character == "c" else top + scale * 0.28
        c_mid = top + scale * 0.72 if character == "c" else mid
        c_bottom = bottom - scale * 0.18 if character == "c" else bottom - scale * 0.14
        line(
            [
                (x + scale * 0.58, c_top),
                (x + scale * 0.28, c_top - scale * 0.04),
                (x + scale * 0.10, c_mid),
                (x + scale * 0.24, c_bottom),
                (x + scale * 0.58, c_bottom - scale * 0.04),
            ]
        )
        return scale * 0.64
    if character in {"e", "E"}:
        line(
            [
                (x + scale * 0.58, top + scale * 0.28),
                (x + scale * 0.22, top + scale * 0.24),
                (x + scale * 0.08, mid),
                (x + scale * 0.52, mid),
                (x + scale * 0.18, bottom - scale * 0.10),
                (x + scale * 0.58, bottom - scale * 0.12),
            ]
        )
        return scale * 0.64
    if character in {"s", "S", "5"}:
        s_top = top + scale * 0.36 if character == "s" else top + scale * 0.22
        s_mid = top + scale * 0.64 if character == "s" else mid
        s_bottom = bottom - scale * 0.10
        line(
            [
                (x + scale * 0.58, s_top),
                (x + scale * 0.18, s_top),
                (x + scale * 0.18, s_mid),
                (x + scale * 0.55, s_mid),
                (x + scale * 0.55, s_bottom),
                (x + scale * 0.12, s_bottom),
            ]
        )
        return scale * 0.66
    if character in {"a", "A"}:
        line(
            [
                (x + scale * 0.12, bottom - scale * 0.10),
                (x + scale * 0.34, top + scale * 0.18),
                (x + scale * 0.58, bottom - scale * 0.10),
            ]
        )
        line([(x + scale * 0.22, mid), (x + scale * 0.50, mid)])
        if character == "a":
            ellipse((x + scale * 0.12, top + scale * 0.36, x + scale * 0.52, bottom - scale * 0.08))
            line([(x + scale * 0.54, top + scale * 0.32), (x + scale * 0.54, bottom - scale * 0.06)])
        return scale * 0.66
    if character in {"b", "B"}:
        line([(x + scale * 0.12, top + scale * 0.12), (x + scale * 0.12, bottom - scale * 0.06)])
        if character == "B":
            line(
                [
                    (x + scale * 0.12, top + scale * 0.14),
                    (x + scale * 0.52, top + scale * 0.18),
                    (x + scale * 0.54, mid),
                    (x + scale * 0.14, mid),
                    (x + scale * 0.56, mid + scale * 0.06),
                    (x + scale * 0.54, bottom - scale * 0.08),
                    (x + scale * 0.12, bottom - scale * 0.06),
                ]
            )
        else:
            ellipse((x + scale * 0.10, mid - scale * 0.04, x + scale * 0.58, bottom - scale * 0.06))
        return scale * 0.66
    if character in {"d", "D"}:
        line([(x + scale * 0.54, top + scale * 0.10), (x + scale * 0.54, bottom - scale * 0.06)])
        ellipse((x + scale * 0.10, mid - scale * 0.04, x + scale * 0.56, bottom - scale * 0.06))
        return scale * 0.66
    if character == "F":
        line([(x + scale * 0.42, top + scale * 0.10), (x + scale * 0.30, bottom - scale * 0.04)])
        line([(x + scale * 0.16, top + scale * 0.18), (x + scale * 0.62, top + scale * 0.18)])
        line([(x + scale * 0.16, mid), (x + scale * 0.50, mid)])
        return scale * 0.62
    if character == "f":
        line([(x + scale * 0.42, top + scale * 0.16), (x + scale * 0.26, bottom + scale * 0.16)])
        line([(x + scale * 0.18, mid + scale * 0.02), (x + scale * 0.48, mid + scale * 0.02)])
        return scale * 0.52
    if character in {"g", "G", "q", "9", "6"}:
        ellipse((x + scale * 0.10, top + scale * 0.24, x + scale * 0.58, bottom - scale * 0.12))
        if character in {"g", "q"}:
            line([(x + scale * 0.52, bottom - scale * 0.14), (x + scale * 0.38, bottom + scale * 0.24)])
        elif character == "9":
            line([(x + scale * 0.52, mid), (x + scale * 0.32, bottom - scale * 0.02)])
        elif character == "6":
            line([(x + scale * 0.20, mid), (x + scale * 0.42, top + scale * 0.12)])
        else:
            line([(x + scale * 0.42, mid), (x + scale * 0.62, mid)])
        return scale * 0.68
    if character in {"h", "H"}:
        line([(x + scale * 0.12, top + scale * 0.10), (x + scale * 0.12, bottom - scale * 0.06)])
        line([(x + scale * 0.54, top + scale * 0.12), (x + scale * 0.54, bottom - scale * 0.06)])
        line([(x + scale * 0.12, mid), (x + scale * 0.54, mid)])
        return scale * 0.68
    if character in {"m", "M"}:
        m_top = top + scale * 0.36 if character == "m" else top + scale * 0.28
        line(
            [
                (x + scale * 0.10, bottom - scale * 0.08),
                (x + scale * 0.10, m_top),
                (x + scale * 0.34, mid),
                (x + scale * 0.56, m_top),
                (x + scale * 0.56, bottom - scale * 0.08),
            ]
        )
        return scale * 0.72
    if character in {"n", "N"}:
        line([(x + scale * 0.12, bottom - scale * 0.08), (x + scale * 0.12, top + scale * 0.30), (x + scale * 0.54, bottom - scale * 0.08), (x + scale * 0.54, top + scale * 0.30)])
        return scale * 0.66
    if character in {"p", "P"}:
        stem_bottom = bottom + scale * 0.24 if character == "p" else bottom - scale * 0.06
        loop_top = top + scale * 0.36 if character == "p" else top + scale * 0.16
        line([(x + scale * 0.12, top + scale * 0.12), (x + scale * 0.12, stem_bottom)])
        ellipse((x + scale * 0.10, loop_top, x + scale * 0.58, mid + scale * 0.12))
        return scale * 0.66
    if character in {"r", "R"}:
        line([(x + scale * 0.12, top + scale * 0.12), (x + scale * 0.12, bottom - scale * 0.06)])
        line([(x + scale * 0.12, top + scale * 0.16), (x + scale * 0.54, top + scale * 0.22), (x + scale * 0.48, mid), (x + scale * 0.12, mid)])
        if character == "R":
            line([(x + scale * 0.22, mid), (x + scale * 0.58, bottom - scale * 0.06)])
        return scale * 0.66
    if character in {"u", "U"}:
        u_top = top + scale * 0.40 if character == "u" else top + scale * 0.28
        line(
            [
                (x + scale * 0.12, u_top),
                (x + scale * 0.12, bottom - scale * 0.14),
                (x + scale * 0.50, bottom - scale * 0.14),
                (x + scale * 0.50, u_top),
            ]
        )
        return scale * 0.66
    if character in {"v", "V", "y", "Y"}:
        v_top = top + scale * 0.40 if character in {"v", "y"} else top + scale * 0.18
        line([(x + scale * 0.10, v_top), (x + scale * 0.34, bottom - scale * 0.06), (x + scale * 0.60, v_top)])
        if character in {"y", "Y"}:
            line([(x + scale * 0.34, bottom - scale * 0.06), (x + scale * 0.28, bottom + scale * 0.32)])
        return scale * 0.70
    if character in {"w", "W"}:
        w_top = top + scale * 0.40 if character == "w" else top + scale * 0.18
        line(
            [
                (x + scale * 0.08, w_top),
                (x + scale * 0.22, bottom - scale * 0.06),
                (x + scale * 0.38, w_top + scale * 0.16),
                (x + scale * 0.54, bottom - scale * 0.06),
                (x + scale * 0.68, w_top),
            ]
        )
        return scale * 0.76
    if character in {"T", "t", "7"}:
        line([(x + scale * 0.06, top + scale * 0.18), (x + scale * 0.62, top + scale * 0.18)])
        line([(x + scale * 0.36, top + scale * 0.18), (x + scale * 0.34, bottom)])
        if character == "7":
            line([(x + scale * 0.62, top + scale * 0.18), (x + scale * 0.26, bottom)])
        return scale * 0.70
    if character in {"2", "3", "8"}:
        if character == "2":
            line(
                [
                    (x + scale * 0.12, top + scale * 0.24),
                    (x + scale * 0.54, top + scale * 0.22),
                    (x + scale * 0.52, mid),
                    (x + scale * 0.14, bottom - scale * 0.08),
                    (x + scale * 0.60, bottom - scale * 0.08),
                ]
            )
        elif character == "3":
            line(
                [
                    (x + scale * 0.12, top + scale * 0.22),
                    (x + scale * 0.54, top + scale * 0.24),
                    (x + scale * 0.42, mid),
                    (x + scale * 0.58, bottom - scale * 0.12),
                    (x + scale * 0.12, bottom - scale * 0.08),
                ]
            )
        else:
            ellipse((x + scale * 0.12, top + scale * 0.16, x + scale * 0.56, mid + scale * 0.04))
            ellipse((x + scale * 0.10, mid - scale * 0.02, x + scale * 0.58, bottom - scale * 0.06))
        return scale * 0.66
    if character in {"Z", "z"}:
        z_top = top + scale * 0.40 if character == "z" else top + scale * 0.20
        line(
            [
                (x + scale * 0.10, z_top),
                (x + scale * 0.58, z_top),
                (x + scale * 0.12, bottom - scale * 0.08),
                (x + scale * 0.60, bottom - scale * 0.08),
            ]
        )
        return scale * 0.66
    if character in {"(", ")"}:
        side = -1 if character == "(" else 1
        points = []
        for index in range(8):
            phase = index / 7
            y = top + scale * phase
            curve = math.sin(math.pi * phase) * scale * 0.18 * side
            points.append((x + scale * 0.34 + curve, y))
        line(points)
        return scale * 0.45
    if character in {".", "'", ",", ":"}:
        radius = scale * 0.055
        dot_x = x + scale * 0.24
        if character == "'":
            draw.ellipse((dot_x - radius, top + scale * 0.06, dot_x + radius, top + scale * 0.06 + radius * 2), fill="black")
        elif character == ",":
            draw.ellipse((dot_x - radius, bottom - scale * 0.02, dot_x + radius, bottom - scale * 0.02 + radius * 2), fill="black")
            line([(dot_x + radius * 0.4, bottom + scale * 0.04), (dot_x - radius * 0.6, bottom + scale * 0.14)])
        elif character == ":":
            draw.ellipse((dot_x - radius, mid - scale * 0.18, dot_x + radius, mid - scale * 0.18 + radius * 2), fill="black")
            draw.ellipse((dot_x - radius, bottom - scale * 0.06, dot_x + radius, bottom - scale * 0.06 + radius * 2), fill="black")
        else:
            draw.ellipse((dot_x - radius, bottom - scale * 0.02, dot_x + radius, bottom - scale * 0.02 + radius * 2), fill="black")
        return scale * 0.34
    if character in {"!", "i"}:
        if character == "i":
            draw.ellipse((x + scale * 0.26, top, x + scale * 0.36, top + scale * 0.10), fill="black")
        else:
            draw.ellipse((x + scale * 0.25, bottom + scale * 0.03, x + scale * 0.35, bottom + scale * 0.13), fill="black")
        line([(x + scale * 0.30, top + scale * 0.22), (x + scale * 0.30, bottom)])
        return scale * 0.42

    # Unknown letters use the current font renderer as a fallback, but keep
    # the surrounding jittered spacing so generated cases remain messy.
    draw.text((x, top), character, fill="black")
    return advance


def render_script_case(text: str, seed: int = 42) -> bytes:
    """Render a rough deterministic line-drawn handwriting sample."""

    rng = random.Random(seed)
    scale = 72.0
    image = Image.new("RGB", (max(160, int(len(text) * scale * 0.78 + 96)), 180), "white")
    draw = ImageDraw.Draw(image)
    x = 42.0
    baseline = 112.0
    for index, character in enumerate(text):
        y_shift = rng.uniform(-5.0, 5.0)
        advance = _draw_script_like_character(draw, character, x, baseline + y_shift, scale, rng)
        x += advance + rng.uniform(8.0, 18.0)
        if index < len(text) - 1 and rng.random() < 0.18:
            x += rng.uniform(10.0, 20.0)
    crop = image.crop(image.getbbox() or (0, 0, image.width, image.height))
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    return buffer.getvalue()


def evaluate_cases(cases: list[str] | None = None, all_fonts: bool = False, script_cases: bool = False) -> dict[str, object]:
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
                    exact=display_matches(target, prediction),
                    ambiguity_aware=sequence_matches_with_ambiguity(target, prediction),
                    font=font_name,
                )
            )
    if script_cases:
        for index, target in enumerate(selected_cases):
            payload = render_script_case(target, seed=1000 + index)
            classified = main.classify_files([(f"{target}-script.png", payload)], model, device, save_sources=False)[0]
            prediction = str(classified.get("sequence", ""))
            results.append(
                HardCaseResult(
                    target=target,
                    prediction=prediction,
                    exact=display_matches(target, prediction),
                    ambiguity_aware=sequence_matches_with_ambiguity(target, prediction),
                    font="script",
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


def evaluate_uploaded_fixtures(fixtures: list[dict[str, object]] | None = None) -> dict[str, object]:
    """Run saved real-upload fixtures through the app classifier."""

    selected_fixtures = fixtures if fixtures is not None else UPLOADED_FIXTURES
    model, device = load_web_models()
    results: list[HardCaseResult] = []
    for fixture in selected_fixtures:
        path = Path(str(fixture.get("path", "")))
        target = str(fixture.get("target", ""))
        if not path.exists() or not target:
            continue
        classified = main.classify_files([(path.name, path.read_bytes())], model, device, save_sources=False)[0]
        prediction = str(classified.get("sequence", ""))
        raw_rows = classified.get("raw_row_sequences", [])
        raw_prediction = (
            "\n".join(str(row) for row in raw_rows)
            if isinstance(raw_rows, list) and raw_rows
            else str(classified.get("raw_sequence", prediction))
        )
        prediction_items = classified.get("predictions", [])
        prediction_count = len(prediction_items) if isinstance(prediction_items, list) else None
        results.append(
            HardCaseResult(
                target=target,
                prediction=prediction,
                exact=display_matches(target, prediction),
                ambiguity_aware=sequence_matches_with_ambiguity(target, prediction),
                font="uploaded",
                raw_prediction=raw_prediction,
                raw_rows=[str(row) for row in raw_rows] if isinstance(raw_rows, list) else None,
                prediction_count=prediction_count,
                raw_exact=display_matches(target, raw_prediction),
                raw_ambiguity_aware=sequence_matches_with_ambiguity(target, raw_prediction),
            )
        )
    exact = sum(result.exact for result in results)
    ambiguity = sum(result.ambiguity_aware for result in results)
    raw_exact = sum(bool(result.raw_exact) for result in results)
    raw_ambiguity = sum(bool(result.raw_ambiguity_aware) for result in results)
    return {
        "total": len(results),
        "exact_correct": exact,
        "exact_accuracy": 100.0 * exact / max(len(results), 1),
        "ambiguity_aware_correct": ambiguity,
        "ambiguity_aware_accuracy": 100.0 * ambiguity / max(len(results), 1),
        "raw_exact_correct": raw_exact,
        "raw_exact_accuracy": 100.0 * raw_exact / max(len(results), 1),
        "raw_ambiguity_aware_correct": raw_ambiguity,
        "raw_ambiguity_aware_accuracy": 100.0 * raw_ambiguity / max(len(results), 1),
        "results": [result.__dict__ for result in results],
    }


def main_cli() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Evaluate generated hard-case strings against the web recognizer.")
    parser.add_argument("--case", action="append", default=[], help="Specific case to evaluate; repeatable.")
    parser.add_argument("--all-fonts", action="store_true", help="Evaluate every available configured font.")
    parser.add_argument("--script-cases", action="store_true", help="Also evaluate rough line-drawn handwriting cases.")
    parser.add_argument("--uploaded-fixtures", action="store_true", help="Evaluate saved real-upload fixture images.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    report = (
        evaluate_uploaded_fixtures()
        if args.uploaded_fixtures
        else evaluate_cases(args.case or None, all_fonts=args.all_fonts, script_cases=args.script_cases)
    )
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
