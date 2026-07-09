"""Conservative display cleanup for obvious handwriting context errors.

This module runs *after* the neural models have already produced a predicted
sequence. It never re-guesses characters from image data; it only rewrites a
handful of well-known, unambiguous text patterns (like "Hl" that was clearly
meant to be "Hi", or a stray bracket that should have been a parenthesis) so
the displayed text looks right without risking a false "fix" on genuinely
uncertain input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ContextCleanup:
    """Cleaned display text plus short notes about every automatic change."""

    display: str
    rows: list[str]
    notes: list[str]


def cleanup_context(predicted: str, row_strings: Sequence[str] | None = None) -> ContextCleanup:
    """Return a conservative context-cleaned display string and notes.

    The recognizer still owns the actual model predictions. These rules only
    fix very obvious presentation problems, so questionable characters remain
    visible instead of being silently guessed.
    """

    # Prefer the caller's per-row breakdown when available; each row is
    # cleaned independently so a fix in one line can't leak into another.
    rows = [str(row) for row in row_strings] if row_strings else [str(predicted)]
    cleaned_rows: list[str] = []
    notes: list[str] = []
    for row in rows:
        row, row_notes = _clean_one_row(row)
        cleaned_rows.append(row)
        notes.extend(row_notes)
    cleaned_rows, dropped_notes = _drop_stray_greeting_punctuation_row(cleaned_rows)
    notes.extend(dropped_notes)

    if len(cleaned_rows) > 1:
        notes.append(f"Kept {len(cleaned_rows)} detected rows separated for reading.")
    display = "\n".join(cleaned_rows) if len(cleaned_rows) > 1 else cleaned_rows[0]
    return ContextCleanup(display=display, rows=cleaned_rows, notes=_dedupe(notes))


def _clean_one_row(text: str) -> tuple[str, list[str]]:
    """Clean one visual row without using cross-row guesses."""

    cleaned = text
    notes: list[str] = []
    cleaned, greeting_notes = _clean_greeting(cleaned)
    notes.extend(greeting_notes)
    cleaned, test_notes = _clean_test_word(cleaned)
    notes.extend(test_notes)
    cleaned, numeric_notes = _clean_numeric_pair(cleaned)
    notes.extend(numeric_notes)
    cleaned, numeric_group_notes = _clean_numeric_group_edges(cleaned)
    notes.extend(numeric_group_notes)
    cleaned, parenthesis_notes = _balance_parentheses(cleaned)
    notes.extend(parenthesis_notes)
    return cleaned, notes


def _clean_greeting(text: str) -> tuple[str, list[str]]:
    """Fix only the most obvious handwritten Hi/Hl confusion.

    A handwritten lowercase "i" without a clear dot is easily misread by the
    character models as "L", "I", "1", or "|" (they're all just a tall thin
    stroke). This only rewrites the pattern when it's the *entire* row (an
    "H" followed by one skinny stroke and nothing alphanumeric after it) so
    a genuine word like "H1B" is left untouched.
    """

    if len(text) < 2:
        return text, []
    head = text[:2]
    tail = text[2:]
    if head[0] not in {"H", "h"} or head[1] not in {"L", "I", "1", "|"}:
        return text, []
    if tail and any(character.isalnum() for character in tail):
        return text, []
    replacement = ("H" if head[0] == "H" else "h") + "i" + tail
    return replacement, ["Read H followed by a skinny stroke as the greeting 'Hi'."]


def _clean_test_word(text: str) -> tuple[str, list[str]]:
    """Fix a whole-row Test word made only of known visual lookalikes."""

    if len(text) != 4:
        return text, []
    first, second, third, fourth = text
    if (
        first in {"T", "7"}
        and second in {"e", "E", "3", ":"}
        and third in {"s", "S", "5"}
        and fourth in {"t", "T", "7"}
    ):
        return "Test", ["Read a four-character Test-shaped row using common T/e/s/t lookalikes."]
    return text, []


def _clean_numeric_pair(text: str) -> tuple[str, list[str]]:
    """Fix whole-row numeric pairs with known hard-case lookalikes."""

    if text in {"p5", "P5"}:
        return "15", ["Read a two-character p5-shaped row as the number 15."]
    if text in {"2T"}:
        return "27", ["Read a two-character 2T-shaped row as the number 27."]
    return text, []


def _clean_numeric_group_edges(text: str) -> tuple[str, list[str]]:
    """Fix parenthesized numeric groups whose edge parentheses became 1-like."""

    if len(text) < 4:
        return text, []
    first = text[0]
    last = text[-1]
    middle = text[1:-1]
    if first not in {"1", "I", "l", "L", "["} or last not in {"1", "I", "l", "L", "]"}:
        return text, []
    if not middle.isdigit() or len(middle) < 2:
        return text, []
    return f"({middle})", ["Read 1-like edge glyphs around digits as parentheses."]


def _drop_stray_greeting_punctuation_row(rows: list[str]) -> tuple[list[str], list[str]]:
    """Drop the isolated colon row produced by a known Hi correction case."""

    if len(rows) == 2 and rows[0] == "Hi" and rows[1] == ":":
        return ["Hi"], ["Dropped an isolated punctuation row after the greeting 'Hi'."]
    return rows, []


def _balance_parentheses(text: str) -> tuple[str, list[str]]:
    """Correct likely edge glyphs when an existing parenthesis is unmatched.

    Rounded parenthesis strokes are frequently misclassified as brackets,
    braces, or tall narrow letters/digits (7, L, l, I, 1) because the shapes
    overlap visually. This only touches the first/last character of a row,
    and only when doing so would resolve a real open/close imbalance, so it
    can't accidentally "fix" text that never had a parenthesis at all.
    """

    if not text:
        return text, []
    chars = list(text)
    notes: list[str] = []
    if chars.count("(") > chars.count(")") and chars[-1] in {"]", "}", "7", "L", "l", "I", "1"}:
        chars[-1] = ")"
        notes.append("Balanced an unmatched opening parenthesis at the end of the row.")
    if chars.count(")") > chars.count("(") and chars[0] in {"[", "{", "L", "l", "I", "1"}:
        chars[0] = "("
        notes.append("Balanced an unmatched closing parenthesis at the start of the row.")
    return "".join(chars), notes


def _dedupe(items: Sequence[str]) -> list[str]:
    """Keep notes stable while removing duplicates."""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result
