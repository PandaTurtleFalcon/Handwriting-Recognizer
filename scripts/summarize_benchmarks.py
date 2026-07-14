"""Summarize the recognizer's saved benchmark gates against a target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _read_json(path: Path) -> Any:
    """Read a JSON object, returning an empty dict when it is absent."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(name: str, value: float | None, target: float) -> dict[str, object]:
    """Create one pass/fail benchmark row."""

    return {
        "name": name,
        "value": value,
        "target": target,
        "passed": value is not None and value >= target,
    }


def _counted_gate(
    name: str,
    value: float | None,
    target: float,
    correct: object,
    total: object,
) -> dict[str, object]:
    """Create a benchmark row that also carries numerator/denominator counts."""

    row = _gate(name, value, target)
    row["correct"] = int(correct)
    row["total"] = int(total)
    return row


def summarize_saved_metrics(project_dir: Path = PROJECT_DIR, target: float = 95.0) -> list[dict[str, object]]:
    """Return saved model-metric gates for the current checkpoints."""

    digit_metrics = _read_json(project_dir / "training_metrics.json")
    folded_metrics = _read_json(project_dir / "alnum_training_metrics.json")
    mixed_metrics = _read_json(project_dir / "mixedcase_training_metrics.json")
    character_metrics = _read_json(project_dir / "character_training_metrics.json")

    digit_best = _best_checkpoint(digit_metrics)
    folded_best = folded_metrics.get("best_checkpoint", {})
    mixed_best = mixed_metrics.get("best_checkpoint", {})
    character_best = character_metrics.get("best_checkpoint", {})

    return [
        _gate("digit_specialist_exact", _float_or_none(digit_best.get("test_accuracy")), target),
        _gate("folded_alnum_exact", _float_or_none(folded_best.get("test_accuracy")), target),
        _gate("mixedcase_exact", _float_or_none(mixed_best.get("test_accuracy")), target),
        _gate("mixedcase_case_or_visual", _float_or_none(mixed_best.get("case_or_ambiguity_aware_test_accuracy")), target),
        _gate("character_exact", _float_or_none(character_best.get("validation_accuracy")), target),
        _gate("character_ambiguity", _float_or_none(character_best.get("ambiguity_aware_validation_accuracy")), target),
        _gate("punctuation_exact", _float_or_none(character_best.get("punctuation_validation_accuracy")), target),
        _gate(
            "punctuation_ambiguity",
            _float_or_none(character_best.get("punctuation_ambiguity_aware_validation_accuracy")),
            target,
        ),
    ]


def summarize_app_hardcases(target: float = 95.0, all_fonts: bool = True) -> list[dict[str, object]]:
    """Return app-level generated hard-case gates for the live recognizer stack."""

    from scripts.evaluate_hardcases import evaluate_cases

    report = evaluate_cases(all_fonts=all_fonts)
    return [
        _counted_gate(
            "app_hardcase_exact",
            _float_or_none(report.get("exact_accuracy")),
            target,
            report.get("exact_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            "app_hardcase_ambiguity",
            _float_or_none(report.get("ambiguity_aware_accuracy")),
            target,
            report.get("ambiguity_aware_correct", 0),
            report.get("total", 0),
        ),
    ]


def _float_or_none(value: object) -> float | None:
    """Convert metric values to float when possible."""

    if value is None:
        return None
    return float(value)


def _best_checkpoint(metrics: Any) -> dict[str, Any]:
    """Return best-checkpoint data from either modern or legacy metric files."""

    if isinstance(metrics, dict):
        checkpoint = metrics.get("best_checkpoint", {})
        return checkpoint if isinstance(checkpoint, dict) else {}
    if isinstance(metrics, list):
        epoch_rows = [row for row in metrics if isinstance(row, dict) and "test_accuracy" in row]
        if not epoch_rows:
            return {}
        return max(epoch_rows, key=lambda row: float(row.get("test_accuracy", 0)))
    return {}


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Summarize saved recognizer benchmark gates.")
    parser.add_argument("--target", type=float, default=95.0)
    parser.add_argument(
        "--include-app-hardcases",
        action="store_true",
        help="Also run generated app-level hard cases through the live recognizer.",
    )
    parser.add_argument(
        "--single-font-hardcases",
        action="store_true",
        help="Use one font instead of all fonts when --include-app-hardcases is set.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = summarize_saved_metrics(target=args.target)
    if args.include_app_hardcases:
        report.extend(summarize_app_hardcases(target=args.target, all_fonts=not args.single_font_hardcases))
    if args.json:
        print(json.dumps(report, indent=2))
        return
    for item in report:
        value = item["value"]
        value_text = "missing" if value is None else f"{float(value):.2f}%"
        if "correct" in item and "total" in item:
            value_text = f"{value_text} ({int(item['correct'])}/{int(item['total'])})"
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {item['name']}: {value_text} (target {float(item['target']):.2f}%)")


if __name__ == "__main__":
    main()
