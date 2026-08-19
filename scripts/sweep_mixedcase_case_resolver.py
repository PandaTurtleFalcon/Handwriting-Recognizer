"""Sweep mixed-case case-resolver settings without promoting artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.probe_mixedcase_case_resolver import prepare_case_resolver_data, run_probe_from_data  # noqa: E402


def parse_int_values(raw: str) -> list[int]:
    """Parse a required comma-separated integer list."""

    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def parse_float_values(raw: str) -> list[float]:
    """Parse a required comma-separated float list."""

    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one float value is required.")
    return values


def parse_choice_values(raw: str, choices: tuple[str, ...], label: str) -> list[str]:
    """Parse comma-separated choices and validate them against a small enum."""

    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"At least one {label} value is required.")
    invalid = [value for value in values if value not in choices]
    if invalid:
        raise ValueError(f"Unsupported {label} values: {', '.join(invalid)}")
    return values


def compact_probe_report(report: dict[str, object], parameters: dict[str, object]) -> dict[str, object]:
    """Return the stable fields needed to compare one case-resolver run."""

    final_candidate = report.get("final_selected_candidate")
    final_delta = 0.0
    final_safe = False
    final_floor_failures: list[str] = []
    if isinstance(final_candidate, dict):
        final_delta = float(final_candidate.get("test_delta", 0.0))
        final_safe = bool(final_candidate.get("safe"))
        floor_failures = final_candidate.get("floor_failures", [])
        if isinstance(floor_failures, list):
            final_floor_failures = [str(item) for item in floor_failures]
    selection_rows = report.get("selection_sweep_rows", [])
    best_selection_row = None
    if isinstance(selection_rows, list):
        best_selection_row = max(
            (row for row in selection_rows if isinstance(row, dict)),
            key=lambda row: float(row.get("test_delta", 0.0)),
            default=None,
        )
    return {
        "parameters": parameters,
        "promotable": bool(report.get("promotable")),
        "test_delta": float(report.get("test_delta", 0.0)),
        "final_selected_delta": final_delta,
        "final_selected_safe": final_safe,
        "final_floor_failures": final_floor_failures,
        "base": report.get("base"),
        "resolved": report.get("resolved"),
        "selected_thresholds": report.get("selected_thresholds"),
        "confirmation": report.get("confirmation"),
        "final_selected_candidate": final_candidate,
        "best_selection_row": best_selection_row,
        "selection_safe_sweep_count": report.get("selection_safe_sweep_count"),
        "safe_sweep_count": report.get("safe_sweep_count"),
    }


def best_sweep_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the best row, preferring safe final-test rows and then smaller harm."""

    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            bool(row["promotable"]),
            bool(row["final_selected_safe"]),
            float(row["test_delta"]),
            float(row["final_selected_delta"]),
        ),
    )


def run_sweep(
    batch_size: int,
    train_sample_limit: int | None,
    epochs: list[int],
    learning_rates: list[float],
    hidden_units: list[int],
    objectives: list[str],
    class_weightings: list[str],
    confidence_thresholds: list[float],
    margin_thresholds: list[float],
    calibration_ratio: float,
    confirmation_ratio: float,
    seeds: list[int],
    extra_roots: list[Path],
    extra_samples_per_class: int | None,
    include_embedding_features: bool,
    max_runs: int | None = None,
) -> dict[str, object]:
    """Run bounded confirmed mixed-case case-resolver probes."""

    rows: list[dict[str, object]] = []
    all_runs = list(product(epochs, learning_rates, hidden_units, objectives, class_weightings, seeds))
    selected_runs = all_runs[:max_runs] if max_runs is not None else all_runs
    cached_data = {}
    for epoch_count, learning_rate, hidden_count, objective, class_weighting, seed in selected_runs:
        parameters = {
            "epochs": epoch_count,
            "learning_rate": learning_rate,
            "hidden_units": hidden_count,
            "objective": objective,
            "class_weighting": class_weighting,
            "seed": seed,
        }
        if seed not in cached_data:
            cached_data[seed] = prepare_case_resolver_data(
                batch_size=batch_size,
                train_sample_limit=train_sample_limit,
                seed=seed,
                extra_roots=extra_roots,
                extra_samples_per_class=extra_samples_per_class,
                calibration_ratio=calibration_ratio,
                confirmation_ratio=confirmation_ratio,
                include_embedding_features=include_embedding_features,
            )
        report = run_probe_from_data(
            cached_data[seed],
            epochs=epoch_count,
            learning_rate=learning_rate,
            hidden_units=hidden_count,
            confidence_threshold=0.0,
            margin_threshold=0.0,
            confidence_thresholds=confidence_thresholds,
            margin_thresholds=margin_thresholds,
            seed=seed,
            calibration_ratio=calibration_ratio,
            confirmation_ratio=confirmation_ratio,
            include_embedding_features=include_embedding_features,
            objective=objective,
            class_weighting=class_weighting,
        )
        rows.append(compact_probe_report(report, parameters))
    best = best_sweep_row(rows)
    return {
        "rows": rows,
        "best": best,
        "promotable_count": sum(1 for row in rows if bool(row["promotable"])),
        "final_safe_count": sum(1 for row in rows if bool(row["final_selected_safe"])),
        "train_sample_limit": train_sample_limit,
        "confidence_thresholds": confidence_thresholds,
        "margin_thresholds": margin_thresholds,
        "calibration_ratio": calibration_ratio,
        "confirmation_ratio": confirmation_ratio,
        "extra_roots": [str(root) for root in extra_roots],
        "extra_samples_per_class": extra_samples_per_class,
        "include_embedding_features": include_embedding_features,
        "planned_runs": len(all_runs),
        "completed_runs": len(rows),
        "truncated": max_runs is not None and len(all_runs) > max_runs,
        "cached_seed_count": len(cached_data),
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Sweep confirmed mixed-case case-resolver probes.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--train-sample-limit", type=int, default=60000)
    parser.add_argument("--epochs", default="120,160")
    parser.add_argument("--learning-rates", default="0.003,0.006")
    parser.add_argument("--hidden-units", default="0,64")
    parser.add_argument("--objectives", default="exact,balanced")
    parser.add_argument("--class-weightings", default="none,balanced")
    parser.add_argument("--confidence-thresholds", default="0.0,0.4,0.6,0.75,0.9,0.95")
    parser.add_argument("--margin-thresholds", default="0.0,0.05,0.12,0.2,0.35")
    parser.add_argument("--calibration-ratio", type=float, default=0.25)
    parser.add_argument("--confirmation-ratio", type=float, default=0.5)
    parser.add_argument("--seeds", default="20260832")
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--extra-samples-per-class", type=int, default=None)
    parser.add_argument("--include-embedding-features", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_sweep(
                batch_size=args.batch_size,
                train_sample_limit=args.train_sample_limit,
                epochs=parse_int_values(args.epochs),
                learning_rates=parse_float_values(args.learning_rates),
                hidden_units=parse_int_values(args.hidden_units),
                objectives=parse_choice_values(args.objectives, ("exact", "balanced"), "objective"),
                class_weightings=parse_choice_values(args.class_weightings, ("none", "balanced"), "class weighting"),
                confidence_thresholds=parse_float_values(args.confidence_thresholds),
                margin_thresholds=parse_float_values(args.margin_thresholds),
                calibration_ratio=args.calibration_ratio,
                confirmation_ratio=args.confirmation_ratio,
                seeds=parse_int_values(args.seeds),
                extra_roots=args.extra_root,
                extra_samples_per_class=args.extra_samples_per_class,
                include_embedding_features=args.include_embedding_features,
                max_runs=args.max_runs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
