"""Sweep mixed-case family-reranker settings without promoting artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.probe_mixedcase_feature_reranker import (  # noqa: E402
    parse_family_names,
    parse_source_groups,
    prepare_feature_probe_data,
    run_probe_from_data,
)


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


def parse_source_group_sets(raw: str) -> list[tuple[str, ...]]:
    """Parse semicolon-separated source-group sets for a sweep."""

    groups = [parse_source_groups(item.strip()) for item in raw.split(";") if item.strip()]
    if not groups:
        raise ValueError("At least one source-group set is required.")
    return groups


def compact_probe_report(report: dict[str, object], parameters: dict[str, object]) -> dict[str, object]:
    """Return the stable fields needed to compare one probe run."""

    return {
        "parameters": parameters,
        "promotable": bool(report.get("promotable")),
        "test_delta": float(report.get("test_delta", 0.0)),
        "balanced_delta": float(report.get("balanced_delta", 0.0)),
        "balanced_score": float(report.get("balanced_score", 0.0)),
        "base": report.get("base"),
        "reranked": report.get("reranked"),
        "families": report.get("families"),
    }


def best_sweep_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the best row, preferring promotable rows then balanced movement."""

    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            bool(row["promotable"]),
            float(row.get("balanced_delta", 0.0)),
            float(row["test_delta"]),
        ),
    )


def run_sweep(
    batch_size: int,
    epochs: list[int],
    learning_rates: list[float],
    hidden_units: list[int],
    source_group_sets: list[tuple[str, ...]],
    probe_confidences: list[float],
    probe_margins: list[float],
    train_sample_limit: int | None,
    family_limit: int | None,
    families: tuple[str, ...] | None,
    calibration_ratio: float,
    confirmation_ratio: float,
    min_family_delta: float,
    seed: int,
    extra_roots: list[Path],
    extra_samples_per_class: int | None,
    include_digit_features: bool,
    include_pixel_features: bool,
    include_embedding_features: bool,
    min_digit: float | None,
    min_upper: float | None,
    min_lower: float | None,
    min_case_or_visual: float | None,
    max_probe_train_samples: int | None = None,
    mini_batch_size: int | None = None,
    max_runs: int | None = None,
) -> dict[str, object]:
    """Run bounded mixed-case family-reranker probes and summarize results."""

    rows: list[dict[str, object]] = []
    all_runs = list(product(epochs, learning_rates, hidden_units, source_group_sets, probe_confidences, probe_margins))
    selected_runs = all_runs[:max_runs] if max_runs is not None else all_runs
    data = prepare_feature_probe_data(
        batch_size=batch_size,
        train_sample_limit=train_sample_limit,
        calibration_ratio=calibration_ratio,
        seed=seed,
        extra_roots=extra_roots,
        extra_samples_per_class=extra_samples_per_class,
        confirmation_ratio=confirmation_ratio,
        include_digit_features=include_digit_features,
        include_embedding_features=include_embedding_features,
    )
    for epoch_count, learning_rate, hidden_count, source_groups, probe_confidence, probe_margin in selected_runs:
        parameters = {
            "epochs": epoch_count,
            "learning_rate": learning_rate,
            "hidden_units": hidden_count,
            "source_groups": list(source_groups),
            "probe_confidence": probe_confidence,
            "probe_margin": probe_margin,
        }
        report = run_probe_from_data(
            data=data,
            epochs=epoch_count,
            learning_rate=learning_rate,
            family_limit=family_limit,
            min_family_delta=min_family_delta,
            seed=seed,
            hidden_units=hidden_count,
            confirmation_ratio=confirmation_ratio,
            family_names=families,
            source_groups=source_groups,
            include_pixel_features=include_pixel_features,
            min_digit=min_digit,
            min_upper=min_upper,
            min_lower=min_lower,
            min_case_or_visual=min_case_or_visual,
            probe_confidence=probe_confidence,
            probe_margin=probe_margin,
            max_probe_train_samples=max_probe_train_samples,
            mini_batch_size=mini_batch_size,
            write=False,
        )
        rows.append(compact_probe_report(report, parameters))
    best = best_sweep_row(rows)
    return {
        "rows": rows,
        "best": best,
        "promotable_count": sum(1 for row in rows if bool(row["promotable"])),
        "families": list(families) if families is not None else None,
        "family_limit": family_limit,
        "calibration_ratio": calibration_ratio,
        "confirmation_ratio": confirmation_ratio,
        "min_family_delta": min_family_delta,
        "seed": seed,
        "extra_roots": [str(root) for root in extra_roots],
        "extra_samples_per_class": extra_samples_per_class,
        "include_digit_features": include_digit_features,
        "include_pixel_features": include_pixel_features,
        "include_embedding_features": include_embedding_features,
        "max_probe_train_samples": max_probe_train_samples,
        "mini_batch_size": mini_batch_size,
        "minimum_gates": {
            "case_or_ambiguity_aware_test_accuracy": min_case_or_visual,
            "digit_test_accuracy": min_digit,
            "upper_test_accuracy": min_upper,
            "lower_test_accuracy": min_lower,
        },
        "planned_runs": len(all_runs),
        "completed_runs": len(rows),
        "truncated": max_runs is not None and len(all_runs) > max_runs,
        "prepared_once": True,
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Sweep mixed-case visual-family reranker probes.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", default="80,160")
    parser.add_argument("--learning-rates", default="0.006,0.01,0.02")
    parser.add_argument("--hidden-units", default="0,64")
    parser.add_argument("--source-groups", default="digit,upper;digit,upper,lower")
    parser.add_argument("--probe-confidences", default="0.0,0.5")
    parser.add_argument("--probe-margins", default="0.0,0.05")
    parser.add_argument("--train-sample-limit", type=int, default=None)
    parser.add_argument("--family-limit", type=int, default=None)
    parser.add_argument("--families", default="", help="Comma-separated visual-family labels to probe explicitly.")
    parser.add_argument("--calibration-ratio", type=float, default=0.25)
    parser.add_argument("--confirmation-ratio", type=float, default=0.5)
    parser.add_argument("--min-family-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--extra-samples-per-class", type=int, default=None)
    parser.add_argument("--include-digit-features", action="store_true")
    parser.add_argument("--include-pixel-features", action="store_true")
    parser.add_argument("--include-embedding-features", action="store_true")
    parser.add_argument("--min-digit", type=float, default=None)
    parser.add_argument("--min-upper", type=float, default=None)
    parser.add_argument("--min-lower", type=float, default=None)
    parser.add_argument("--min-case-or-visual", type=float, default=None)
    parser.add_argument("--max-probe-train-samples", type=int, default=None)
    parser.add_argument("--mini-batch-size", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_sweep(
                batch_size=args.batch_size,
                epochs=parse_int_values(args.epochs),
                learning_rates=parse_float_values(args.learning_rates),
                hidden_units=parse_int_values(args.hidden_units),
                source_group_sets=parse_source_group_sets(args.source_groups),
                probe_confidences=parse_float_values(args.probe_confidences),
                probe_margins=parse_float_values(args.probe_margins),
                train_sample_limit=args.train_sample_limit,
                family_limit=args.family_limit,
                families=parse_family_names(args.families),
                calibration_ratio=args.calibration_ratio,
                confirmation_ratio=args.confirmation_ratio,
                min_family_delta=args.min_family_delta,
                seed=args.seed,
                extra_roots=args.extra_root,
                extra_samples_per_class=args.extra_samples_per_class,
                include_digit_features=args.include_digit_features,
                include_pixel_features=args.include_pixel_features,
                include_embedding_features=args.include_embedding_features,
                min_digit=args.min_digit,
                min_upper=args.min_upper,
                min_lower=args.min_lower,
                min_case_or_visual=args.min_case_or_visual,
                max_probe_train_samples=args.max_probe_train_samples,
                mini_batch_size=args.mini_batch_size,
                max_runs=args.max_runs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
