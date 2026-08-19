"""Sweep character-family reranker settings without promoting artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.probe_character_family_reranker import (  # noqa: E402
    parse_families,
    parse_label_groups,
    prepare_probe_data,
    run_probe,
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


def parse_source_group_sets(raw: str) -> list[tuple[str, ...] | None]:
    """Parse semicolon-separated source-group sets for a sweep."""

    groups: list[tuple[str, ...] | None] = []
    for item in raw.split(";"):
        stripped = item.strip()
        if stripped in {"", "all", "none"}:
            groups.append(None)
        else:
            groups.append(parse_label_groups(stripped))
    if not groups:
        raise ValueError("At least one source-group set is required.")
    return groups


def compact_probe_report(report: dict[str, object], parameters: dict[str, object]) -> dict[str, object]:
    """Return the stable fields needed to compare one probe run."""

    return {
        "parameters": parameters,
        "promotable": bool(report.get("promotable")),
        "validation_delta": float(report.get("validation_delta", 0.0)),
        "base": report.get("base"),
        "reranked": report.get("reranked"),
        "families": report.get("families"),
        "skipped": report.get("skipped"),
    }


def best_sweep_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the best row, preferring promotable rows and then larger delta."""

    if not rows:
        return None
    return max(rows, key=lambda row: (bool(row["promotable"]), float(row["validation_delta"])))


def run_sweep(
    batch_size: int,
    epochs: list[int],
    learning_rates: list[float],
    hidden_units: list[int],
    families: tuple[str, ...],
    source_group_sets: list[tuple[str, ...] | None],
    probe_confidences: list[float],
    probe_margins: list[float],
    calibration_ratio: float,
    confirmation_ratio: float,
    min_family_delta: float,
    seed: int,
    train_only_extra_roots: tuple[Path, ...] = (),
    include_pixel_features: bool = False,
    include_embedding_features: bool = False,
    max_probe_train_samples: int | None = None,
    mini_batch_size: int | None = None,
    max_runs: int | None = None,
) -> dict[str, object]:
    """Run bounded character-family reranker probes and summarize results."""

    rows: list[dict[str, object]] = []
    probe_data = prepare_probe_data(
        batch_size=batch_size,
        calibration_ratio=calibration_ratio,
        confirmation_ratio=confirmation_ratio,
        seed=seed,
        train_only_extra_roots=train_only_extra_roots,
        include_embedding_features=include_embedding_features,
    )
    all_runs = list(
        product(
            epochs,
            learning_rates,
            hidden_units,
            source_group_sets,
            probe_confidences,
            probe_margins,
        )
    )
    selected_runs = all_runs[:max_runs] if max_runs is not None else all_runs
    for epoch_count, learning_rate, hidden_count, source_groups, probe_confidence, probe_margin in selected_runs:
        parameters = {
            "epochs": epoch_count,
            "learning_rate": learning_rate,
            "hidden_units": hidden_count,
            "source_groups": list(source_groups) if source_groups is not None else None,
            "probe_confidence": probe_confidence,
            "probe_margin": probe_margin,
        }
        report = run_probe(
            batch_size=batch_size,
            epochs=epoch_count,
            learning_rate=learning_rate,
            families=families,
            calibration_ratio=calibration_ratio,
            confirmation_ratio=confirmation_ratio,
            min_family_delta=min_family_delta,
            seed=seed,
            hidden_units=hidden_count,
            source_groups=source_groups,
            train_only_extra_roots=train_only_extra_roots,
            include_pixel_features=include_pixel_features,
            include_embedding_features=include_embedding_features,
            probe_confidence=probe_confidence,
            probe_margin=probe_margin,
            max_probe_train_samples=max_probe_train_samples,
            mini_batch_size=mini_batch_size,
            probe_data=probe_data,
        )
        rows.append(compact_probe_report(report, parameters))
    best = best_sweep_row(rows)
    return {
        "rows": rows,
        "best": best,
        "promotable_count": sum(1 for row in rows if bool(row["promotable"])),
        "families": list(families),
        "calibration_ratio": calibration_ratio,
        "confirmation_ratio": confirmation_ratio,
        "min_family_delta": min_family_delta,
        "seed": seed,
        "train_only_extra_roots": [str(root) for root in train_only_extra_roots],
        "include_pixel_features": include_pixel_features,
        "include_embedding_features": include_embedding_features,
        "max_probe_train_samples": max_probe_train_samples,
        "mini_batch_size": mini_batch_size,
        "planned_runs": len(all_runs),
        "completed_runs": len(rows),
        "truncated": max_runs is not None and len(all_runs) > max_runs,
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Sweep character visual-family reranker probes.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", default="60,120")
    parser.add_argument("--learning-rates", default="0.005,0.01,0.02")
    parser.add_argument("--hidden-units", default="0,32,64")
    parser.add_argument("--families", default="1Ili|!/")
    parser.add_argument("--source-groups", default="letter;punctuation;letter,punctuation;all")
    parser.add_argument("--probe-confidences", default="0.0,0.5")
    parser.add_argument("--probe-margins", default="0.0,0.05")
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--confirmation-ratio", type=float, default=0.5)
    parser.add_argument("--min-family-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--train-only-extra-root",
        action="append",
        default=[],
        help="Extra ASCII folder or .pt tensor cache used only for fitting rerankers.",
    )
    parser.add_argument("--include-pixel-features", action="store_true")
    parser.add_argument("--include-embedding-features", action="store_true")
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
                families=parse_families(args.families),
                source_group_sets=parse_source_group_sets(args.source_groups),
                probe_confidences=parse_float_values(args.probe_confidences),
                probe_margins=parse_float_values(args.probe_margins),
                calibration_ratio=args.calibration_ratio,
                confirmation_ratio=args.confirmation_ratio,
                min_family_delta=args.min_family_delta,
                seed=args.seed,
                train_only_extra_roots=tuple(Path(root) for root in args.train_only_extra_root),
                include_pixel_features=args.include_pixel_features,
                include_embedding_features=args.include_embedding_features,
                max_probe_train_samples=args.max_probe_train_samples,
                mini_batch_size=args.mini_batch_size,
                max_runs=args.max_runs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
