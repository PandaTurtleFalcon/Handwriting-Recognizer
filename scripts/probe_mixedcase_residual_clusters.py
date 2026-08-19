"""Probe mixed-case rerankers for high-volume non-family residual clusters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import MIXEDCASE_LABELS  # noqa: E402
from scripts.calibrate_mixedcase_hybrid import hybrid_predictions  # noqa: E402
from scripts.probe_mixedcase_feature_reranker import (  # noqa: E402
    _fit_tensors,
    _is_promotable,
    _load_hybrid_artifact,
    _metrics,
    _model_outputs,
    _split_tensors,
    apply_family_probe,
    family_features,
    parse_source_groups,
    train_family_probe,
)


DEFAULT_CLUSTERS = ("6bG", "0DO", "4HUA9", "2a")
PROTECTED_METRICS = (
    "case_or_ambiguity_aware_test_accuracy",
    "digit_test_accuracy",
    "upper_test_accuracy",
    "lower_test_accuracy",
)


def cluster_indices(cluster: str, labels: tuple[str, ...] = MIXEDCASE_LABELS) -> tuple[int, ...]:
    """Return mixed-case label indices for one ordered residual cluster."""

    label_to_index = {label: index for index, label in enumerate(labels)}
    return tuple(label_to_index[label] for label in dict.fromkeys(cluster) if label in label_to_index)


def parse_clusters(value: str) -> tuple[str, ...]:
    """Parse comma-separated residual cluster labels."""

    clusters = tuple(part.strip() for part in value.split(",") if part.strip())
    return clusters or DEFAULT_CLUSTERS


def _split_calibration(
    train_targets: torch.Tensor,
    calibration_ratio: float,
    confirmation_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fit, selection, and confirmation indices for train tensors."""

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(int(train_targets.numel()), generator=generator)
    calibration_count = max(
        1,
        min(int(train_targets.numel()) - 1, int(round(train_targets.numel() * calibration_ratio))),
    )
    calibration_indices = order[:calibration_count]
    fit_indices = order[calibration_count:]
    confirmation_count = int(round(calibration_count * confirmation_ratio))
    confirmation_count = max(0, min(calibration_count - 1, confirmation_count))
    selection_count = calibration_count - confirmation_count
    return fit_indices, calibration_indices[:selection_count], calibration_indices[selection_count:]


def _gate_metrics(
    before: dict[str, float],
    after: dict[str, float],
    min_delta: float,
) -> tuple[bool, str | None, float]:
    """Return whether a candidate improves exact while preserving split floors."""

    delta = after["test_accuracy"] - before["test_accuracy"]
    if delta < min_delta:
        return False, "test_delta_below_floor", delta
    for metric in PROTECTED_METRICS:
        if after[metric] < before[metric]:
            return False, f"{metric}_regressed", delta
    return True, None, delta


def apply_cluster_probe(
    predictions: torch.Tensor,
    images: torch.Tensor,
    mixed_outputs: torch.Tensor,
    folded_outputs: torch.Tensor,
    probe,
    source_groups: tuple[str, ...],
) -> torch.Tensor:
    """Return residual-cluster predictions after applying the selected source groups."""

    return apply_family_probe(
        predictions,
        images,
        mixed_outputs,
        folded_outputs,
        probe,
        source_groups=source_groups,
    )


def run_probe(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    train_sample_limit: int | None,
    clusters: tuple[str, ...],
    calibration_ratio: float,
    confirmation_ratio: float,
    min_cluster_delta: float,
    seed: int,
    extra_roots: list[Path] | None = None,
    extra_samples_per_class: int | None = None,
    hidden_units: int = 0,
    source_groups: tuple[str, ...] = ("digit", "upper", "lower"),
) -> dict[str, object]:
    """Train cluster probes on train split and evaluate only confirmed adapters."""

    torch.manual_seed(seed)
    train_images, train_targets = _split_tensors(train=True, sample_limit=train_sample_limit)
    test_images, test_targets = _split_tensors(train=False, sample_limit=None)
    fit_indices, selection_indices, confirmation_indices = _split_calibration(
        train_targets,
        calibration_ratio,
        confirmation_ratio,
        seed,
    )
    fit_images, fit_targets = _fit_tensors(
        train_images[fit_indices],
        train_targets[fit_indices],
        extra_roots or [],
        extra_samples_per_class,
        seed,
    )
    selection_images = train_images[selection_indices]
    selection_targets = train_targets[selection_indices]
    confirmation_images = train_images[confirmation_indices]
    confirmation_targets = train_targets[confirmation_indices]

    fit_mixed, fit_folded = _model_outputs(fit_images, batch_size)
    selection_mixed, selection_folded = _model_outputs(selection_images, batch_size)
    confirmation_mixed, confirmation_folded = (
        _model_outputs(confirmation_images, batch_size)
        if int(confirmation_targets.numel()) > 0
        else (torch.empty((0, len(MIXEDCASE_LABELS))), torch.empty((0, 36)))
    )
    test_mixed, test_folded = _model_outputs(test_images, batch_size)
    artifact = _load_hybrid_artifact()
    selection_predictions = hybrid_predictions(selection_mixed, selection_folded, artifact)
    confirmation_predictions = (
        hybrid_predictions(confirmation_mixed, confirmation_folded, artifact)
        if int(confirmation_targets.numel()) > 0
        else torch.empty((0,), dtype=torch.long)
    )
    base_predictions = hybrid_predictions(test_mixed, test_folded, artifact)
    probe_predictions = base_predictions.clone()

    cluster_reports = []
    skipped = []
    for cluster in clusters:
        indices = cluster_indices(cluster)
        if len(indices) < 2:
            skipped.append(cluster)
            continue
        train_features = family_features(fit_images, fit_mixed, fit_folded, indices)
        probe = train_family_probe(train_features, fit_targets, indices, epochs, learning_rate, hidden_units)
        if probe is None:
            skipped.append(cluster)
            continue

        selection_candidate = apply_cluster_probe(
            selection_predictions,
            selection_images,
            selection_mixed,
            selection_folded,
            probe,
            source_groups,
        )
        selection_before = _metrics(selection_predictions, selection_targets)
        selection_after = _metrics(selection_candidate, selection_targets)
        selection_passed, selection_reason, selection_delta = _gate_metrics(
            selection_before,
            selection_after,
            min_cluster_delta,
        )
        confirmation_delta = None
        confirmation_passed = True
        confirmation_reason = None
        if int(confirmation_targets.numel()) > 0:
            confirmation_candidate = apply_cluster_probe(
                confirmation_predictions,
                confirmation_images,
                confirmation_mixed,
                confirmation_folded,
                probe,
                source_groups,
            )
            confirmation_before = _metrics(confirmation_predictions, confirmation_targets)
            confirmation_after = _metrics(confirmation_candidate, confirmation_targets)
            confirmation_passed, confirmation_reason, confirmation_delta = _gate_metrics(
                confirmation_before,
                confirmation_after,
                min_cluster_delta,
            )
        if not selection_passed:
            cluster_reports.append(
                {
                    "cluster": cluster,
                    "accepted": False,
                    "selection_delta": selection_delta,
                    "confirmation_delta": confirmation_delta,
                    "rejection_reason": f"selection_{selection_reason}",
                }
            )
            continue
        if not confirmation_passed:
            cluster_reports.append(
                {
                    "cluster": cluster,
                    "accepted": False,
                    "selection_delta": selection_delta,
                    "confirmation_delta": confirmation_delta,
                    "rejection_reason": f"confirmation_{confirmation_reason}",
                }
            )
            continue

        before = _metrics(probe_predictions, test_targets)
        candidate_predictions = apply_cluster_probe(
            probe_predictions,
            test_images,
            test_mixed,
            test_folded,
            probe,
            source_groups,
        )
        after = _metrics(candidate_predictions, test_targets)
        test_passed, test_reason, test_delta = _gate_metrics(before, after, min_cluster_delta)
        if not test_passed:
            cluster_reports.append(
                {
                    "cluster": cluster,
                    "accepted": False,
                    "selection_delta": selection_delta,
                    "confirmation_delta": confirmation_delta,
                    "before_test_accuracy": before["test_accuracy"],
                    "after_test_accuracy": after["test_accuracy"],
                    "delta": test_delta,
                    "rejection_reason": f"test_{test_reason}",
                }
            )
            continue
        cluster_reports.append(
            {
                "cluster": cluster,
                "accepted": True,
                "selection_delta": selection_delta,
                "confirmation_delta": confirmation_delta,
                "before_test_accuracy": before["test_accuracy"],
                "after_test_accuracy": after["test_accuracy"],
                "delta": test_delta,
            }
        )
        probe_predictions = candidate_predictions

    base_metrics = _metrics(base_predictions, test_targets)
    reranked_metrics = _metrics(probe_predictions, test_targets)
    return {
        "clusters": cluster_reports,
        "skipped": skipped,
        "base": base_metrics,
        "reranked": reranked_metrics,
        "test_delta": reranked_metrics["test_accuracy"] - base_metrics["test_accuracy"],
        "promotable": _is_promotable(base_metrics, reranked_metrics),
        "train_samples": int(train_targets.numel()),
        "fit_samples": int(fit_targets.numel()),
        "selection_samples": int(selection_targets.numel()),
        "confirmation_samples": int(confirmation_targets.numel()),
        "test_samples": int(test_targets.numel()),
        "extra_roots": [str(path) for path in (extra_roots or [])],
        "extra_samples_per_class": extra_samples_per_class,
        "hidden_units": hidden_units,
        "confirmation_ratio": confirmation_ratio,
        "source_groups": list(source_groups),
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Probe mixed-case non-family residual-cluster reranking.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--train-sample-limit", type=int, default=None)
    parser.add_argument("--clusters", default=",".join(DEFAULT_CLUSTERS))
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--confirmation-ratio", type=float, default=0.5)
    parser.add_argument("--min-cluster-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--extra-samples-per-class", type=int, default=None)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--source-groups", default="digit,upper,lower")
    args = parser.parse_args()
    print(
        json.dumps(
            run_probe(
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                train_sample_limit=args.train_sample_limit,
                clusters=parse_clusters(args.clusters),
                calibration_ratio=args.calibration_ratio,
                confirmation_ratio=args.confirmation_ratio,
                min_cluster_delta=args.min_cluster_delta,
                seed=args.seed,
                extra_roots=args.extra_root,
                extra_samples_per_class=args.extra_samples_per_class,
                hidden_units=args.hidden_units,
                source_groups=parse_source_groups(args.source_groups),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
