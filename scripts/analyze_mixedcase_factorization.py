"""Analyze separated identity/type decoding for the mixed-case recognizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import mixedcase_type_logits  # noqa: E402
from scripts.calibrate_mixedcase_hybrid import (  # noqa: E402
    _load_hybrid_artifact,
    _model_outputs,
    hybrid_metrics,
    hybrid_predictions,
)


def factorized_predictions(
    mixed_outputs: torch.Tensor,
    folded_outputs: torch.Tensor,
    base_predictions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return folded-identity/type-decoded predictions plus gate features."""

    folded_probabilities = folded_outputs.softmax(dim=1)
    folded_top2 = folded_probabilities.topk(2, dim=1).values
    folded_predictions = folded_probabilities.argmax(dim=1)
    type_probabilities = mixedcase_type_logits(mixed_outputs).softmax(dim=1)
    type_top2 = type_probabilities.topk(2, dim=1).values
    type_predictions = type_probabilities.argmax(dim=1)
    mixed_digit_predictions = mixed_outputs[:, :10].argmax(dim=1)

    predictions = base_predictions.clone()
    for index, (folded_index, type_index) in enumerate(zip(folded_predictions.tolist(), type_predictions.tolist())):
        if folded_index < 10 or type_index == 0:
            predictions[index] = int(mixed_digit_predictions[index].item()) if type_index == 0 else folded_index
        elif type_index == 1:
            predictions[index] = 10 + (folded_index - 10)
        else:
            predictions[index] = 36 + (folded_index - 10)

    return predictions, {
        "folded_confidence": folded_top2[:, 0],
        "folded_margin": folded_top2[:, 0] - folded_top2[:, 1],
        "type_confidence": type_top2[:, 0],
        "type_margin": type_top2[:, 0] - type_top2[:, 1],
        "type_prediction": type_predictions,
        "folded_prediction": folded_predictions,
    }


def type_targets(targets: torch.Tensor) -> torch.Tensor:
    """Return digit/upper/lower targets in the same order as mixedcase_type_logits."""

    result = torch.zeros_like(targets)
    result[(targets >= 10) & (targets < 36)] = 1
    result[targets >= 36] = 2
    return result


def folded_identity_targets(targets: torch.Tensor) -> torch.Tensor:
    """Return 36-class identity targets for mixed-case labels."""

    result = targets.clone()
    result[result >= 36] -= 26
    return result


def protected_promotable(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    """Return whether candidate metrics preserve every deployed mixed-case split."""

    protected_names = (
        "test_accuracy",
        "case_or_ambiguity_aware_test_accuracy",
        "digit_test_accuracy",
        "upper_test_accuracy",
        "lower_test_accuracy",
    )
    return all(float(candidate[name]) >= float(baseline[name]) for name in protected_names)


def sweep_factorized_gates(
    mixed_outputs: torch.Tensor,
    folded_outputs: torch.Tensor,
    targets: torch.Tensor,
    labels: list[str],
    base_predictions: torch.Tensor,
    folded_confidences: Iterable[float],
    folded_margins: Iterable[float],
    type_confidences: Iterable[float],
    type_margins: Iterable[float],
) -> dict[str, object]:
    """Sweep confidence gates for replacing base predictions with factorized ones."""

    factorized, features = factorized_predictions(mixed_outputs, folded_outputs, base_predictions)
    baseline = hybrid_metrics(base_predictions, targets, labels)
    rows: list[dict[str, object]] = []
    for folded_confidence in folded_confidences:
        for folded_margin in folded_margins:
            for type_confidence in type_confidences:
                for type_margin in type_margins:
                    replace_mask = (
                        (factorized != base_predictions)
                        & (features["folded_confidence"] >= folded_confidence)
                        & (features["folded_margin"] >= folded_margin)
                        & (features["type_confidence"] >= type_confidence)
                        & (features["type_margin"] >= type_margin)
                    )
                    if not bool(replace_mask.any()):
                        continue
                    candidate = base_predictions.clone()
                    candidate[replace_mask] = factorized[replace_mask]
                    metrics = hybrid_metrics(candidate, targets, labels)
                    rows.append(
                        {
                            "folded_confidence": folded_confidence,
                            "folded_margin": folded_margin,
                            "type_confidence": type_confidence,
                            "type_margin": type_margin,
                            "changed": int(replace_mask.sum().item()),
                            "fixed": int(((candidate == targets) & (base_predictions != targets)).sum().item()),
                            "broken": int(((candidate != targets) & (base_predictions == targets)).sum().item()),
                            "metrics": metrics,
                            "promotable": protected_promotable(metrics, baseline),
                            "test_delta": metrics["test_accuracy"] - baseline["test_accuracy"],
                            "balanced_delta": metrics["balanced_group_accuracy"] - baseline["balanced_group_accuracy"],
                        }
                    )
    rows.sort(
        key=lambda row: (
            bool(row["promotable"]),
            float(row["balanced_delta"]),
            float(row["test_delta"]),
            -int(row["changed"]),
        ),
        reverse=True,
    )
    factorized_metrics = hybrid_metrics(factorized, targets, labels)
    type_accuracy = 100.0 * float((features["type_prediction"] == type_targets(targets)).float().mean().item())
    identity_ok = features["folded_prediction"] == folded_identity_targets(targets)
    return {
        "baseline": baseline,
        "factorized": factorized_metrics,
        "factorized_changed": int((factorized != base_predictions).sum().item()),
        "factorized_fixed": int(((factorized == targets) & (base_predictions != targets)).sum().item()),
        "factorized_broken": int(((factorized != targets) & (base_predictions == targets)).sum().item()),
        "factorized_fix_oracle": hybrid_metrics(
            torch.where((factorized == targets) & (base_predictions != targets), factorized, base_predictions),
            targets,
            labels,
        ),
        "folded_identity_accuracy": 100.0 * float(identity_ok.float().mean().item()),
        "type_accuracy": type_accuracy,
        "folded_identity_and_type_accuracy": 100.0
        * float((identity_ok & (features["type_prediction"] == type_targets(targets))).float().mean().item()),
        "rows": rows,
        "promotable_count": sum(1 for row in rows if bool(row["promotable"])),
        "best": rows[0] if rows else None,
    }


def parse_float_values(raw: str) -> list[float]:
    """Parse a comma-separated float grid."""

    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def main() -> None:
    """Run the factorized decoder analysis."""

    parser = argparse.ArgumentParser(description="Analyze mixed-case identity/type factorization.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--folded-confidences", default="0,0.5,0.7,0.85,0.93,0.97")
    parser.add_argument("--folded-margins", default="0,0.1,0.25,0.5,0.8")
    parser.add_argument("--type-confidences", default="0,0.5,0.7,0.85,0.93,0.97")
    parser.add_argument("--type-margins", default="0,0.1,0.25,0.5,0.8")
    args = parser.parse_args()

    mixed_outputs, folded_outputs, targets, labels = _model_outputs(batch_size=args.batch_size)
    base_predictions = hybrid_predictions(mixed_outputs, folded_outputs, _load_hybrid_artifact())
    print(
        json.dumps(
            sweep_factorized_gates(
                mixed_outputs=mixed_outputs,
                folded_outputs=folded_outputs,
                targets=targets,
                labels=labels,
                base_predictions=base_predictions,
                folded_confidences=parse_float_values(args.folded_confidences),
                folded_margins=parse_float_values(args.folded_margins),
                type_confidences=parse_float_values(args.type_confidences),
                type_margins=parse_float_values(args.type_margins),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
