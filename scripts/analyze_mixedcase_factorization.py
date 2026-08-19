"""Analyze separated identity/type decoding for the mixed-case recognizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

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


def subset_metrics(predictions: torch.Tensor, targets: torch.Tensor, labels: list[str], indices: torch.Tensor) -> dict[str, float]:
    """Return hybrid metrics on an indexed subset, falling back to full data when empty."""

    if int(indices.numel()) == 0:
        return hybrid_metrics(predictions, targets, labels)
    return hybrid_metrics(predictions[indices], targets[indices], labels)


def split_indices(sample_count: int, calibration_ratio: float, confirmation_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fit, selection, and confirmation indices for learned-gate diagnostics."""

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(sample_count, generator=generator)
    calibration_count = max(2, min(sample_count - 1, int(round(sample_count * calibration_ratio))))
    confirmation_count = int(round(calibration_count * confirmation_ratio))
    confirmation_count = max(1, min(calibration_count - 1, confirmation_count))
    selection_count = calibration_count - confirmation_count
    return order[calibration_count:], order[:selection_count], order[selection_count:calibration_count]


def replacement_gate_features(
    mixed_outputs: torch.Tensor,
    folded_outputs: torch.Tensor,
    base_predictions: torch.Tensor,
    factorized: torch.Tensor,
    features: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build numeric features for deciding whether a factorized replacement is safe."""

    mixed_probabilities = mixed_outputs.softmax(dim=1)
    base_confidence = mixed_probabilities.gather(1, base_predictions.view(-1, 1)).squeeze(1)
    factorized_confidence = mixed_probabilities.gather(1, factorized.view(-1, 1)).squeeze(1)
    return torch.stack(
        [
            features["folded_confidence"],
            features["folded_margin"],
            features["type_confidence"],
            features["type_margin"],
            base_confidence,
            factorized_confidence,
            factorized_confidence - base_confidence,
            (features["type_prediction"] == 0).float(),
            (features["type_prediction"] == 1).float(),
            (features["type_prediction"] == 2).float(),
        ],
        dim=1,
    )


def train_linear_replacement_gate(
    feature_rows: torch.Tensor,
    replace_targets: torch.Tensor,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train a tiny logistic gate that predicts useful factorized replacements."""

    if int(feature_rows.shape[0]) == 0:
        raise ValueError("At least one replacement candidate is required.")
    torch.manual_seed(seed)
    means = feature_rows.mean(dim=0)
    scales = feature_rows.std(dim=0).clamp_min(1e-5)
    normalized = (feature_rows - means) / scales
    model = nn.Linear(normalized.shape[1], 1)
    positives = float(replace_targets.sum().item())
    negatives = float(replace_targets.numel() - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(normalized).squeeze(1)
        loss = criterion(logits, replace_targets.float())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        weight = model.weight.detach().clone().squeeze(0)
        bias = model.bias.detach().clone()
    return weight / scales, bias - (weight * means / scales).sum()


def learned_gate_scores(feature_rows: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Return replacement probabilities for a trained linear gate."""

    return torch.sigmoid(feature_rows @ weights + bias)


def evaluate_learned_replacement_gate(
    mixed_outputs: torch.Tensor,
    folded_outputs: torch.Tensor,
    targets: torch.Tensor,
    labels: list[str],
    base_predictions: torch.Tensor,
    calibration_ratio: float,
    confirmation_ratio: float,
    seed: int,
    epochs: int,
    learning_rate: float,
    thresholds: Iterable[float],
) -> dict[str, object]:
    """Train and threshold a learned factorized-replacement gate."""

    factorized, features = factorized_predictions(mixed_outputs, folded_outputs, base_predictions)
    changed_mask = factorized != base_predictions
    changed_indices = torch.where(changed_mask)[0]
    if int(changed_indices.numel()) == 0:
        return {"available": False, "reason": "no_factorized_replacements"}
    fit_indices, selection_indices, confirmation_indices = split_indices(
        int(targets.numel()),
        calibration_ratio=calibration_ratio,
        confirmation_ratio=confirmation_ratio,
        seed=seed,
    )
    candidate_features = replacement_gate_features(mixed_outputs, folded_outputs, base_predictions, factorized, features)
    useful_replacement = ((factorized == targets) & (base_predictions != targets)).float()
    fit_candidate_mask = changed_mask[fit_indices]
    fit_candidate_indices = fit_indices[fit_candidate_mask]
    if int(fit_candidate_indices.numel()) == 0:
        return {"available": False, "reason": "no_fit_replacements"}
    weights, bias = train_linear_replacement_gate(
        candidate_features[fit_candidate_indices],
        useful_replacement[fit_candidate_indices],
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    scores = learned_gate_scores(candidate_features, weights, bias)
    baseline = hybrid_metrics(base_predictions, targets, labels)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        replace_mask = changed_mask & (scores >= threshold)
        if not bool(replace_mask.any()):
            continue
        candidate = base_predictions.clone()
        candidate[replace_mask] = factorized[replace_mask]
        selection_metrics = subset_metrics(candidate, targets, labels, selection_indices)
        selection_baseline = subset_metrics(base_predictions, targets, labels, selection_indices)
        confirmation_metrics = subset_metrics(candidate, targets, labels, confirmation_indices)
        confirmation_baseline = subset_metrics(base_predictions, targets, labels, confirmation_indices)
        metrics = hybrid_metrics(candidate, targets, labels)
        rows.append(
            {
                "threshold": threshold,
                "changed": int(replace_mask.sum().item()),
                "fixed": int(((candidate == targets) & (base_predictions != targets)).sum().item()),
                "broken": int(((candidate != targets) & (base_predictions == targets)).sum().item()),
                "selection_promotable": protected_promotable(selection_metrics, selection_baseline),
                "confirmation_promotable": protected_promotable(confirmation_metrics, confirmation_baseline),
                "promotable": protected_promotable(metrics, baseline),
                "metrics": metrics,
                "test_delta": metrics["test_accuracy"] - baseline["test_accuracy"],
                "balanced_delta": metrics["balanced_group_accuracy"] - baseline["balanced_group_accuracy"],
            }
        )
    rows.sort(
        key=lambda row: (
            bool(row["selection_promotable"]) and bool(row["confirmation_promotable"]) and bool(row["promotable"]),
            bool(row["promotable"]),
            float(row["balanced_delta"]),
            float(row["test_delta"]),
            -int(row["changed"]),
        ),
        reverse=True,
    )
    return {
        "available": True,
        "fit_replacements": int(fit_candidate_indices.numel()),
        "selection_samples": int(selection_indices.numel()),
        "confirmation_samples": int(confirmation_indices.numel()),
        "rows": rows,
        "promotable_count": sum(1 for row in rows if bool(row["promotable"])),
        "confirmed_promotable_count": sum(
            1
            for row in rows
            if bool(row["selection_promotable"]) and bool(row["confirmation_promotable"]) and bool(row["promotable"])
        ),
        "best": rows[0] if rows else None,
    }


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
    parser.add_argument("--learned-gate", action="store_true")
    parser.add_argument("--learned-gate-epochs", type=int, default=200)
    parser.add_argument("--learned-gate-learning-rate", type=float, default=0.02)
    parser.add_argument("--learned-gate-thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95,0.98")
    parser.add_argument("--calibration-ratio", type=float, default=0.4)
    parser.add_argument("--confirmation-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260845)
    args = parser.parse_args()

    mixed_outputs, folded_outputs, targets, labels = _model_outputs(batch_size=args.batch_size)
    base_predictions = hybrid_predictions(mixed_outputs, folded_outputs, _load_hybrid_artifact())
    report = sweep_factorized_gates(
        mixed_outputs=mixed_outputs,
        folded_outputs=folded_outputs,
        targets=targets,
        labels=labels,
        base_predictions=base_predictions,
        folded_confidences=parse_float_values(args.folded_confidences),
        folded_margins=parse_float_values(args.folded_margins),
        type_confidences=parse_float_values(args.type_confidences),
        type_margins=parse_float_values(args.type_margins),
    )
    if args.learned_gate:
        report["learned_gate"] = evaluate_learned_replacement_gate(
            mixed_outputs=mixed_outputs,
            folded_outputs=folded_outputs,
            targets=targets,
            labels=labels,
            base_predictions=base_predictions,
            calibration_ratio=args.calibration_ratio,
            confirmation_ratio=args.confirmation_ratio,
            seed=args.seed,
            epochs=args.learned_gate_epochs,
            learning_rate=args.learned_gate_learning_rate,
            thresholds=parse_float_values(args.learned_gate_thresholds),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
