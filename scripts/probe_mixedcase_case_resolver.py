"""Probe a learned upper/lower case resolver on top of folded identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import LABELS, MIXEDCASE_LABELS, load_mixedcase_extra_cache, limit_mixedcase_extra_cache  # noqa: E402
from scripts.calibrate_mixedcase_hybrid import hybrid_predictions  # noqa: E402
from scripts.probe_mixedcase_feature_reranker import (  # noqa: E402
    _load_hybrid_artifact,
    _metrics,
    _model_outputs,
    _split_tensors,
    geometry_features,
)


def _letter_identity_index(targets: torch.Tensor) -> torch.Tensor:
    """Return folded A-Z identity indices, or -1 for non-letters."""

    identities = torch.full_like(targets, -1)
    upper_mask = (targets >= 10) & (targets < 36)
    lower_mask = (targets >= 36) & (targets < 62)
    identities[upper_mask] = targets[upper_mask] - 10
    identities[lower_mask] = targets[lower_mask] - 36
    return identities


def _folded_letter_predictions(folded_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return folded prediction, confidence, and top-two margin."""

    folded_predictions = folded_outputs.argmax(dim=1)
    folded_confidence = folded_outputs.softmax(dim=1).max(dim=1).values
    top2 = folded_outputs.topk(2, dim=1).values
    folded_margin = top2[:, 0] - top2[:, 1]
    return folded_predictions, folded_confidence, folded_margin


def case_resolver_features(
    images: torch.Tensor,
    mixed_outputs: torch.Tensor,
    folded_outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-sample case features and the folded predicted identity."""

    folded_predictions, folded_confidence, folded_margin = _folded_letter_predictions(folded_outputs)
    folded_identity = folded_predictions - 10
    clamped_identity = folded_identity.clamp(0, 25)
    row_indices = torch.arange(mixed_outputs.shape[0])
    upper_indices = 10 + clamped_identity
    lower_indices = 36 + clamped_identity
    upper_logits = mixed_outputs[row_indices, upper_indices]
    lower_logits = mixed_outputs[row_indices, lower_indices]
    upper_probs = mixed_outputs.softmax(dim=1)[row_indices, upper_indices]
    lower_probs = mixed_outputs.softmax(dim=1)[row_indices, lower_indices]
    identity_one_hot = torch.nn.functional.one_hot(clamped_identity, num_classes=26).float()
    numeric_features = torch.stack(
        (
            upper_logits,
            lower_logits,
            lower_logits - upper_logits,
            upper_probs,
            lower_probs,
            lower_probs - upper_probs,
            folded_confidence,
            folded_margin,
        ),
        dim=1,
    )
    return torch.cat((identity_one_hot, numeric_features, geometry_features(images)), dim=1).float(), folded_predictions


def train_case_resolver(
    features: torch.Tensor,
    targets: torch.Tensor,
    folded_predictions: torch.Tensor,
    hidden_units: int,
    epochs: int,
    learning_rate: float,
) -> nn.Module | None:
    """Train a binary model for upper/lower case when folded identity is right."""

    target_identity = _letter_identity_index(targets)
    eligible = (target_identity >= 0) & (folded_predictions == target_identity + 10)
    if int(eligible.sum().item()) < 16:
        return None
    train_targets = (targets[eligible] >= 36).long()
    if hidden_units > 0:
        model: nn.Module = nn.Sequential(
            nn.Linear(features.shape[1], hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 2),
        )
    else:
        model = nn.Linear(features.shape[1], 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    train_features = features[eligible]
    for _epoch in range(max(1, epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_features), train_targets)
        loss.backward()
        optimizer.step()
    return model.eval()


def apply_case_resolver(
    base_predictions: torch.Tensor,
    features: torch.Tensor,
    folded_predictions: torch.Tensor,
    model: nn.Module,
    confidence_threshold: float,
    margin_threshold: float,
) -> torch.Tensor:
    """Apply learned case labels only when folded identity is alphabetic."""

    with torch.no_grad():
        probabilities = model(features).softmax(dim=1)
    confidence, local_case = probabilities.max(dim=1)
    top2 = probabilities.topk(2, dim=1).values
    case_margin = top2[:, 0] - top2[:, 1]
    eligible = (
        (base_predictions >= 10)
        & (folded_predictions >= 10)
        & (folded_predictions < 36)
        & (confidence >= confidence_threshold)
        & (case_margin >= margin_threshold)
    )
    if not bool(eligible.any()):
        return base_predictions
    identity = folded_predictions[eligible] - 10
    replacements = torch.where(local_case[eligible] == 1, 36 + identity, 10 + identity)
    next_predictions = base_predictions.clone()
    next_predictions[eligible] = replacements
    return next_predictions


def oracle_case_predictions(
    base_predictions: torch.Tensor,
    folded_outputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return an oracle letter-case result on top of deployed predictions."""

    folded_predictions = folded_outputs.argmax(dim=1)
    predictions = base_predictions.clone()
    target_is_lower = targets >= 36
    target_is_letter = _letter_identity_index(targets) >= 0
    letter_mask = target_is_letter & (folded_predictions >= 10) & (folded_predictions < 36)
    predictions[letter_mask & target_is_lower] = folded_predictions[letter_mask & target_is_lower] + 26
    return predictions


def _append_extra_tensors(
    images: torch.Tensor,
    targets: torch.Tensor,
    extra_roots: list[Path],
    extra_samples_per_class: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append optional capped extra tensors for resolver fitting only."""

    image_parts = [images]
    target_parts = [targets]
    for extra_index, extra_root in enumerate(extra_roots):
        extra_images, extra_targets = load_mixedcase_extra_cache(extra_root)
        extra_images, extra_targets = limit_mixedcase_extra_cache(
            extra_images,
            extra_targets,
            extra_samples_per_class,
            seed + extra_index + 9000,
        )
        image_parts.append(extra_images)
        target_parts.append(extra_targets)
    return torch.cat(image_parts), torch.cat(target_parts)


def run_probe(
    batch_size: int,
    train_sample_limit: int | None,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    confidence_threshold: float,
    margin_threshold: float,
    seed: int,
    extra_roots: list[Path] | None = None,
    extra_samples_per_class: int | None = None,
) -> dict[str, object]:
    """Train and evaluate a case-resolver probe without writing artifacts."""

    torch.manual_seed(seed)
    train_images, train_targets = _split_tensors(train=True, sample_limit=train_sample_limit)
    train_images, train_targets = _append_extra_tensors(
        train_images,
        train_targets,
        extra_roots or [],
        extra_samples_per_class,
        seed,
    )
    test_images, test_targets = _split_tensors(train=False, sample_limit=None)
    train_mixed, train_folded = _model_outputs(train_images, batch_size)
    test_mixed, test_folded = _model_outputs(test_images, batch_size)
    train_features, train_folded_predictions = case_resolver_features(train_images, train_mixed, train_folded)
    test_features, test_folded_predictions = case_resolver_features(test_images, test_mixed, test_folded)
    resolver = train_case_resolver(
        train_features,
        train_targets,
        train_folded_predictions,
        hidden_units,
        epochs,
        learning_rate,
    )
    artifact = _load_hybrid_artifact()
    base_predictions = hybrid_predictions(test_mixed, test_folded, artifact)
    oracle_predictions = oracle_case_predictions(base_predictions, test_folded, test_targets)
    target_identity = _letter_identity_index(test_targets)
    folded_identity = test_folded_predictions - 10
    letter_mask = target_identity >= 0
    folded_identity_accuracy = 100.0 * float((folded_identity[letter_mask] == target_identity[letter_mask]).float().mean())
    if resolver is None:
        resolved_predictions = base_predictions
    else:
        resolved_predictions = apply_case_resolver(
            base_predictions,
            test_features,
            test_folded_predictions,
            resolver,
            confidence_threshold,
            margin_threshold,
        )
    base_metrics = _metrics(base_predictions, test_targets)
    resolved_metrics = _metrics(resolved_predictions, test_targets)
    oracle_metrics = _metrics(oracle_predictions, test_targets)
    return {
        "base": base_metrics,
        "resolved": resolved_metrics,
        "oracle": oracle_metrics,
        "folded_letter_identity_accuracy": folded_identity_accuracy,
        "test_delta": resolved_metrics["test_accuracy"] - base_metrics["test_accuracy"],
        "oracle_delta": oracle_metrics["test_accuracy"] - base_metrics["test_accuracy"],
        "promotable": resolved_metrics["test_accuracy"] > base_metrics["test_accuracy"]
        and all(
            resolved_metrics[name] >= base_metrics[name]
            for name in (
                "case_or_ambiguity_aware_test_accuracy",
                "digit_test_accuracy",
                "upper_test_accuracy",
                "lower_test_accuracy",
            )
        ),
        "resolver_trained": resolver is not None,
        "train_samples": int(train_targets.numel()),
        "test_samples": int(test_targets.numel()),
        "hidden_units": hidden_units,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "extra_roots": [str(path) for path in (extra_roots or [])],
        "extra_samples_per_class": extra_samples_per_class,
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Probe a learned mixed-case upper/lower resolver.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--train-sample-limit", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--extra-samples-per-class", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_probe(
                batch_size=args.batch_size,
                train_sample_limit=args.train_sample_limit,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                hidden_units=args.hidden_units,
                confidence_threshold=args.confidence_threshold,
                margin_threshold=args.margin_threshold,
                seed=args.seed,
                extra_roots=args.extra_root,
                extra_samples_per_class=args.extra_samples_per_class,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
