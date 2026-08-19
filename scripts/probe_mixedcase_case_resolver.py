"""Probe a learned upper/lower case resolver on top of folded identity."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import LABELS, MIXEDCASE_LABELS, load_mixedcase_extra_cache, limit_mixedcase_extra_cache  # noqa: E402
from character_model import stratified_split_indices  # noqa: E402
from scripts.calibrate_mixedcase_hybrid import hybrid_predictions  # noqa: E402
from scripts.evaluate_mixedcase_candidate import load_tensor_pack  # noqa: E402
from scripts.probe_mixedcase_feature_reranker import (  # noqa: E402
    _load_hybrid_artifact,
    _metrics,
    _model_outputs,
    _model_outputs_with_embeddings,
    _split_tensors,
    geometry_features,
)


@dataclass(frozen=True)
class CaseResolverData:
    """Precomputed tensors shared by one or more case-resolver probes."""

    train_features: torch.Tensor
    train_targets: torch.Tensor
    train_folded_predictions: torch.Tensor
    selection_predictions: torch.Tensor
    selection_targets: torch.Tensor
    selection_features: torch.Tensor
    selection_folded_predictions: torch.Tensor
    confirmation_predictions: torch.Tensor
    confirmation_targets: torch.Tensor
    confirmation_features: torch.Tensor
    confirmation_folded_predictions: torch.Tensor
    base_predictions: torch.Tensor
    test_targets: torch.Tensor
    test_features: torch.Tensor
    test_folded_predictions: torch.Tensor
    test_folded_outputs: torch.Tensor
    fit_case_counts: dict[str, int]
    folded_letter_identity_accuracy: float
    fit_samples: int
    selection_samples: int
    confirmation_samples: int
    test_samples: int
    extra_roots: tuple[Path, ...]
    extra_samples_per_class: int | None
    test_tensor_path: Path | None


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
    embedding_outputs: torch.Tensor | None = None,
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
    parts = [identity_one_hot, numeric_features, geometry_features(images)]
    if embedding_outputs is not None:
        normalized_embeddings = torch.nn.functional.normalize(embedding_outputs.float(), dim=1)
        parts.append(normalized_embeddings)
    return torch.cat(parts, dim=1).float(), folded_predictions


def train_case_resolver(
    features: torch.Tensor,
    targets: torch.Tensor,
    folded_predictions: torch.Tensor,
    hidden_units: int,
    epochs: int,
    learning_rate: float,
    class_weighting: str = "none",
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
    if class_weighting == "balanced":
        counts = torch.bincount(train_targets, minlength=2).float().clamp_min(1.0)
        weights = train_targets.numel() / (2.0 * counts)
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif class_weighting == "none":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported case-resolver class weighting: {class_weighting}")
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


def parse_threshold_values(raw: str) -> list[float]:
    """Parse comma-separated threshold values for resolver sweeps."""

    values = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("At least one threshold value is required.")
    return values


def _resolver_candidate_is_safe(
    base_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> bool:
    """Return whether a resolver candidate improves exact without regressions."""

    return candidate_metrics["test_accuracy"] > base_metrics["test_accuracy"] and all(
        candidate_metrics[name] >= base_metrics[name]
        for name in (
            "case_or_ambiguity_aware_test_accuracy",
            "digit_test_accuracy",
            "upper_test_accuracy",
            "lower_test_accuracy",
        )
    )


def _resolver_objective(metrics: dict[str, float], objective: str) -> float:
    """Score safe resolver rows for threshold selection."""

    if objective == "exact":
        return metrics["test_accuracy"]
    if objective == "balanced":
        return min(
            metrics["test_accuracy"],
            metrics["case_or_ambiguity_aware_test_accuracy"],
            metrics["digit_test_accuracy"],
            metrics["upper_test_accuracy"],
            metrics["lower_test_accuracy"],
        )
    raise ValueError(f"Unsupported resolver objective: {objective}")


def sweep_case_resolver_thresholds(
    base_predictions: torch.Tensor,
    targets: torch.Tensor,
    features: torch.Tensor,
    folded_predictions: torch.Tensor,
    model: nn.Module | None,
    confidence_thresholds: list[float],
    margin_thresholds: list[float],
) -> tuple[torch.Tensor, dict[str, float], list[dict[str, object]]]:
    """Evaluate resolver thresholds and return the best safe predictions."""

    base_metrics = _metrics(base_predictions, targets)
    best_predictions = base_predictions
    best_metrics = base_metrics
    rows: list[dict[str, object]] = []
    if model is None:
        return best_predictions, best_metrics, rows
    for confidence_threshold in confidence_thresholds:
        for margin_threshold in margin_thresholds:
            candidate_predictions = apply_case_resolver(
                base_predictions,
                features,
                folded_predictions,
                model,
                confidence_threshold,
                margin_threshold,
            )
            candidate_metrics = _metrics(candidate_predictions, targets)
            safe = _resolver_candidate_is_safe(base_metrics, candidate_metrics)
            rows.append(
                {
                    "confidence_threshold": confidence_threshold,
                    "margin_threshold": margin_threshold,
                    "safe": safe,
                    "metrics": candidate_metrics,
                    "test_delta": candidate_metrics["test_accuracy"] - base_metrics["test_accuracy"],
                }
            )
            if safe and candidate_metrics["test_accuracy"] > best_metrics["test_accuracy"]:
                best_predictions = candidate_predictions
                best_metrics = candidate_metrics
    return best_predictions, best_metrics, rows


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


def _case_target_counts(targets: torch.Tensor, folded_predictions: torch.Tensor) -> dict[str, int]:
    """Count eligible upper/lower case training labels for resolver diagnostics."""

    target_identity = _letter_identity_index(targets)
    eligible = (target_identity >= 0) & (folded_predictions == target_identity + 10)
    case_targets = (targets[eligible] >= 36).long()
    counts = torch.bincount(case_targets, minlength=2)
    return {"upper": int(counts[0].item()), "lower": int(counts[1].item())}


def _split_fit_selection_confirmation(
    targets: torch.Tensor,
    calibration_ratio: float,
    confirmation_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split training tensors into fit, threshold-selection, and confirmation indices."""

    if not 0.0 < calibration_ratio < 1.0:
        raise ValueError("calibration_ratio must be between 0 and 1.")
    if not 0.0 < confirmation_ratio < 1.0:
        raise ValueError("confirmation_ratio must be between 0 and 1.")
    indices = list(range(int(targets.numel())))
    fit_indices, calibration_indices = stratified_split_indices(
        indices,
        test_size=calibration_ratio,
        random_state=seed,
        stratify=targets.numpy(),
    )
    calibration_targets = targets[torch.tensor(calibration_indices, dtype=torch.long)]
    selection_indices, confirmation_indices = stratified_split_indices(
        list(range(len(calibration_indices))),
        test_size=confirmation_ratio,
        random_state=seed + 1,
        stratify=calibration_targets.numpy(),
    )
    calibration_tensor = torch.tensor(calibration_indices, dtype=torch.long)
    return (
        torch.tensor(fit_indices, dtype=torch.long),
        calibration_tensor[torch.tensor(selection_indices, dtype=torch.long)],
        calibration_tensor[torch.tensor(confirmation_indices, dtype=torch.long)],
    )


def prepare_case_resolver_data(
    batch_size: int,
    train_sample_limit: int | None,
    seed: int,
    extra_roots: list[Path] | None = None,
    extra_samples_per_class: int | None = None,
    test_tensor_path: Path | None = None,
    calibration_ratio: float = 0.2,
    confirmation_ratio: float = 0.5,
    include_embedding_features: bool = False,
) -> CaseResolverData:
    """Precompute fixed split features and deployed predictions for resolver probes."""

    train_images, train_targets = _split_tensors(train=True, sample_limit=train_sample_limit)
    fit_indices, selection_indices, confirmation_indices = _split_fit_selection_confirmation(
        train_targets,
        calibration_ratio,
        confirmation_ratio,
        seed,
    )
    fit_images = train_images[fit_indices]
    fit_targets = train_targets[fit_indices]
    fit_images, fit_targets = _append_extra_tensors(
        fit_images,
        fit_targets,
        extra_roots or [],
        extra_samples_per_class,
        seed,
    )
    selection_images = train_images[selection_indices]
    selection_targets = train_targets[selection_indices]
    confirmation_images = train_images[confirmation_indices]
    confirmation_targets = train_targets[confirmation_indices]
    test_images, test_targets = (
        _split_tensors(train=False, sample_limit=None)
        if test_tensor_path is None
        else load_tensor_pack(test_tensor_path)
    )
    if include_embedding_features:
        train_mixed, train_folded, train_embedding = _model_outputs_with_embeddings(
            fit_images,
            batch_size,
            include_embedding_features=True,
        )
        selection_mixed, selection_folded, selection_embedding = _model_outputs_with_embeddings(
            selection_images,
            batch_size,
            include_embedding_features=True,
        )
        confirmation_mixed, confirmation_folded, confirmation_embedding = _model_outputs_with_embeddings(
            confirmation_images,
            batch_size,
            include_embedding_features=True,
        )
        test_mixed, test_folded, test_embedding = _model_outputs_with_embeddings(
            test_images,
            batch_size,
            include_embedding_features=True,
        )
    else:
        train_mixed, train_folded = _model_outputs(fit_images, batch_size)
        selection_mixed, selection_folded = _model_outputs(selection_images, batch_size)
        confirmation_mixed, confirmation_folded = _model_outputs(confirmation_images, batch_size)
        test_mixed, test_folded = _model_outputs(test_images, batch_size)
        train_embedding = selection_embedding = confirmation_embedding = test_embedding = None
    train_features, train_folded_predictions = case_resolver_features(
        fit_images,
        train_mixed,
        train_folded,
        train_embedding,
    )
    selection_features, selection_folded_predictions = case_resolver_features(
        selection_images,
        selection_mixed,
        selection_folded,
        selection_embedding,
    )
    confirmation_features, confirmation_folded_predictions = case_resolver_features(
        confirmation_images,
        confirmation_mixed,
        confirmation_folded,
        confirmation_embedding,
    )
    test_features, test_folded_predictions = case_resolver_features(
        test_images,
        test_mixed,
        test_folded,
        test_embedding,
    )
    artifact = _load_hybrid_artifact()
    selection_predictions = hybrid_predictions(selection_mixed, selection_folded, artifact)
    confirmation_predictions = hybrid_predictions(confirmation_mixed, confirmation_folded, artifact)
    base_predictions = hybrid_predictions(test_mixed, test_folded, artifact)
    target_identity = _letter_identity_index(test_targets)
    folded_identity = test_folded_predictions - 10
    letter_mask = target_identity >= 0
    folded_letter_identity_accuracy = 100.0 * float(
        (folded_identity[letter_mask] == target_identity[letter_mask]).float().mean()
    )
    return CaseResolverData(
        train_features=train_features,
        train_targets=fit_targets,
        train_folded_predictions=train_folded_predictions,
        selection_predictions=selection_predictions,
        selection_targets=selection_targets,
        selection_features=selection_features,
        selection_folded_predictions=selection_folded_predictions,
        confirmation_predictions=confirmation_predictions,
        confirmation_targets=confirmation_targets,
        confirmation_features=confirmation_features,
        confirmation_folded_predictions=confirmation_folded_predictions,
        base_predictions=base_predictions,
        test_targets=test_targets,
        test_features=test_features,
        test_folded_predictions=test_folded_predictions,
        test_folded_outputs=test_folded,
        fit_case_counts=_case_target_counts(fit_targets, train_folded_predictions),
        folded_letter_identity_accuracy=folded_letter_identity_accuracy,
        fit_samples=int(fit_targets.numel()),
        selection_samples=int(selection_targets.numel()),
        confirmation_samples=int(confirmation_targets.numel()),
        test_samples=int(test_targets.numel()),
        extra_roots=tuple(extra_roots or []),
        extra_samples_per_class=extra_samples_per_class,
        test_tensor_path=test_tensor_path,
    )


def select_confirm_case_resolver_thresholds(
    selection_predictions: torch.Tensor,
    selection_targets: torch.Tensor,
    selection_features: torch.Tensor,
    selection_folded_predictions: torch.Tensor,
    confirmation_predictions: torch.Tensor,
    confirmation_targets: torch.Tensor,
    confirmation_features: torch.Tensor,
    confirmation_folded_predictions: torch.Tensor,
    model: nn.Module | None,
    confidence_thresholds: list[float],
    margin_thresholds: list[float],
    objective: str = "exact",
) -> tuple[dict[str, object] | None, dict[str, object] | None, list[dict[str, object]]]:
    """Pick resolver gates on selection data and require confirmation before test use."""

    _, _, selection_rows = sweep_case_resolver_thresholds(
        selection_predictions,
        selection_targets,
        selection_features,
        selection_folded_predictions,
        model,
        confidence_thresholds,
        margin_thresholds,
    )
    safe_rows = [row for row in selection_rows if bool(row.get("safe"))]
    selected = max(safe_rows, key=lambda row: _resolver_objective(row["metrics"], objective), default=None)
    if selected is None or model is None:
        return None, None, selection_rows
    base_confirmation = _metrics(confirmation_predictions, confirmation_targets)
    best_confirmed: tuple[dict[str, object], dict[str, object]] | None = None
    best_rejected_confirmation: dict[str, object] | None = None
    for row in safe_rows:
        candidate_predictions = apply_case_resolver(
            confirmation_predictions,
            confirmation_features,
            confirmation_folded_predictions,
            model,
            float(row["confidence_threshold"]),
            float(row["margin_threshold"]),
        )
        confirmation_metrics = _metrics(candidate_predictions, confirmation_targets)
        confirmation = {
            "safe": _resolver_candidate_is_safe(base_confirmation, confirmation_metrics),
            "metrics": confirmation_metrics,
            "test_delta": confirmation_metrics["test_accuracy"] - base_confirmation["test_accuracy"],
        }
        if bool(confirmation["safe"]):
            if best_confirmed is None or _resolver_objective(confirmation_metrics, objective) > _resolver_objective(
                best_confirmed[1]["metrics"],
                objective,
            ):
                best_confirmed = (row, confirmation)
        elif (
            best_rejected_confirmation is None
            or float(confirmation["test_delta"]) > float(best_rejected_confirmation["test_delta"])
        ):
            best_rejected_confirmation = confirmation
    if best_confirmed is None:
        return None, best_rejected_confirmation, selection_rows
    return best_confirmed[0], best_confirmed[1], selection_rows


def run_probe_from_data(
    data: CaseResolverData,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    confidence_threshold: float,
    margin_threshold: float,
    seed: int,
    confidence_thresholds: list[float] | None = None,
    margin_thresholds: list[float] | None = None,
    calibration_ratio: float = 0.2,
    confirmation_ratio: float = 0.5,
    include_embedding_features: bool = False,
    objective: str = "exact",
    class_weighting: str = "none",
) -> dict[str, object]:
    """Train and evaluate a case resolver from precomputed split tensors."""

    torch.manual_seed(seed)
    resolver = train_case_resolver(
        data.train_features,
        data.train_targets,
        data.train_folded_predictions,
        hidden_units,
        epochs,
        learning_rate,
        class_weighting,
    )
    oracle_predictions = oracle_case_predictions(data.base_predictions, data.test_folded_outputs, data.test_targets)
    base_metrics = _metrics(data.base_predictions, data.test_targets)
    confidence_values = confidence_thresholds or [confidence_threshold]
    margin_values = margin_thresholds or [margin_threshold]
    selected_thresholds, confirmation, selection_rows = select_confirm_case_resolver_thresholds(
        data.selection_predictions,
        data.selection_targets,
        data.selection_features,
        data.selection_folded_predictions,
        data.confirmation_predictions,
        data.confirmation_targets,
        data.confirmation_features,
        data.confirmation_folded_predictions,
        resolver,
        confidence_values,
        margin_values,
        objective,
    )
    final_selected_candidate: dict[str, object] | None = None
    if resolver is None or selected_thresholds is None:
        resolved_predictions = data.base_predictions
        sweep_rows = []
        resolved_metrics = _metrics(resolved_predictions, data.test_targets)
    else:
        selected_confidence = float(selected_thresholds["confidence_threshold"])
        selected_margin = float(selected_thresholds["margin_threshold"])
        candidate_predictions = apply_case_resolver(
            data.base_predictions,
            data.test_features,
            data.test_folded_predictions,
            resolver,
            selected_confidence,
            selected_margin,
        )
        candidate_metrics = _metrics(candidate_predictions, data.test_targets)
        final_selected_candidate = {
            "confidence_threshold": selected_confidence,
            "margin_threshold": selected_margin,
            "safe": _resolver_candidate_is_safe(base_metrics, candidate_metrics),
            "metrics": candidate_metrics,
            "test_delta": candidate_metrics["test_accuracy"] - base_metrics["test_accuracy"],
        }
        resolved_predictions, resolved_metrics, sweep_rows = sweep_case_resolver_thresholds(
            data.base_predictions,
            data.test_targets,
            data.test_features,
            data.test_folded_predictions,
            resolver,
            [selected_confidence],
            [selected_margin],
        )
    oracle_metrics = _metrics(oracle_predictions, data.test_targets)
    best_sweep_row = max(sweep_rows, key=lambda row: float(row["test_delta"]), default=None)
    return {
        "base": base_metrics,
        "resolved": resolved_metrics,
        "oracle": oracle_metrics,
        "folded_letter_identity_accuracy": data.folded_letter_identity_accuracy,
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
        "fit_samples": data.fit_samples,
        "selection_samples": data.selection_samples,
        "confirmation_samples": data.confirmation_samples,
        "test_samples": data.test_samples,
        "hidden_units": hidden_units,
        "class_weighting": class_weighting,
        "fit_case_counts": data.fit_case_counts,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "selected_thresholds": selected_thresholds,
        "confirmation": confirmation,
        "final_selected_candidate": final_selected_candidate,
        "selection_sweep_count": len(selection_rows),
        "selection_safe_sweep_count": sum(1 for row in selection_rows if bool(row.get("safe"))),
        "safe_sweep_count": sum(1 for row in sweep_rows if bool(row.get("safe"))),
        "best_sweep_row": best_sweep_row,
        "selection_sweep_rows": selection_rows,
        "sweep_rows": sweep_rows,
        "extra_roots": [str(path) for path in data.extra_roots],
        "extra_samples_per_class": data.extra_samples_per_class,
        "test_tensor_path": str(data.test_tensor_path) if data.test_tensor_path is not None else None,
        "calibration_ratio": calibration_ratio,
        "confirmation_ratio": confirmation_ratio,
        "include_embedding_features": include_embedding_features,
        "objective": objective,
    }


def run_probe(
    batch_size: int,
    train_sample_limit: int | None,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    confidence_threshold: float,
    margin_threshold: float,
    seed: int,
    confidence_thresholds: list[float] | None = None,
    margin_thresholds: list[float] | None = None,
    extra_roots: list[Path] | None = None,
    extra_samples_per_class: int | None = None,
    test_tensor_path: Path | None = None,
    calibration_ratio: float = 0.2,
    confirmation_ratio: float = 0.5,
    include_embedding_features: bool = False,
    objective: str = "exact",
    class_weighting: str = "none",
) -> dict[str, object]:
    """Train and evaluate a case-resolver probe without writing artifacts."""

    data = prepare_case_resolver_data(
        batch_size=batch_size,
        train_sample_limit=train_sample_limit,
        seed=seed,
        extra_roots=extra_roots,
        extra_samples_per_class=extra_samples_per_class,
        test_tensor_path=test_tensor_path,
        calibration_ratio=calibration_ratio,
        confirmation_ratio=confirmation_ratio,
        include_embedding_features=include_embedding_features,
    )
    return run_probe_from_data(
        data,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        seed=seed,
        confidence_thresholds=confidence_thresholds,
        margin_thresholds=margin_thresholds,
        calibration_ratio=calibration_ratio,
        confirmation_ratio=confirmation_ratio,
        include_embedding_features=include_embedding_features,
        objective=objective,
        class_weighting=class_weighting,
    )


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
    parser.add_argument("--confidence-thresholds", default=None)
    parser.add_argument("--margin-thresholds", default=None)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--extra-samples-per-class", type=int, default=None)
    parser.add_argument("--test-tensor-path", type=Path, default=None)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--confirmation-ratio", type=float, default=0.5)
    parser.add_argument("--include-embedding-features", action="store_true")
    parser.add_argument("--objective", choices=("exact", "balanced"), default="exact")
    parser.add_argument("--class-weighting", choices=("none", "balanced"), default="none")
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
                confidence_thresholds=(
                    parse_threshold_values(args.confidence_thresholds)
                    if args.confidence_thresholds is not None
                    else None
                ),
                margin_thresholds=(
                    parse_threshold_values(args.margin_thresholds)
                    if args.margin_thresholds is not None
                    else None
                ),
                extra_roots=args.extra_root,
                extra_samples_per_class=args.extra_samples_per_class,
                test_tensor_path=args.test_tensor_path,
                calibration_ratio=args.calibration_ratio,
                confirmation_ratio=args.confirmation_ratio,
                include_embedding_features=args.include_embedding_features,
                objective=args.objective,
                class_weighting=args.class_weighting,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
