"""Probe a feature-based reranker for exact mixed-case visual families."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import (  # noqa: E402
    EMNIST_MEAN,
    EMNIST_STD,
    LABELS,
    MIXEDCASE_AMBIGUITY_GROUPS,
    MIXEDCASE_HYBRID_PATH,
    MIXEDCASE_LABELS,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    limit_mixedcase_extra_cache,
    load_alnum_model,
    load_mixedcase_extra_cache,
    load_mixedcase_model,
    mixedcase_labels_match_with_ambiguity,
)
from mnist_model import get_device  # noqa: E402
from scripts.calibrate_mixedcase_hybrid import hybrid_predictions  # noqa: E402


@dataclass(frozen=True)
class FamilyProbe:
    """One trained per-family reranker plus its label index mapping."""

    name: str
    family_indices: tuple[int, ...]
    model: nn.Module


def _family_name(indices: tuple[int, ...]) -> str:
    """Return a readable family name from label indices."""

    return "".join(MIXEDCASE_LABELS[index] for index in indices)


def selected_families(limit: int | None = None) -> list[tuple[int, ...]]:
    """Return ambiguity families that are valid for the 62-class mixed-case model."""

    label_to_index = {label: index for index, label in enumerate(MIXEDCASE_LABELS)}
    families = []
    for group in MIXEDCASE_AMBIGUITY_GROUPS:
        indices = tuple(label_to_index[label] for label in sorted(group) if label in label_to_index)
        if len(indices) > 1:
            families.append(indices)
    return families[:limit] if limit is not None else families


def _load_hybrid_artifact() -> dict[str, object]:
    """Return the deployed hybrid settings, or a disabled default."""

    if not MIXEDCASE_HYBRID_PATH.exists():
        return {"enabled": False}
    try:
        return json.loads(MIXEDCASE_HYBRID_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"enabled": False}


def _split_tensors(train: bool, sample_limit: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MNIST plus EMNIST ByClass tensors for one split."""

    mnist_images, mnist_targets = build_or_load_mnist_cache(train=train)
    byclass_images, byclass_targets = build_or_load_emnist_byclass_mixedcase_cache(train=train)
    images = torch.cat([mnist_images, byclass_images])
    targets = torch.cat([mnist_targets, byclass_targets])
    if sample_limit is None or sample_limit >= int(targets.numel()):
        return images, targets
    generator = torch.Generator().manual_seed(123 if train else 456)
    selected = torch.randperm(int(targets.numel()), generator=generator)[:sample_limit]
    return images[selected], targets[selected]


def _fit_tensors(
    train_images: torch.Tensor,
    train_targets: torch.Tensor,
    extra_roots: list[Path],
    extra_samples_per_class: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reranker fit tensors with optional capped extra datasets."""

    image_parts = [train_images]
    target_parts = [train_targets]
    for extra_index, extra_root in enumerate(extra_roots):
        extra_images, extra_targets = load_mixedcase_extra_cache(extra_root)
        extra_images, extra_targets = limit_mixedcase_extra_cache(
            extra_images,
            extra_targets,
            extra_samples_per_class,
            seed + 3000 + extra_index,
        )
        image_parts.append(extra_images)
        target_parts.append(extra_targets)
    return torch.cat(image_parts), torch.cat(target_parts)


def _model_outputs(images: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return calibrated mixed-case logits and folded logits for images."""

    device = get_device()
    mixed_model, mixed_labels = load_mixedcase_model(device=device, hybrid_path=None)
    folded_model, folded_labels = load_alnum_model(device=device)
    if mixed_model is None or folded_model is None or mixed_labels is None or folded_labels is None:
        raise RuntimeError("Mixed-case and folded alnum checkpoints are required.")
    if list(mixed_labels) != list(MIXEDCASE_LABELS) or list(folded_labels) != list(LABELS):
        raise RuntimeError("Checkpoint labels do not match expected label order.")
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    mixed_outputs: list[torch.Tensor] = []
    folded_outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch_images,) in loader:
            inputs = batch_images.to(device)
            mixed_outputs.append(mixed_model(inputs).cpu())
            folded_outputs.append(folded_model(inputs).cpu())
    return torch.cat(mixed_outputs), torch.cat(folded_outputs)


def geometry_features(images: torch.Tensor) -> torch.Tensor:
    """Extract simple shape features from normalized handwriting tensors."""

    foreground = (images.squeeze(1) * EMNIST_STD + EMNIST_MEAN).clamp(0.0, 1.0)
    mask = foreground > 0.18
    rows = torch.linspace(0.0, 1.0, foreground.shape[1], dtype=torch.float32).view(1, -1, 1)
    cols = torch.linspace(0.0, 1.0, foreground.shape[2], dtype=torch.float32).view(1, 1, -1)
    mass = foreground.sum(dim=(1, 2)).clamp_min(1e-6)
    binary_mass = mask.float().sum(dim=(1, 2)).clamp_min(1.0)
    row_weight = (foreground * rows).sum(dim=(1, 2)) / mass
    col_weight = (foreground * cols).sum(dim=(1, 2)) / mass
    row_var = (foreground * (rows - row_weight.view(-1, 1, 1)).pow(2)).sum(dim=(1, 2)) / mass
    col_var = (foreground * (cols - col_weight.view(-1, 1, 1)).pow(2)).sum(dim=(1, 2)) / mass
    any_row = mask.any(dim=2)
    any_col = mask.any(dim=1)
    height = any_row.float().sum(dim=1) / foreground.shape[1]
    width = any_col.float().sum(dim=1) / foreground.shape[2]
    density = mass / binary_mass
    aspect = width / height.clamp_min(1e-6)
    top_mass = foreground[:, :14, :].sum(dim=(1, 2)) / mass
    bottom_mass = foreground[:, 14:, :].sum(dim=(1, 2)) / mass
    left_mass = foreground[:, :, :14].sum(dim=(1, 2)) / mass
    right_mass = foreground[:, :, 14:].sum(dim=(1, 2)) / mass
    quadrants = torch.stack(
        (
            foreground[:, :14, :14].sum(dim=(1, 2)) / mass,
            foreground[:, :14, 14:].sum(dim=(1, 2)) / mass,
            foreground[:, 14:, :14].sum(dim=(1, 2)) / mass,
            foreground[:, 14:, 14:].sum(dim=(1, 2)) / mass,
        ),
        dim=1,
    )
    vertical_symmetry = (foreground - torch.flip(foreground, dims=(2,))).abs().mean(dim=(1, 2))
    horizontal_symmetry = (foreground - torch.flip(foreground, dims=(1,))).abs().mean(dim=(1, 2))
    center_row = mask[:, 14, :].float()
    center_col = mask[:, :, 14].float()
    row_transitions = (center_row[:, 1:] != center_row[:, :-1]).float().sum(dim=1) / foreground.shape[2]
    col_transitions = (center_col[:, 1:] != center_col[:, :-1]).float().sum(dim=1) / foreground.shape[1]
    inner_mass = foreground[:, 7:21, 7:21].sum(dim=(1, 2)) / mass
    return torch.cat(
        (
            torch.stack(
                (
                    mass / foreground[0].numel(),
                    density,
                    row_weight,
                    col_weight,
                    row_var,
                    col_var,
                    height,
                    width,
                    aspect,
                    top_mass,
                    bottom_mass,
                    left_mass,
                    right_mass,
                    vertical_symmetry,
                    horizontal_symmetry,
                    row_transitions,
                    col_transitions,
                    inner_mass,
                ),
                dim=1,
            ),
            quadrants,
        ),
        dim=1,
    )


def family_features(
    images: torch.Tensor,
    mixed_outputs: torch.Tensor,
    folded_outputs: torch.Tensor,
    family_indices: tuple[int, ...],
) -> torch.Tensor:
    """Build reranker features for one visual family."""

    family_logits = mixed_outputs[:, list(family_indices)]
    family_probs = family_logits.softmax(dim=1)
    folded_parts = []
    for index in family_indices:
        label = MIXEDCASE_LABELS[index]
        if label.isalpha():
            folded_parts.append(folded_outputs[:, 10 + ord(label.upper()) - ord("A")].unsqueeze(1))
        elif label.isdigit():
            folded_parts.append(folded_outputs[:, int(label)].unsqueeze(1))
        else:
            folded_parts.append(torch.zeros((folded_outputs.shape[0], 1), dtype=folded_outputs.dtype))
    folded_logits = torch.cat(folded_parts, dim=1)
    return torch.cat((family_logits, family_probs, folded_logits, geometry_features(images)), dim=1).float()


def train_family_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    family_indices: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    hidden_units: int = 0,
) -> FamilyProbe | None:
    """Train one small classifier for a visual family."""

    target_to_local = {target: index for index, target in enumerate(family_indices)}
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for target in family_indices:
        mask |= targets == target
    if int(mask.sum().item()) < len(family_indices) * 8:
        return None
    local_targets = torch.tensor([target_to_local[int(target)] for target in targets[mask].tolist()], dtype=torch.long)
    if hidden_units > 0:
        model = nn.Sequential(
            nn.Linear(features.shape[1], hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, len(family_indices)),
        )
    else:
        model = nn.Linear(features.shape[1], len(family_indices))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    train_features = features[mask]
    for _epoch in range(max(1, epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_features), local_targets)
        loss.backward()
        optimizer.step()
    return FamilyProbe(_family_name(family_indices), family_indices, model.eval())


def apply_family_probe(
    predictions: torch.Tensor,
    images: torch.Tensor,
    mixed_outputs: torch.Tensor,
    folded_outputs: torch.Tensor,
    probe: FamilyProbe,
) -> torch.Tensor:
    """Return predictions after one family probe replaces in-family guesses."""

    current_in_family = torch.zeros_like(predictions, dtype=torch.bool)
    for family_index in probe.family_indices:
        current_in_family |= predictions == family_index
    if not bool(current_in_family.any()):
        return predictions
    features = family_features(images, mixed_outputs, folded_outputs, probe.family_indices)
    with torch.no_grad():
        local_predictions = probe.model(features[current_in_family]).argmax(dim=1)
    replacements = torch.tensor(
        [probe.family_indices[int(index)] for index in local_predictions.tolist()],
        dtype=torch.long,
    )
    next_predictions = predictions.clone()
    next_predictions[current_in_family] = replacements
    return next_predictions


def _metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Return mixed-case benchmark metrics for probe predictions."""

    exact = predictions == targets
    case_or = torch.tensor(
        [
            mixedcase_labels_match_with_ambiguity(MIXEDCASE_LABELS[int(target)], MIXEDCASE_LABELS[int(prediction)])
            for target, prediction in zip(targets.tolist(), predictions.tolist())
        ],
        dtype=torch.bool,
    )
    is_digit = torch.tensor([label.isdigit() for label in MIXEDCASE_LABELS], dtype=torch.bool)
    is_upper = torch.tensor([label.isupper() for label in MIXEDCASE_LABELS], dtype=torch.bool)
    is_lower = torch.tensor([label.islower() for label in MIXEDCASE_LABELS], dtype=torch.bool)

    def masked(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return 100.0 * float(exact[mask].float().mean().item())

    return {
        "test_accuracy": 100.0 * float(exact.float().mean().item()),
        "case_or_ambiguity_aware_test_accuracy": 100.0 * float(case_or.float().mean().item()),
        "digit_test_accuracy": masked(is_digit[targets]),
        "upper_test_accuracy": masked(is_upper[targets]),
        "lower_test_accuracy": masked(is_lower[targets]),
    }


def _is_promotable(base_metrics: dict[str, float], candidate_metrics: dict[str, float]) -> bool:
    """Return whether a probe improved exact accuracy without split regressions."""

    protected_metrics = (
        "case_or_ambiguity_aware_test_accuracy",
        "digit_test_accuracy",
        "upper_test_accuracy",
        "lower_test_accuracy",
    )
    if candidate_metrics["test_accuracy"] <= base_metrics["test_accuracy"]:
        return False
    return all(candidate_metrics[name] >= base_metrics[name] for name in protected_metrics)


def run_probe(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    train_sample_limit: int | None,
    family_limit: int | None,
    calibration_ratio: float,
    min_family_delta: float,
    seed: int,
    extra_roots: list[Path] | None = None,
    extra_samples_per_class: int | None = None,
    hidden_units: int = 0,
) -> dict[str, object]:
    """Train family probes on train split and evaluate on test split."""

    torch.manual_seed(seed)
    train_images, train_targets = _split_tensors(train=True, sample_limit=train_sample_limit)
    test_images, test_targets = _split_tensors(train=False, sample_limit=None)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(int(train_targets.numel()), generator=generator)
    calibration_count = max(1, min(int(train_targets.numel()) - 1, int(round(train_targets.numel() * calibration_ratio))))
    calibration_indices = order[:calibration_count]
    fit_indices = order[calibration_count:]
    fit_images = train_images[fit_indices]
    fit_targets = train_targets[fit_indices]
    fit_images, fit_targets = _fit_tensors(
        fit_images,
        fit_targets,
        extra_roots or [],
        extra_samples_per_class,
        seed,
    )
    calibration_images = train_images[calibration_indices]
    calibration_targets = train_targets[calibration_indices]
    fit_mixed, fit_folded = _model_outputs(fit_images, batch_size)
    calibration_mixed, calibration_folded = _model_outputs(calibration_images, batch_size)
    test_mixed, test_folded = _model_outputs(test_images, batch_size)
    artifact = _load_hybrid_artifact()
    calibration_predictions = hybrid_predictions(calibration_mixed, calibration_folded, artifact)
    base_predictions = hybrid_predictions(test_mixed, test_folded, artifact)
    probe_predictions = base_predictions.clone()
    family_reports = []
    for family_indices in selected_families(family_limit):
        train_features = family_features(fit_images, fit_mixed, fit_folded, family_indices)
        probe = train_family_probe(train_features, fit_targets, family_indices, epochs, learning_rate, hidden_units)
        if probe is None:
            continue
        calibration_candidate = apply_family_probe(
            calibration_predictions,
            calibration_images,
            calibration_mixed,
            calibration_folded,
            probe,
        )
        calibration_before = _metrics(calibration_predictions, calibration_targets)
        calibration_after = _metrics(calibration_candidate, calibration_targets)
        validation_delta = calibration_after["test_accuracy"] - calibration_before["test_accuracy"]
        if validation_delta < min_family_delta:
            family_reports.append(
                {
                    "family": probe.name,
                    "accepted": False,
                    "validation_delta": validation_delta,
                }
            )
            continue
        before = _metrics(probe_predictions, test_targets)
        candidate_predictions = apply_family_probe(probe_predictions, test_images, test_mixed, test_folded, probe)
        after = _metrics(candidate_predictions, test_targets)
        family_reports.append(
            {
                "family": probe.name,
                "accepted": True,
                "validation_delta": validation_delta,
                "before_test_accuracy": before["test_accuracy"],
                "after_test_accuracy": after["test_accuracy"],
                "delta": after["test_accuracy"] - before["test_accuracy"],
            }
        )
        probe_predictions = candidate_predictions
    base_metrics = _metrics(base_predictions, test_targets)
    reranked_metrics = _metrics(probe_predictions, test_targets)
    return {
        "base": base_metrics,
        "reranked": reranked_metrics,
        "promotable": _is_promotable(base_metrics, reranked_metrics),
        "test_delta": reranked_metrics["test_accuracy"] - base_metrics["test_accuracy"],
        "families": family_reports,
        "train_samples": int(train_targets.numel()),
        "fit_samples": int(fit_targets.numel()),
        "calibration_samples": int(calibration_targets.numel()),
        "test_samples": int(test_targets.numel()),
        "extra_roots": [str(path) for path in (extra_roots or [])],
        "extra_samples_per_class": extra_samples_per_class,
        "hidden_units": hidden_units,
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Probe exact mixed-case visual-family reranking.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--train-sample-limit", type=int, default=None)
    parser.add_argument("--family-limit", type=int, default=None)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--min-family-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--extra-samples-per-class", type=int, default=None)
    parser.add_argument("--hidden-units", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            run_probe(
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                train_sample_limit=args.train_sample_limit,
                family_limit=args.family_limit,
                calibration_ratio=args.calibration_ratio,
                min_family_delta=args.min_family_delta,
                seed=args.seed,
                extra_roots=args.extra_root,
                extra_samples_per_class=args.extra_samples_per_class,
                hidden_units=args.hidden_units,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
