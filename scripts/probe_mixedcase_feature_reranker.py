"""Probe a feature-based reranker for exact mixed-case visual families."""

from __future__ import annotations

import argparse
import hashlib
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
    MIXEDCASE_FAMILY_RERANKER_PATH,
    MIXEDCASE_HYBRID_PATH,
    MIXEDCASE_LABELS,
    MIXEDCASE_LOGIT_BIAS_PATH,
    MIXEDCASE_PAIR_RULES_PATH,
    MIXEDCASE_WEIGHTS_PATH,
    WEIGHTS_PATH,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    limit_mixedcase_extra_cache,
    load_alnum_model,
    load_mixedcase_extra_cache,
    load_mixedcase_model,
    mixedcase_labels_match_with_ambiguity,
)
from mnist_model import MNIST_MEAN, MNIST_STD, load_model as load_digit_model  # noqa: E402
from mnist_model import get_device  # noqa: E402
from scripts.calibrate_mixedcase_hybrid import hybrid_predictions  # noqa: E402


def _file_sha256(path: Path) -> str | None:
    """Return a stable digest for an artifact dependency."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _current_artifact_hashes() -> dict[str, str | None]:
    """Return dependency hashes for the current mixed-case artifact set."""

    return {
        "mixedcase_checkpoint_sha256": _file_sha256(MIXEDCASE_WEIGHTS_PATH),
        "folded_checkpoint_sha256": _file_sha256(WEIGHTS_PATH),
        "mixedcase_logit_bias_sha256": _file_sha256(MIXEDCASE_LOGIT_BIAS_PATH),
        "mixedcase_pair_rules_sha256": _file_sha256(MIXEDCASE_PAIR_RULES_PATH),
        "mixedcase_hybrid_sha256": _file_sha256(MIXEDCASE_HYBRID_PATH),
    }


def _compatible_existing_family_probes(output_path: Path) -> list[dict[str, object]]:
    """Load existing probes that still match the current dependency hashes."""

    if not output_path.exists():
        return []
    try:
        artifact = torch.load(output_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError):
        return []
    if not isinstance(artifact, dict) or not artifact.get("enabled", True):
        return []
    if list(artifact.get("labels", [])) != list(MIXEDCASE_LABELS):
        return []
    current_hashes = _current_artifact_hashes()
    if any(artifact.get(name) != value for name, value in current_hashes.items()):
        return []
    probes = artifact.get("probes", [])
    return [probe for probe in probes if isinstance(probe, dict)]


def merge_family_probe_artifacts(
    existing_probes: list[dict[str, object]],
    accepted_probes: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Merge accepted probes without dropping unrelated existing families."""

    accepted_families = {str(probe.get("family", "")) for probe in accepted_probes}
    kept = [probe for probe in existing_probes if str(probe.get("family", "")) not in accepted_families]
    return [*kept, *accepted_probes]


@dataclass(frozen=True)
class FamilyProbe:
    """One trained per-family reranker plus its label index mapping."""

    name: str
    family_indices: tuple[int, ...]
    model: nn.Module


def _family_name(indices: tuple[int, ...]) -> str:
    """Return a readable family name from label indices."""

    return "".join(MIXEDCASE_LABELS[index] for index in indices)


def selected_families(limit: int | None = None, family_names: tuple[str, ...] | None = None) -> list[tuple[int, ...]]:
    """Return ambiguity families that are valid for the 62-class mixed-case model."""

    label_to_index = {label: index for index, label in enumerate(MIXEDCASE_LABELS)}
    if family_names:
        families = []
        for family_name in family_names:
            indices = tuple(label_to_index[label] for label in dict.fromkeys(family_name) if label in label_to_index)
            if len(indices) > 1:
                families.append(indices)
        return families[:limit] if limit is not None else families
    families = []
    for group in MIXEDCASE_AMBIGUITY_GROUPS:
        indices = tuple(label_to_index[label] for label in sorted(group) if label in label_to_index)
        if len(indices) > 1:
            families.append(indices)
    return families[:limit] if limit is not None else families


def parse_family_names(value: str) -> tuple[str, ...] | None:
    """Parse an optional comma-separated family list."""

    families = tuple(part.strip() for part in value.split(",") if part.strip())
    return families or None


def parse_source_groups(value: str) -> tuple[str, ...]:
    """Parse prediction source groups allowed to receive reranker replacements."""

    groups = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {"digit", "upper", "lower"}
    invalid = sorted(set(groups) - allowed)
    if invalid:
        raise ValueError(f"Unknown source group(s): {', '.join(invalid)}")
    return groups or ("digit", "upper", "lower")


def source_group_mask(predictions: torch.Tensor, groups: tuple[str, ...]) -> torch.Tensor:
    """Return a mask for predictions belonging to the selected source groups."""

    if set(groups) == {"digit", "upper", "lower"}:
        return torch.ones_like(predictions, dtype=torch.bool)
    mask = torch.zeros_like(predictions, dtype=torch.bool)
    if "digit" in groups:
        mask |= predictions < 10
    if "upper" in groups:
        mask |= (predictions >= 10) & (predictions < 36)
    if "lower" in groups:
        mask |= predictions >= 36
    return mask


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


def _digit_outputs(images: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return MNIST digit-specialist logits for EMNIST-normalized tensors."""

    device = get_device()
    digit_model = load_digit_model(device=device)
    foreground = (images * EMNIST_STD + EMNIST_MEAN).clamp(0.0, 1.0)
    digit_inputs = (foreground - MNIST_MEAN) / MNIST_STD
    loader = DataLoader(TensorDataset(digit_inputs), batch_size=batch_size, shuffle=False)
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch_images,) in loader:
            outputs.append(digit_model(batch_images.to(device)).cpu())
    return torch.cat(outputs)


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


def pixel_features(images: torch.Tensor, size: int = 12) -> torch.Tensor:
    """Return a compact foreground-pixel sketch for reranker features."""

    foreground = (images * EMNIST_STD + EMNIST_MEAN).clamp(0.0, 1.0)
    resized = torch.nn.functional.interpolate(
        foreground,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.flatten(start_dim=1).float()


def family_features(
    images: torch.Tensor,
    mixed_outputs: torch.Tensor,
    folded_outputs: torch.Tensor,
    family_indices: tuple[int, ...],
    digit_outputs: torch.Tensor | None = None,
    include_pixel_features: bool = False,
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
    parts = [family_logits, family_probs, folded_logits, geometry_features(images)]
    if include_pixel_features:
        parts.append(pixel_features(images))
    if digit_outputs is not None:
        digit_probs = digit_outputs.softmax(dim=1)
        digit_top2 = digit_probs.topk(2, dim=1).values
        parts.extend(
            (
                digit_outputs,
                digit_probs,
                digit_top2[:, :1],
                (digit_top2[:, :1] - digit_top2[:, 1:2]),
            )
        )
    return torch.cat(parts, dim=1).float()


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
    source_groups: tuple[str, ...] = ("digit", "upper", "lower"),
    digit_outputs: torch.Tensor | None = None,
    probe_confidence: float = 0.0,
    probe_margin: float = 0.0,
    include_pixel_features: bool = False,
) -> torch.Tensor:
    """Return predictions after one family probe replaces in-family guesses."""

    current_in_family = torch.zeros_like(predictions, dtype=torch.bool)
    for family_index in probe.family_indices:
        current_in_family |= predictions == family_index
    current_in_family &= source_group_mask(predictions, source_groups)
    if not bool(current_in_family.any()):
        return predictions
    features = family_features(
        images,
        mixed_outputs,
        folded_outputs,
        probe.family_indices,
        digit_outputs,
        include_pixel_features,
    )
    with torch.no_grad():
        logits = probe.model(features[current_in_family])
        probabilities = logits.softmax(dim=1)
        top2 = probabilities.topk(min(2, probabilities.shape[1]), dim=1)
        local_predictions = top2.indices[:, 0]
        confidence = top2.values[:, 0]
        margin = (
            top2.values[:, 0] - top2.values[:, 1]
            if top2.values.shape[1] > 1
            else torch.ones_like(top2.values[:, 0])
        )
        replace_mask = (confidence >= probe_confidence) & (margin >= probe_margin)
    replacements = torch.tensor(
        [probe.family_indices[int(index)] for index in local_predictions.tolist()],
        dtype=torch.long,
    )
    candidate_indices = torch.where(current_in_family)[0]
    next_predictions = predictions.clone()
    next_predictions[candidate_indices[replace_mask]] = replacements[replace_mask]
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


def _is_promotable(
    base_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    min_digit: float | None = None,
    min_upper: float | None = None,
    min_lower: float | None = None,
    min_case_or_visual: float | None = None,
) -> bool:
    """Return whether a probe improved exact accuracy and preserved gates."""

    if candidate_metrics["test_accuracy"] <= base_metrics["test_accuracy"]:
        return False
    return (
        _final_gate_rejection(
            base_metrics,
            candidate_metrics,
            min_delta=0.0,
            min_digit=min_digit,
            min_upper=min_upper,
            min_lower=min_lower,
            min_case_or_visual=min_case_or_visual,
        )
        is None
    )


def _final_gate_rejection(
    base_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    min_delta: float,
    min_digit: float | None = None,
    min_upper: float | None = None,
    min_lower: float | None = None,
    min_case_or_visual: float | None = None,
) -> str | None:
    """Return why a final-test candidate must be rejected, or None if safe."""

    if candidate_metrics["test_accuracy"] - base_metrics["test_accuracy"] < min_delta:
        return "final_delta_below_floor"
    floors = {
        "case_or_ambiguity_aware_test_accuracy": min_case_or_visual,
        "digit_test_accuracy": min_digit,
        "upper_test_accuracy": min_upper,
        "lower_test_accuracy": min_lower,
    }
    for name, requested_floor in floors.items():
        floor = base_metrics[name] if requested_floor is None else requested_floor
        if candidate_metrics[name] < floor:
            return f"final_{name}_regressed"
    return None


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
    confirmation_ratio: float = 0.5,
    family_names: tuple[str, ...] | None = None,
    source_groups: tuple[str, ...] = ("digit", "upper", "lower"),
    include_digit_features: bool = False,
    include_pixel_features: bool = False,
    min_digit: float | None = None,
    min_upper: float | None = None,
    min_lower: float | None = None,
    min_case_or_visual: float | None = None,
    probe_confidence: float = 0.0,
    probe_margin: float = 0.0,
    output_path: Path = MIXEDCASE_FAMILY_RERANKER_PATH,
    write: bool = False,
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
    confirmation_count = int(round(calibration_count * confirmation_ratio))
    confirmation_count = max(0, min(calibration_count - 1, confirmation_count))
    selection_count = calibration_count - confirmation_count
    selection_indices = calibration_indices[:selection_count]
    confirmation_indices = calibration_indices[selection_count:]
    fit_images = train_images[fit_indices]
    fit_targets = train_targets[fit_indices]
    fit_images, fit_targets = _fit_tensors(
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
    fit_mixed, fit_folded = _model_outputs(fit_images, batch_size)
    selection_mixed, selection_folded = _model_outputs(selection_images, batch_size)
    confirmation_mixed, confirmation_folded = (
        _model_outputs(confirmation_images, batch_size)
        if int(confirmation_targets.numel()) > 0
        else (torch.empty((0, len(MIXEDCASE_LABELS))), torch.empty((0, len(LABELS))))
    )
    test_mixed, test_folded = _model_outputs(test_images, batch_size)
    fit_digit = _digit_outputs(fit_images, batch_size) if include_digit_features else None
    selection_digit = _digit_outputs(selection_images, batch_size) if include_digit_features else None
    confirmation_digit = (
        _digit_outputs(confirmation_images, batch_size)
        if include_digit_features and int(confirmation_targets.numel()) > 0
        else None
    )
    test_digit = _digit_outputs(test_images, batch_size) if include_digit_features else None
    artifact = _load_hybrid_artifact()
    selection_predictions = hybrid_predictions(selection_mixed, selection_folded, artifact)
    confirmation_predictions = (
        hybrid_predictions(confirmation_mixed, confirmation_folded, artifact)
        if int(confirmation_targets.numel()) > 0
        else torch.empty((0,), dtype=torch.long)
    )
    base_predictions = hybrid_predictions(test_mixed, test_folded, artifact)
    probe_predictions = base_predictions.clone()
    family_reports = []
    accepted_probe_artifacts: list[dict[str, object]] = []
    for family_indices in selected_families(family_limit, family_names):
        train_features = family_features(
            fit_images,
            fit_mixed,
            fit_folded,
            family_indices,
            fit_digit,
            include_pixel_features,
        )
        probe = train_family_probe(train_features, fit_targets, family_indices, epochs, learning_rate, hidden_units)
        if probe is None:
            continue
        selection_candidate = apply_family_probe(
            selection_predictions,
            selection_images,
            selection_mixed,
            selection_folded,
            probe,
            source_groups,
            selection_digit,
            probe_confidence,
            probe_margin,
            include_pixel_features,
        )
        selection_before = _metrics(selection_predictions, selection_targets)
        selection_after = _metrics(selection_candidate, selection_targets)
        selection_delta = selection_after["test_accuracy"] - selection_before["test_accuracy"]
        confirmation_delta = None
        if int(confirmation_targets.numel()) > 0:
            confirmation_candidate = apply_family_probe(
                confirmation_predictions,
                confirmation_images,
                confirmation_mixed,
                confirmation_folded,
                probe,
                source_groups,
                confirmation_digit,
                probe_confidence,
                probe_margin,
                include_pixel_features,
            )
            confirmation_before = _metrics(confirmation_predictions, confirmation_targets)
            confirmation_after = _metrics(confirmation_candidate, confirmation_targets)
            confirmation_delta = confirmation_after["test_accuracy"] - confirmation_before["test_accuracy"]
        if selection_delta < min_family_delta:
            family_reports.append(
                {
                    "family": probe.name,
                    "accepted": False,
                    "selection_delta": selection_delta,
                    "confirmation_delta": confirmation_delta,
                    "rejection_reason": "selection_delta_below_floor",
                }
            )
            continue
        if confirmation_delta is not None and confirmation_delta < min_family_delta:
            family_reports.append(
                {
                    "family": probe.name,
                    "accepted": False,
                    "selection_delta": selection_delta,
                    "confirmation_delta": confirmation_delta,
                    "rejection_reason": "confirmation_delta_below_floor",
                }
            )
            continue
        before = _metrics(probe_predictions, test_targets)
        candidate_predictions = apply_family_probe(
            probe_predictions,
            test_images,
            test_mixed,
            test_folded,
            probe,
            source_groups,
            test_digit,
            probe_confidence,
            probe_margin,
            include_pixel_features,
        )
        after = _metrics(candidate_predictions, test_targets)
        final_rejection = _final_gate_rejection(
            before,
            after,
            min_family_delta,
            min_digit=min_digit,
            min_upper=min_upper,
            min_lower=min_lower,
            min_case_or_visual=min_case_or_visual,
        )
        if final_rejection is not None:
            family_reports.append(
                {
                    "family": probe.name,
                    "accepted": False,
                    "selection_delta": selection_delta,
                    "confirmation_delta": confirmation_delta,
                    "before_test_accuracy": before["test_accuracy"],
                    "after_test_accuracy": after["test_accuracy"],
                    "before_metrics": before,
                    "after_metrics": after,
                    "delta": after["test_accuracy"] - before["test_accuracy"],
                    "rejection_reason": final_rejection,
                }
            )
            continue
        family_reports.append(
            {
                "family": probe.name,
                "accepted": True,
                "selection_delta": selection_delta,
                "confirmation_delta": confirmation_delta,
                "before_test_accuracy": before["test_accuracy"],
                "after_test_accuracy": after["test_accuracy"],
                "before_metrics": before,
                "after_metrics": after,
                "delta": after["test_accuracy"] - before["test_accuracy"],
            }
        )
        accepted_probe_artifacts.append(
            {
                "family": probe.name,
                "family_indices": family_indices,
                "state_dict": {key: value.detach().cpu() for key, value in probe.model.state_dict().items()},
                "input_dim": int(train_features.shape[1]),
                "hidden_units": hidden_units,
                "source_groups": source_groups,
                "probe_confidence": probe_confidence,
                "probe_margin": probe_margin,
                "include_digit_features": include_digit_features,
                "include_pixel_features": include_pixel_features,
            }
        )
        probe_predictions = candidate_predictions
    base_metrics = _metrics(base_predictions, test_targets)
    reranked_metrics = _metrics(probe_predictions, test_targets)
    promotable = _is_promotable(
        base_metrics,
        reranked_metrics,
        min_digit=min_digit,
        min_upper=min_upper,
        min_lower=min_lower,
        min_case_or_visual=min_case_or_visual,
    )
    wrote = False
    if write and promotable and accepted_probe_artifacts:
        merged_probe_artifacts = merge_family_probe_artifacts(
            _compatible_existing_family_probes(output_path),
            accepted_probe_artifacts,
        )
        artifact_hashes = _current_artifact_hashes()
        torch.save(
            {
                "enabled": True,
                "source": "mixedcase_feature_family_reranker_probe",
                "labels": list(MIXEDCASE_LABELS),
                "probes": merged_probe_artifacts,
                "best_checkpoint": reranked_metrics,
                "base_checkpoint": base_metrics,
                **artifact_hashes,
            },
            output_path,
        )
        wrote = True
    return {
        "base": base_metrics,
        "reranked": reranked_metrics,
        "promotable": promotable,
        "test_delta": reranked_metrics["test_accuracy"] - base_metrics["test_accuracy"],
        "families": family_reports,
        "train_samples": int(train_targets.numel()),
        "fit_samples": int(fit_targets.numel()),
        "calibration_samples": int(calibration_count),
        "selection_samples": int(selection_targets.numel()),
        "confirmation_samples": int(confirmation_targets.numel()),
        "test_samples": int(test_targets.numel()),
        "extra_roots": [str(path) for path in (extra_roots or [])],
        "extra_samples_per_class": extra_samples_per_class,
        "hidden_units": hidden_units,
        "confirmation_ratio": confirmation_ratio,
        "family_names": list(family_names or []),
        "source_groups": list(source_groups),
        "include_digit_features": include_digit_features,
        "include_pixel_features": include_pixel_features,
        "minimum_gates": {
            "case_or_ambiguity_aware_test_accuracy": min_case_or_visual,
            "digit_test_accuracy": min_digit,
            "upper_test_accuracy": min_upper,
            "lower_test_accuracy": min_lower,
        },
        "probe_thresholds": {
            "confidence": probe_confidence,
            "margin": probe_margin,
        },
        "wrote": wrote,
        "output_path": str(output_path),
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Probe exact mixed-case visual-family reranking.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--train-sample-limit", type=int, default=None)
    parser.add_argument("--family-limit", type=int, default=None)
    parser.add_argument("--families", default="", help="Comma-separated visual-family labels to probe explicitly.")
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--min-family-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--extra-samples-per-class", type=int, default=None)
    parser.add_argument("--hidden-units", type=int, default=0)
    parser.add_argument(
        "--confirmation-ratio",
        type=float,
        default=0.5,
        help="Fraction of held-out calibration samples reserved for a second acceptance check.",
    )
    parser.add_argument(
        "--source-groups",
        default="digit,upper,lower",
        help="Comma-separated current prediction groups eligible for reranking: digit, upper, lower.",
    )
    parser.add_argument(
        "--include-digit-features",
        action="store_true",
        help="Append MNIST digit-specialist logits/probabilities to reranker features.",
    )
    parser.add_argument(
        "--include-pixel-features",
        action="store_true",
        help="Append a compact downsampled foreground-pixel sketch to reranker features.",
    )
    parser.add_argument("--min-digit", type=float, default=None)
    parser.add_argument("--min-upper", type=float, default=None)
    parser.add_argument("--min-lower", type=float, default=None)
    parser.add_argument("--min-case-or-visual", type=float, default=None)
    parser.add_argument("--probe-confidence", type=float, default=0.0)
    parser.add_argument("--probe-margin", type=float, default=0.0)
    parser.add_argument("--output-path", type=Path, default=MIXEDCASE_FAMILY_RERANKER_PATH)
    parser.add_argument("--write", action="store_true")
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
                confirmation_ratio=args.confirmation_ratio,
                family_names=parse_family_names(args.families),
                source_groups=parse_source_groups(args.source_groups),
                include_digit_features=args.include_digit_features,
                include_pixel_features=args.include_pixel_features,
                min_digit=args.min_digit,
                min_upper=args.min_upper,
                min_lower=args.min_lower,
                min_case_or_visual=args.min_case_or_visual,
                probe_confidence=args.probe_confidence,
                probe_margin=args.probe_margin,
                output_path=args.output_path,
                write=args.write,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
