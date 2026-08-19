"""Probe feature rerankers for high-headroom 93-class character families."""

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

from character_model import (  # noqa: E402
    CHAR_MEAN,
    CHAR_STD,
    DATASET_ROOT,
    build_or_load_combined_cache,
    labels_match_with_ambiguity,
    load_character_model,
    load_extra_character_tensors,
    stratified_split_indices,
)
from mnist_model import get_device  # noqa: E402
from scripts.analyze_character_confusions import _metric_extra_roots  # noqa: E402


DEFAULT_FAMILIES = ("1Ili|!/", "0Oo", "5Ss")
PROTECTED_METRICS = (
    "ambiguity_aware_validation_accuracy",
    "digit_validation_accuracy",
    "letter_validation_accuracy",
    "punctuation_validation_accuracy",
)
LABEL_GROUPS = {"digit", "letter", "punctuation"}


@dataclass(frozen=True)
class CharacterFamilyProbe:
    """One trained character-family reranker."""

    name: str
    indices: tuple[int, ...]
    model: nn.Module


@dataclass(frozen=True)
class CharacterProbeData:
    """Precomputed tensors shared by character-family probe settings."""

    labels: list[str]
    fit_images: torch.Tensor
    fit_targets: torch.Tensor
    fit_outputs: torch.Tensor
    fit_embeddings: torch.Tensor | None
    selection_images: torch.Tensor
    selection_targets: torch.Tensor
    selection_outputs: torch.Tensor
    selection_embeddings: torch.Tensor | None
    confirmation_images: torch.Tensor
    confirmation_targets: torch.Tensor
    confirmation_outputs: torch.Tensor
    confirmation_embeddings: torch.Tensor | None
    validation_images: torch.Tensor
    validation_targets: torch.Tensor
    validation_outputs: torch.Tensor
    validation_embeddings: torch.Tensor | None
    train_only_count: int


def parse_families(value: str) -> tuple[str, ...]:
    """Parse comma-separated character-family labels."""

    families = tuple(part.strip() for part in value.split(",") if part.strip())
    return families or DEFAULT_FAMILIES


def parse_label_groups(value: str) -> tuple[str, ...] | None:
    """Parse comma-separated source label groups."""

    groups = tuple(part.strip() for part in value.split(",") if part.strip())
    if not groups:
        return None
    unknown = sorted(set(groups) - LABEL_GROUPS)
    if unknown:
        raise ValueError(f"Unknown label group(s): {', '.join(unknown)}")
    return groups


def family_indices(family: str, labels: list[str]) -> tuple[int, ...]:
    """Return label indices for one ordered family string."""

    label_to_index = {label: index for index, label in enumerate(labels)}
    return tuple(label_to_index[label] for label in dict.fromkeys(family) if label in label_to_index)


def _group(label: str) -> str:
    """Return the benchmark split name for a character label."""

    if label.isdigit():
        return "digit"
    if label.isalpha():
        return "letter"
    return "punctuation"


def _model_outputs(
    model: nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the deployed character model over a tensor batch."""

    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for (batch_images,) in loader:
            outputs.append(model(batch_images.to(device)).cpu())
    return torch.cat(outputs)


def _embedding_extractor(model: nn.Module) -> nn.Module | None:
    """Return a module that emits penultimate activations for known architectures."""

    network = getattr(model, "network", None)
    if isinstance(network, nn.Sequential) and len(network) >= 2:
        modules = list(network.children())
        if isinstance(modules[-1], nn.Linear):
            return nn.Sequential(*modules[:-1]).eval()
    features = getattr(model, "features", None)
    classifier = getattr(model, "classifier", None)
    if isinstance(features, nn.Sequential) and isinstance(classifier, nn.Sequential) and len(classifier) >= 2:
        classifier_modules = list(classifier.children())
        if isinstance(classifier_modules[-1], nn.Linear):
            return nn.Sequential(features, *classifier_modules[:-1]).eval()
    return None


def _model_embeddings(
    model: nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the model body over a tensor batch and normalize its activations."""

    extractor = _embedding_extractor(model)
    if extractor is None:
        raise RuntimeError("The character checkpoint does not expose an embeddable architecture.")
    extractor.to(device)
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for (batch_images,) in loader:
            embeddings.append(extractor(batch_images.to(device)).flatten(start_dim=1).cpu())
    return torch.nn.functional.normalize(torch.cat(embeddings).float(), dim=1)


def geometry_features(images: torch.Tensor) -> torch.Tensor:
    """Extract size-independent shape features from normalized character tensors."""

    foreground = (images.squeeze(1) * CHAR_STD + CHAR_MEAN).clamp(0.0, 1.0)
    height_px = foreground.shape[1]
    width_px = foreground.shape[2]
    half_h = max(1, height_px // 2)
    half_w = max(1, width_px // 2)
    mask = foreground > 0.18
    rows = torch.linspace(0.0, 1.0, height_px, dtype=torch.float32).view(1, -1, 1)
    cols = torch.linspace(0.0, 1.0, width_px, dtype=torch.float32).view(1, 1, -1)
    mass = foreground.sum(dim=(1, 2)).clamp_min(1e-6)
    binary_mass = mask.float().sum(dim=(1, 2)).clamp_min(1.0)
    row_weight = (foreground * rows).sum(dim=(1, 2)) / mass
    col_weight = (foreground * cols).sum(dim=(1, 2)) / mass
    row_var = (foreground * (rows - row_weight.view(-1, 1, 1)).pow(2)).sum(dim=(1, 2)) / mass
    col_var = (foreground * (cols - col_weight.view(-1, 1, 1)).pow(2)).sum(dim=(1, 2)) / mass
    any_row = mask.any(dim=2)
    any_col = mask.any(dim=1)
    box_height = any_row.float().sum(dim=1) / height_px
    box_width = any_col.float().sum(dim=1) / width_px
    density = mass / binary_mass
    aspect = box_width / box_height.clamp_min(1e-6)
    top_mass = foreground[:, :half_h, :].sum(dim=(1, 2)) / mass
    bottom_mass = foreground[:, half_h:, :].sum(dim=(1, 2)) / mass
    left_mass = foreground[:, :, :half_w].sum(dim=(1, 2)) / mass
    right_mass = foreground[:, :, half_w:].sum(dim=(1, 2)) / mass
    quadrants = torch.stack(
        (
            foreground[:, :half_h, :half_w].sum(dim=(1, 2)) / mass,
            foreground[:, :half_h, half_w:].sum(dim=(1, 2)) / mass,
            foreground[:, half_h:, :half_w].sum(dim=(1, 2)) / mass,
            foreground[:, half_h:, half_w:].sum(dim=(1, 2)) / mass,
        ),
        dim=1,
    )
    vertical_symmetry = (foreground - torch.flip(foreground, dims=(2,))).abs().mean(dim=(1, 2))
    horizontal_symmetry = (foreground - torch.flip(foreground, dims=(1,))).abs().mean(dim=(1, 2))
    center_row = mask[:, height_px // 2, :].float()
    center_col = mask[:, :, width_px // 2].float()
    row_transitions = (center_row[:, 1:] != center_row[:, :-1]).float().sum(dim=1) / width_px
    col_transitions = (center_col[:, 1:] != center_col[:, :-1]).float().sum(dim=1) / height_px
    inner = foreground[:, height_px // 4 : (3 * height_px) // 4, width_px // 4 : (3 * width_px) // 4]
    inner_mass = inner.sum(dim=(1, 2)) / mass
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
                    box_height,
                    box_width,
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

    foreground = (images * CHAR_STD + CHAR_MEAN).clamp(0.0, 1.0)
    resized = torch.nn.functional.interpolate(
        foreground,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.flatten(start_dim=1).float()


def family_features(
    images: torch.Tensor,
    outputs: torch.Tensor,
    indices: tuple[int, ...],
    include_pixel_features: bool = False,
    embedding_outputs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build reranker inputs for one character family."""

    family_logits = outputs[:, list(indices)]
    family_probs = family_logits.softmax(dim=1)
    top2 = outputs.topk(2, dim=1).values
    global_features = torch.stack((outputs.max(dim=1).values, top2[:, 0] - top2[:, 1]), dim=1)
    parts = [family_logits, family_probs, global_features, geometry_features(images)]
    if include_pixel_features:
        parts.append(pixel_features(images))
    if embedding_outputs is not None:
        parts.append(embedding_outputs.float())
    return torch.cat(parts, dim=1).float()


def train_family_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    indices: tuple[int, ...],
    labels: list[str],
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    max_train_samples: int | None = None,
    mini_batch_size: int | None = None,
    seed: int = 20260819,
) -> CharacterFamilyProbe | None:
    """Train one small classifier over samples whose true label is in the family."""

    target_to_local = {target: local for local, target in enumerate(indices)}
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for target in indices:
        mask |= targets == target
    selected_indices = torch.where(mask)[0]
    if max_train_samples is not None and int(selected_indices.numel()) > max_train_samples:
        generator = torch.Generator().manual_seed(seed)
        capped_parts = []
        per_label = max(1, max_train_samples // max(len(indices), 1))
        for target in indices:
            target_indices = selected_indices[targets[selected_indices] == target]
            if int(target_indices.numel()) == 0:
                continue
            order = torch.randperm(int(target_indices.numel()), generator=generator)
            capped_parts.append(target_indices[order[:per_label]])
        if capped_parts:
            selected_indices = torch.cat(capped_parts)
        if int(selected_indices.numel()) > max_train_samples:
            order = torch.randperm(int(selected_indices.numel()), generator=generator)
            selected_indices = selected_indices[order[:max_train_samples]]
    if int(selected_indices.numel()) < len(indices) * 8:
        return None
    local_targets = torch.tensor(
        [target_to_local[int(target)] for target in targets[selected_indices].tolist()],
        dtype=torch.long,
    )
    if hidden_units > 0:
        model = nn.Sequential(
            nn.Linear(features.shape[1], hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, len(indices)),
        )
    else:
        model = nn.Linear(features.shape[1], len(indices))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    train_features = features[selected_indices]
    effective_batch_size = (
        int(train_features.shape[0])
        if mini_batch_size is None or mini_batch_size <= 0
        else min(int(mini_batch_size), int(train_features.shape[0]))
    )
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(max(1, epochs)):
        order = torch.randperm(int(train_features.shape[0]), generator=generator)
        for start in range(0, int(train_features.shape[0]), effective_batch_size):
            batch_indices = order[start : start + effective_batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_features[batch_indices]), local_targets[batch_indices])
            loss.backward()
            optimizer.step()
    name = "".join(labels[index] for index in indices)
    return CharacterFamilyProbe(name=name, indices=indices, model=model.eval())


def apply_family_probe(
    predictions: torch.Tensor,
    images: torch.Tensor,
    outputs: torch.Tensor,
    probe: CharacterFamilyProbe,
    labels: list[str],
    source_groups: tuple[str, ...] | None = None,
    include_pixel_features: bool = False,
    embedding_outputs: torch.Tensor | None = None,
    probe_confidence: float = 0.0,
    probe_margin: float = 0.0,
) -> torch.Tensor:
    """Replace predictions only when the current label is inside the family."""

    current_in_family = torch.zeros_like(predictions, dtype=torch.bool)
    for family_index in probe.indices:
        current_in_family |= predictions == family_index
    if source_groups is not None:
        allowed_source = torch.tensor(
            [_group(label) in source_groups for label in labels],
            dtype=torch.bool,
        )
        current_in_family &= allowed_source[predictions]
    if not bool(current_in_family.any()):
        return predictions
    features = family_features(
        images,
        outputs,
        probe.indices,
        include_pixel_features=include_pixel_features,
        embedding_outputs=embedding_outputs,
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
    replacements = torch.tensor([probe.indices[int(index)] for index in local_predictions.tolist()], dtype=torch.long)
    candidate_indices = torch.where(current_in_family)[0]
    next_predictions = predictions.clone()
    next_predictions[candidate_indices[replace_mask]] = replacements[replace_mask]
    return next_predictions


def _metrics(predictions: torch.Tensor, targets: torch.Tensor, labels: list[str]) -> dict[str, float]:
    """Return character benchmark metrics for one prediction tensor."""

    exact = predictions == targets
    ambiguity = torch.tensor(
        [
            labels_match_with_ambiguity(labels[int(target)], labels[int(prediction)])
            for target, prediction in zip(targets.tolist(), predictions.tolist())
        ],
        dtype=torch.bool,
    )
    split_masks = {
        "digit": torch.tensor([_group(label) == "digit" for label in labels], dtype=torch.bool),
        "letter": torch.tensor([_group(label) == "letter" for label in labels], dtype=torch.bool),
        "punctuation": torch.tensor([_group(label) == "punctuation" for label in labels], dtype=torch.bool),
    }

    def masked_accuracy(mask: torch.Tensor) -> float:
        target_mask = mask[targets]
        if not bool(target_mask.any()):
            return 0.0
        return 100.0 * float(exact[target_mask].float().mean().item())

    return {
        "validation_accuracy": 100.0 * float(exact.float().mean().item()),
        "ambiguity_aware_validation_accuracy": 100.0 * float(ambiguity.float().mean().item()),
        "digit_validation_accuracy": masked_accuracy(split_masks["digit"]),
        "letter_validation_accuracy": masked_accuracy(split_masks["letter"]),
        "punctuation_validation_accuracy": masked_accuracy(split_masks["punctuation"]),
    }


def _gate_metrics(
    before: dict[str, float],
    after: dict[str, float],
    min_delta: float,
) -> tuple[bool, str | None, float]:
    """Return whether a candidate improves exact while preserving split floors."""

    delta = after["validation_accuracy"] - before["validation_accuracy"]
    if delta < min_delta:
        return False, "validation_delta_below_floor", delta
    for metric in PROTECTED_METRICS:
        if after[metric] < before[metric]:
            return False, f"{metric}_regressed", delta
    return True, None, delta


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


def _character_tensors() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return deployed character tensors with the metric extra roots."""

    return build_or_load_combined_cache(DATASET_ROOT, _metric_extra_roots())


def prepare_probe_data(
    batch_size: int,
    calibration_ratio: float,
    confirmation_ratio: float,
    seed: int,
    train_only_extra_roots: tuple[Path, ...] = (),
    include_embedding_features: bool = False,
) -> CharacterProbeData:
    """Precompute fixed tensors and model outputs for character-family probes."""

    device = get_device()
    model, labels = load_character_model(device=device)
    images, targets, cache_labels = _character_tensors()
    if list(cache_labels) != list(labels):
        raise RuntimeError("Character cache labels do not match deployed checkpoint labels.")
    indices = list(range(len(targets)))
    train_indices, validation_indices = stratified_split_indices(
        indices,
        test_size=0.15,
        random_state=42,
        stratify=targets.numpy(),
    )
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long)
    validation_index_tensor = torch.tensor(validation_indices, dtype=torch.long)
    train_images = images[train_index_tensor]
    train_targets = targets[train_index_tensor]
    fit_indices, selection_indices, confirmation_indices = _split_calibration(
        train_targets,
        calibration_ratio,
        confirmation_ratio,
        seed,
    )
    fit_images = train_images[fit_indices]
    fit_targets = train_targets[fit_indices]
    train_only_count = 0
    train_only_extra = load_extra_character_tensors(list(train_only_extra_roots), labels)
    if train_only_extra is not None:
        extra_images, extra_targets = train_only_extra
        train_only_count = int(extra_targets.numel())
        fit_images = torch.cat((fit_images, extra_images), dim=0)
        fit_targets = torch.cat((fit_targets, extra_targets), dim=0)
    selection_images = train_images[selection_indices]
    selection_targets = train_targets[selection_indices]
    confirmation_images = train_images[confirmation_indices]
    confirmation_targets = train_targets[confirmation_indices]
    validation_images = images[validation_index_tensor]
    validation_targets = targets[validation_index_tensor]

    fit_outputs = _model_outputs(model, fit_images, batch_size, device)
    selection_outputs = _model_outputs(model, selection_images, batch_size, device)
    confirmation_outputs = (
        _model_outputs(model, confirmation_images, batch_size, device)
        if int(confirmation_targets.numel()) > 0
        else torch.empty((0, len(labels)))
    )
    validation_outputs = _model_outputs(model, validation_images, batch_size, device)
    fit_embeddings = _model_embeddings(model, fit_images, batch_size, device) if include_embedding_features else None
    selection_embeddings = (
        _model_embeddings(model, selection_images, batch_size, device) if include_embedding_features else None
    )
    confirmation_embeddings = (
        _model_embeddings(model, confirmation_images, batch_size, device)
        if include_embedding_features and int(confirmation_targets.numel()) > 0
        else None
    )
    validation_embeddings = (
        _model_embeddings(model, validation_images, batch_size, device) if include_embedding_features else None
    )

    return CharacterProbeData(
        labels=labels,
        fit_images=fit_images,
        fit_targets=fit_targets,
        fit_outputs=fit_outputs,
        fit_embeddings=fit_embeddings,
        selection_images=selection_images,
        selection_targets=selection_targets,
        selection_outputs=selection_outputs,
        selection_embeddings=selection_embeddings,
        confirmation_images=confirmation_images,
        confirmation_targets=confirmation_targets,
        confirmation_outputs=confirmation_outputs,
        confirmation_embeddings=confirmation_embeddings,
        validation_images=validation_images,
        validation_targets=validation_targets,
        validation_outputs=validation_outputs,
        validation_embeddings=validation_embeddings,
        train_only_count=train_only_count,
    )


def run_probe(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    families: tuple[str, ...],
    calibration_ratio: float,
    confirmation_ratio: float,
    min_family_delta: float,
    seed: int,
    hidden_units: int,
    source_groups: tuple[str, ...] | None = None,
    train_only_extra_roots: tuple[Path, ...] = (),
    include_pixel_features: bool = False,
    include_embedding_features: bool = False,
    probe_confidence: float = 0.0,
    probe_margin: float = 0.0,
    max_probe_train_samples: int | None = None,
    mini_batch_size: int | None = None,
    probe_data: CharacterProbeData | None = None,
) -> dict[str, object]:
    """Train confirmed family rerankers and evaluate them on validation."""

    torch.manual_seed(seed)
    data = probe_data or prepare_probe_data(
        batch_size=batch_size,
        calibration_ratio=calibration_ratio,
        confirmation_ratio=confirmation_ratio,
        seed=seed,
        train_only_extra_roots=train_only_extra_roots,
        include_embedding_features=include_embedding_features,
    )
    labels = data.labels
    fit_images = data.fit_images
    fit_targets = data.fit_targets
    fit_outputs = data.fit_outputs
    fit_embeddings = data.fit_embeddings
    selection_images = data.selection_images
    selection_targets = data.selection_targets
    selection_outputs = data.selection_outputs
    selection_embeddings = data.selection_embeddings
    confirmation_images = data.confirmation_images
    confirmation_targets = data.confirmation_targets
    confirmation_outputs = data.confirmation_outputs
    confirmation_embeddings = data.confirmation_embeddings
    validation_images = data.validation_images
    validation_targets = data.validation_targets
    validation_outputs = data.validation_outputs
    validation_embeddings = data.validation_embeddings
    selection_predictions = selection_outputs.argmax(dim=1)
    confirmation_predictions = (
        confirmation_outputs.argmax(dim=1)
        if int(confirmation_targets.numel()) > 0
        else torch.empty((0,), dtype=torch.long)
    )
    base_predictions = validation_outputs.argmax(dim=1)
    probe_predictions = base_predictions.clone()

    reports = []
    skipped = []
    for family in families:
        indices_tuple = family_indices(family, labels)
        if len(indices_tuple) < 2:
            skipped.append(family)
            continue
        train_features = family_features(
            fit_images,
            fit_outputs,
            indices_tuple,
            include_pixel_features=include_pixel_features,
            embedding_outputs=fit_embeddings,
        )
        probe = train_family_probe(
            train_features,
            fit_targets,
            indices_tuple,
            labels,
            epochs,
            learning_rate,
            hidden_units,
            max_train_samples=max_probe_train_samples,
            mini_batch_size=mini_batch_size,
            seed=seed,
        )
        if probe is None:
            skipped.append(family)
            continue
        selection_candidate = apply_family_probe(
            selection_predictions,
            selection_images,
            selection_outputs,
            probe,
            labels,
            source_groups,
            include_pixel_features=include_pixel_features,
            embedding_outputs=selection_embeddings,
            probe_confidence=probe_confidence,
            probe_margin=probe_margin,
        )
        selection_before = _metrics(selection_predictions, selection_targets, labels)
        selection_after = _metrics(selection_candidate, selection_targets, labels)
        selection_passed, selection_reason, selection_delta = _gate_metrics(
            selection_before,
            selection_after,
            min_family_delta,
        )
        confirmation_delta = None
        confirmation_passed = True
        confirmation_reason = None
        if int(confirmation_targets.numel()) > 0:
            confirmation_candidate = apply_family_probe(
                confirmation_predictions,
                confirmation_images,
                confirmation_outputs,
                probe,
                labels,
                source_groups,
                include_pixel_features=include_pixel_features,
                embedding_outputs=confirmation_embeddings,
                probe_confidence=probe_confidence,
                probe_margin=probe_margin,
            )
            confirmation_before = _metrics(confirmation_predictions, confirmation_targets, labels)
            confirmation_after = _metrics(confirmation_candidate, confirmation_targets, labels)
            confirmation_passed, confirmation_reason, confirmation_delta = _gate_metrics(
                confirmation_before,
                confirmation_after,
                min_family_delta,
            )
        if not selection_passed:
            reports.append(
                {
                    "family": probe.name,
                    "accepted": False,
                    "selection_delta": selection_delta,
                    "confirmation_delta": confirmation_delta,
                    "rejection_reason": f"selection_{selection_reason}",
                }
            )
            continue
        if not confirmation_passed:
            reports.append(
                {
                    "family": probe.name,
                    "accepted": False,
                    "selection_delta": selection_delta,
                    "confirmation_delta": confirmation_delta,
                    "rejection_reason": f"confirmation_{confirmation_reason}",
                }
            )
            continue
        before = _metrics(probe_predictions, validation_targets, labels)
        candidate_predictions = apply_family_probe(
            probe_predictions,
            validation_images,
            validation_outputs,
            probe,
            labels,
            source_groups,
            include_pixel_features=include_pixel_features,
            embedding_outputs=validation_embeddings,
            probe_confidence=probe_confidence,
            probe_margin=probe_margin,
        )
        after = _metrics(candidate_predictions, validation_targets, labels)
        test_passed, test_reason, test_delta = _gate_metrics(before, after, min_family_delta)
        if not test_passed:
            reports.append(
                {
                    "family": probe.name,
                    "accepted": False,
                    "selection_delta": selection_delta,
                    "confirmation_delta": confirmation_delta,
                    "before_validation_accuracy": before["validation_accuracy"],
                    "after_validation_accuracy": after["validation_accuracy"],
                    "delta": test_delta,
                    "rejection_reason": f"validation_{test_reason}",
                }
            )
            continue
        reports.append(
            {
                "family": probe.name,
                "accepted": True,
                "selection_delta": selection_delta,
                "confirmation_delta": confirmation_delta,
                "before_validation_accuracy": before["validation_accuracy"],
                "after_validation_accuracy": after["validation_accuracy"],
                "delta": test_delta,
            }
        )
        probe_predictions = candidate_predictions

    base_metrics = _metrics(base_predictions, validation_targets, labels)
    reranked_metrics = _metrics(probe_predictions, validation_targets, labels)
    promotable = (
        reranked_metrics["validation_accuracy"] > base_metrics["validation_accuracy"]
        and all(reranked_metrics[metric] >= base_metrics[metric] for metric in PROTECTED_METRICS)
    )
    return {
        "families": reports,
        "skipped": skipped,
        "base": base_metrics,
        "reranked": reranked_metrics,
        "validation_delta": reranked_metrics["validation_accuracy"] - base_metrics["validation_accuracy"],
        "promotable": promotable,
        "fit_samples": int(fit_targets.numel()),
        "train_only_extra_samples": data.train_only_count,
        "selection_samples": int(selection_targets.numel()),
        "confirmation_samples": int(confirmation_targets.numel()),
        "validation_samples": int(validation_targets.numel()),
        "hidden_units": hidden_units,
        "confirmation_ratio": confirmation_ratio,
        "source_groups": list(source_groups) if source_groups is not None else None,
        "include_pixel_features": include_pixel_features,
        "include_embedding_features": include_embedding_features,
        "max_probe_train_samples": max_probe_train_samples,
        "mini_batch_size": mini_batch_size,
        "probe_thresholds": {
            "confidence": probe_confidence,
            "margin": probe_margin,
        },
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Probe character visual-family rerankers.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--confirmation-ratio", type=float, default=0.5)
    parser.add_argument("--min-family-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--include-pixel-features", action="store_true")
    parser.add_argument("--include-embedding-features", action="store_true")
    parser.add_argument("--probe-confidence", type=float, default=0.0)
    parser.add_argument("--probe-margin", type=float, default=0.0)
    parser.add_argument(
        "--max-probe-train-samples",
        type=int,
        default=None,
        help="Cap each family probe's balanced training samples for faster bounded sweeps.",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=None,
        help="Train each family probe with mini-batches instead of one full batch.",
    )
    parser.add_argument(
        "--train-only-extra-root",
        action="append",
        default=[],
        help="Extra ASCII folder or .pt tensor cache used only for fitting rerankers, never selection/validation.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_probe(
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                families=parse_families(args.families),
                calibration_ratio=args.calibration_ratio,
                confirmation_ratio=args.confirmation_ratio,
                min_family_delta=args.min_family_delta,
                seed=args.seed,
                hidden_units=args.hidden_units,
                source_groups=parse_label_groups(args.source_groups),
                train_only_extra_roots=tuple(Path(root) for root in args.train_only_extra_root),
                include_pixel_features=args.include_pixel_features,
                include_embedding_features=args.include_embedding_features,
                probe_confidence=args.probe_confidence,
                probe_margin=args.probe_margin,
                max_probe_train_samples=args.max_probe_train_samples,
                mini_batch_size=args.mini_batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
