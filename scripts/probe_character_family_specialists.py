"""Probe character visual-family CNN specialists without deploying them."""

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
    DATASET_ROOT,
    build_or_load_combined_cache,
    labels_match_with_ambiguity,
    load_character_model,
    load_extra_character_tensors,
    stratified_split_indices,
)
from mnist_model import get_device  # noqa: E402
from scripts.analyze_character_confusions import _metric_extra_roots  # noqa: E402


DEFAULT_FAMILIES = ("!/1Iil|", "0Oo", "5Ss")
LABEL_GROUPS = ("digit", "letter", "punctuation")
PROTECTED_METRICS = (
    "ambiguity_aware_validation_accuracy",
    "digit_validation_accuracy",
    "letter_validation_accuracy",
    "punctuation_validation_accuracy",
)


@dataclass(frozen=True)
class Specialist:
    """One trained character-family adviser and its global label indices."""

    family: str
    indices: tuple[int, ...]
    model: nn.Module


class FamilyCNN(nn.Module):
    """Small image specialist for choosing within one ambiguous family."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits over labels in one visual family."""

        return self.classifier(self.features(images))


def parse_families(value: str) -> tuple[str, ...]:
    """Parse comma-separated family label strings."""

    families = tuple(part.strip() for part in value.split(",") if part.strip())
    return families or DEFAULT_FAMILIES


def parse_source_groups(value: str) -> tuple[str, ...]:
    """Parse current prediction groups that specialists may rewrite."""

    groups = tuple(part.strip() for part in value.split(",") if part.strip())
    if not groups:
        return LABEL_GROUPS
    unknown = sorted(set(groups) - set(LABEL_GROUPS))
    if unknown:
        raise ValueError(f"Unknown source group(s): {', '.join(unknown)}")
    return groups


def label_group(label: str) -> str:
    """Return the benchmark group for one character label."""

    if label.isdigit():
        return "digit"
    if label.isalpha():
        return "letter"
    return "punctuation"


def family_indices(family: str, labels: list[str]) -> tuple[int, ...]:
    """Return global label indices for labels in one ordered family."""

    label_to_index = {label: index for index, label in enumerate(labels)}
    return tuple(label_to_index[label] for label in dict.fromkeys(family) if label in label_to_index)


def source_group_mask(predictions: torch.Tensor, labels: list[str], groups: tuple[str, ...]) -> torch.Tensor:
    """Return predictions whose current labels are in selected source groups."""

    if set(groups) == set(LABEL_GROUPS):
        return torch.ones_like(predictions, dtype=torch.bool)
    allowed = set(groups)
    return torch.tensor([label_group(labels[int(index)]) in allowed for index in predictions.tolist()], dtype=torch.bool)


def validation_tensors() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return the same validation split used by character benchmark summaries."""

    images, targets, labels = build_or_load_combined_cache(DATASET_ROOT, _metric_extra_roots())
    _, validation_indices = stratified_split_indices(
        list(range(len(targets))),
        test_size=0.15,
        random_state=42,
        stratify=targets.numpy(),
    )
    index_tensor = torch.tensor(validation_indices, dtype=torch.long)
    return images[index_tensor], targets[index_tensor], list(labels)


def train_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the benchmark-style character training split tensors."""

    images, targets, _labels = build_or_load_combined_cache(DATASET_ROOT, _metric_extra_roots())
    train_indices, _ = stratified_split_indices(
        list(range(len(targets))),
        test_size=0.15,
        random_state=42,
        stratify=targets.numpy(),
    )
    index_tensor = torch.tensor(train_indices, dtype=torch.long)
    train_images = images[index_tensor]
    train_targets = targets[index_tensor]
    return train_images, train_targets


def append_train_only_extras(
    train_images: torch.Tensor,
    train_targets: torch.Tensor,
    labels: list[str],
    extra_roots: list[Path],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append supplemental data to fitting tensors without entering holdouts."""

    extras = load_extra_character_tensors(extra_roots, labels)
    if extras is not None:
        extra_images, extra_targets = extras
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(int(extra_targets.numel()), generator=generator)
        train_images = torch.cat([train_images, extra_images[order]])
        train_targets = torch.cat([train_targets, extra_targets[order]])
    return train_images, train_targets


def split_holdout(
    images: torch.Tensor,
    targets: torch.Tensor,
    ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split tensors into fit and holdout slices."""

    if ratio <= 0.0 or int(targets.numel()) < 2:
        return images, targets, images[:0], targets[:0]
    holdout_count = max(1, int(round(int(targets.numel()) * min(ratio, 0.5))))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(int(targets.numel()), generator=generator)
    holdout = order[:holdout_count]
    fit = order[holdout_count:]
    return images[fit], targets[fit], images[holdout], targets[holdout]


def capped_family_dataset(
    images: torch.Tensor,
    targets: torch.Tensor,
    indices: tuple[int, ...],
    max_per_label: int | None,
    seed: int,
) -> TensorDataset:
    """Return a shuffled local-label dataset for one family."""

    generator = torch.Generator().manual_seed(seed)
    selected_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for local_index, global_index in enumerate(indices):
        matches = torch.where(targets == global_index)[0]
        if max_per_label is not None and int(matches.numel()) > max_per_label:
            matches = matches[torch.randperm(int(matches.numel()), generator=generator)[:max_per_label]]
        selected_parts.append(matches)
        target_parts.append(torch.full((int(matches.numel()),), local_index, dtype=torch.long))
    selected = torch.cat(selected_parts) if selected_parts else torch.empty((0,), dtype=torch.long)
    local_targets = torch.cat(target_parts) if target_parts else torch.empty((0,), dtype=torch.long)
    if not int(selected.numel()):
        return TensorDataset(images[:0], targets[:0])
    order = torch.randperm(int(selected.numel()), generator=generator)
    return TensorDataset(images[selected][order], local_targets[order])


def train_specialist(
    family: str,
    indices: tuple[int, ...],
    images: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_per_label: int | None,
    seed: int,
) -> Specialist | None:
    """Train one small family classifier."""

    dataset = capped_family_dataset(images, targets, indices, max_per_label, seed)
    if len(dataset) < max(20, len(indices) * 5):
        return None
    model = FamilyCNN(classes=len(indices)).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(max(1, epochs)):
        for batch_images, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_images.to(device)), batch_targets.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    return Specialist(family=family, indices=indices, model=model)


def deployed_predictions(images: torch.Tensor, batch_size: int, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    """Return current deployed character predictions."""

    model, labels = load_character_model(device=device)
    if not int(images.shape[0]):
        return torch.empty((0,), dtype=torch.long), labels
    loader = DataLoader(TensorDataset(images), batch_size=batch_size)
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch_images,) in loader:
            parts.append(model(batch_images.to(device)).argmax(dim=1).cpu())
    return torch.cat(parts), labels


def metrics(predictions: torch.Tensor, targets: torch.Tensor, labels: list[str]) -> dict[str, float]:
    """Return exact and protected character metrics."""

    totals = {"digit": 0, "letter": 0, "punctuation": 0}
    correct = {"digit": 0, "letter": 0, "punctuation": 0}
    ambiguity = 0
    for expected_index, predicted_index in zip(targets.tolist(), predictions.tolist()):
        expected = labels[int(expected_index)]
        predicted = labels[int(predicted_index)]
        group = label_group(expected)
        totals[group] += 1
        correct[group] += int(expected == predicted)
        ambiguity += int(labels_match_with_ambiguity(expected, predicted))
    exact = int((predictions == targets).sum().item())
    total = int(targets.numel())
    return {
        "validation_accuracy": 100.0 * exact / max(total, 1),
        "ambiguity_aware_validation_accuracy": 100.0 * ambiguity / max(total, 1),
        "digit_validation_accuracy": 100.0 * correct["digit"] / max(totals["digit"], 1),
        "letter_validation_accuracy": 100.0 * correct["letter"] / max(totals["letter"], 1),
        "punctuation_validation_accuracy": 100.0 * correct["punctuation"] / max(totals["punctuation"], 1),
    }


def apply_specialists(
    base_predictions: torch.Tensor,
    images: torch.Tensor,
    specialists: list[Specialist],
    labels: list[str],
    batch_size: int,
    device: torch.device,
    confidence_threshold: float,
    margin_threshold: float,
    source_groups: tuple[str, ...],
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Apply family predictions only to eligible current predictions."""

    predictions = base_predictions.clone()
    allowed_sources = source_group_mask(predictions, labels, source_groups)
    reports: list[dict[str, object]] = []
    for specialist in specialists:
        family_mask = torch.zeros_like(predictions, dtype=torch.bool)
        for index in specialist.indices:
            family_mask |= predictions == index
        candidate_indices = torch.where(family_mask & allowed_sources)[0]
        if not int(candidate_indices.numel()):
            reports.append({"family": specialist.family, "eligible": 0, "changed": 0})
            continue
        outputs: list[torch.Tensor] = []
        loader = DataLoader(TensorDataset(images[candidate_indices]), batch_size=batch_size)
        with torch.no_grad():
            for (batch_images,) in loader:
                outputs.append(specialist.model(batch_images.to(device)).cpu())
        probabilities = torch.cat(outputs).softmax(dim=1)
        top2 = probabilities.topk(min(2, probabilities.shape[1]), dim=1)
        confidence = top2.values[:, 0]
        margin = (
            top2.values[:, 0] - top2.values[:, 1]
            if top2.values.shape[1] > 1
            else torch.ones_like(top2.values[:, 0])
        )
        override_mask = (confidence >= confidence_threshold) & (margin >= margin_threshold)
        replacements = torch.tensor([specialist.indices[int(index)] for index in top2.indices[:, 0]], dtype=torch.long)
        changed = int((predictions[candidate_indices][override_mask] != replacements[override_mask]).sum().item())
        predictions[candidate_indices[override_mask]] = replacements[override_mask]
        reports.append({"family": specialist.family, "eligible": int(candidate_indices.numel()), "changed": changed})
    return predictions, reports


def threshold_report(base: torch.Tensor, candidate: torch.Tensor, targets: torch.Tensor) -> dict[str, int]:
    """Return how many overrides fixed and broke predictions."""

    changed = candidate != base
    fixed = changed & (candidate == targets) & (base != targets)
    broken = changed & (candidate != targets) & (base == targets)
    return {"changed": int(changed.sum().item()), "fixed": int(fixed.sum().item()), "broken": int(broken.sum().item())}


def protected_ok(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    """Return whether candidate metrics preserve protected splits."""

    return all(candidate[name] >= baseline[name] for name in PROTECTED_METRICS)


def protected_failures(candidate: dict[str, float], baseline: dict[str, float]) -> list[str]:
    """Return protected metric regressions for diagnostics."""

    return [
        f"{name} {candidate[name]:.4f}% < baseline {baseline[name]:.4f}%"
        for name in PROTECTED_METRICS
        if candidate[name] < baseline[name]
    ]


def choose_thresholds(
    base_predictions: torch.Tensor,
    images: torch.Tensor,
    targets: torch.Tensor,
    labels: list[str],
    specialists: list[Specialist],
    batch_size: int,
    device: torch.device,
    confidence_grid: tuple[float, ...],
    margin_grid: tuple[float, ...],
    source_groups: tuple[str, ...],
) -> dict[str, object]:
    """Choose thresholds that improve exact accuracy while preserving splits."""

    base_metrics = metrics(base_predictions, targets, labels)
    best: dict[str, object] = {
        "confidence": None,
        "margin": None,
        "base": base_metrics,
        "candidate": base_metrics,
        "replacement_report": {"changed": 0, "fixed": 0, "broken": 0},
        "best_rejected": None,
    }
    best_gain = 0.0
    best_rejected_gain = float("-inf")
    for confidence in confidence_grid:
        for margin in margin_grid:
            candidate_predictions, _ = apply_specialists(
                base_predictions,
                images,
                specialists,
                labels,
                batch_size,
                device,
                confidence,
                margin,
                source_groups,
            )
            candidate_metrics = metrics(candidate_predictions, targets, labels)
            gain = candidate_metrics["validation_accuracy"] - base_metrics["validation_accuracy"]
            failures = protected_failures(candidate_metrics, base_metrics)
            if gain > best_rejected_gain:
                best_rejected_gain = gain
                best["best_rejected"] = {
                    "confidence": confidence,
                    "margin": margin,
                    "gain": gain,
                    "candidate": candidate_metrics,
                    "replacement_report": threshold_report(base_predictions, candidate_predictions, targets),
                    "protected_failures": failures,
                }
            if gain > best_gain and protected_ok(candidate_metrics, base_metrics):
                best_gain = gain
                best = {
                    "confidence": confidence,
                    "margin": margin,
                    "base": base_metrics,
                    "candidate": candidate_metrics,
                    "replacement_report": threshold_report(base_predictions, candidate_predictions, targets),
                    "best_rejected": best["best_rejected"],
                }
    return best


def probe_family_specialists(
    families: tuple[str, ...],
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_per_label: int | None,
    extra_roots: list[Path],
    confidence_grid: tuple[float, ...],
    margin_grid: tuple[float, ...],
    validation_ratio: float,
    confirmation_ratio: float,
    source_groups: tuple[str, ...],
    seed: int,
) -> dict[str, object]:
    """Train specialists, tune on train holdouts, and evaluate validation."""

    torch.manual_seed(seed)
    device = get_device()
    validation_images, validation_targets, labels = validation_tensors()
    base_validation_predictions, labels = deployed_predictions(validation_images, batch_size, device)
    train_images, train_targets = train_tensors()
    fit_images, fit_targets, holdout_images, holdout_targets = split_holdout(
        train_images,
        train_targets,
        validation_ratio,
        seed + 1,
    )
    selection_images, selection_targets, confirmation_images, confirmation_targets = split_holdout(
        holdout_images,
        holdout_targets,
        confirmation_ratio,
        seed + 2,
    )
    fit_images, fit_targets = append_train_only_extras(fit_images, fit_targets, labels, extra_roots, seed)
    specialists: list[Specialist] = []
    skipped: list[str] = []
    for family in families:
        indices = family_indices(family, labels)
        if len(indices) < 2:
            skipped.append(family)
            continue
        specialist = train_specialist(
            family,
            indices,
            fit_images,
            fit_targets,
            device,
            batch_size,
            epochs,
            learning_rate,
            max_per_label,
            seed + len(specialists),
        )
        if specialist is None:
            skipped.append(family)
        else:
            specialists.append(specialist)
    if int(selection_targets.numel()):
        selection_predictions, _ = deployed_predictions(selection_images, batch_size, device)
        selected = choose_thresholds(
            selection_predictions,
            selection_images,
            selection_targets,
            labels,
            specialists,
            batch_size,
            device,
            confidence_grid,
            margin_grid,
            source_groups,
        )
    else:
        selected = {
            "confidence": confidence_grid[0] if confidence_grid else None,
            "margin": margin_grid[0] if margin_grid else None,
            "base": {},
            "candidate": {},
            "replacement_report": {"changed": 0, "fixed": 0, "broken": 0},
            "best_rejected": None,
        }
    confidence = selected.get("confidence")
    margin = selected.get("margin")
    confirmation: dict[str, object] | None = None
    if confidence is not None and margin is not None and int(confirmation_targets.numel()):
        confirmation_predictions, _ = deployed_predictions(confirmation_images, batch_size, device)
        confirmation_candidate, _ = apply_specialists(
            confirmation_predictions,
            confirmation_images,
            specialists,
            labels,
            batch_size,
            device,
            float(confidence),
            float(margin),
            source_groups,
        )
        before = metrics(confirmation_predictions, confirmation_targets, labels)
        after = metrics(confirmation_candidate, confirmation_targets, labels)
        selection_gain = float(dict(selected["candidate"])["validation_accuracy"]) - float(
            dict(selected["base"])["validation_accuracy"]
        )
        confirmation_gain = after["validation_accuracy"] - before["validation_accuracy"]
        confirmation = {
            "base": before,
            "candidate": after,
            "replacement_report": threshold_report(confirmation_predictions, confirmation_candidate, confirmation_targets),
            "gain": confirmation_gain,
            "confirmed": confirmation_gain >= max(0.0, selection_gain) and protected_ok(after, before),
        }
        if not bool(confirmation["confirmed"]):
            confidence = None
            margin = None
    if confidence is None or margin is None:
        candidate_predictions = base_validation_predictions.clone()
        family_reports = [{"family": specialist.family, "eligible": 0, "changed": 0} for specialist in specialists]
    else:
        candidate_predictions, family_reports = apply_specialists(
            base_validation_predictions,
            validation_images,
            specialists,
            labels,
            batch_size,
            device,
            float(confidence),
            float(margin),
            source_groups,
        )
    base_metrics = metrics(base_validation_predictions, validation_targets, labels)
    candidate_metrics = metrics(candidate_predictions, validation_targets, labels)
    return {
        "families": [specialist.family for specialist in specialists],
        "skipped_families": skipped,
        "thresholds": {"confidence": confidence, "margin": margin},
        "threshold_selection": selected,
        "confirmation": confirmation,
        "base": base_metrics,
        "candidate": candidate_metrics,
        "delta": {key: candidate_metrics[key] - base_metrics[key] for key in sorted(candidate_metrics)},
        "family_reports": family_reports,
        "selection_samples": int(selection_targets.numel()),
        "confirmation_samples": int(confirmation_targets.numel()),
        "validation_samples": int(validation_targets.numel()),
        "source_groups": list(source_groups),
        "extra_roots": [str(path) for path in extra_roots],
        "promotable": candidate_metrics["validation_accuracy"] > base_metrics["validation_accuracy"]
        and protected_ok(candidate_metrics, base_metrics),
    }


def parse_float_grid(value: str) -> tuple[float, ...]:
    """Parse comma-separated floats."""

    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    """Run the character specialist probe."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-per-label", type=int, default=600)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--confidence-grid", default="0.5,0.65,0.8,0.9,0.95")
    parser.add_argument("--margin-grid", default="0.0,0.1,0.2,0.35,0.5")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--confirmation-ratio", type=float, default=0.5)
    parser.add_argument("--source-groups", default="digit,letter,punctuation")
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    report = probe_family_specialists(
        families=parse_families(args.families),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_per_label=args.max_per_label,
        extra_roots=args.extra_root,
        confidence_grid=parse_float_grid(args.confidence_grid),
        margin_grid=parse_float_grid(args.margin_grid),
        validation_ratio=args.validation_ratio,
        confirmation_ratio=args.confirmation_ratio,
        source_groups=parse_source_groups(args.source_groups),
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
