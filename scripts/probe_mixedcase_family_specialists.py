"""Probe small visual-family specialists for mixed-case exact recognition."""

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
    MIXEDCASE_AMBIGUITY_GROUPS,
    MIXEDCASE_LABELS,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    limit_mixedcase_extra_cache,
    load_mixedcase_extra_cache,
    load_mixedcase_model,
)
from mnist_model import get_device  # noqa: E402
from scripts.calibrate_mixedcase_logits import _metrics  # noqa: E402


DEFAULT_FAMILIES = ("1Ili", "0Oo", "5Ss", "MNmn", "9qg", "Uuv")
SOURCE_GROUPS = ("digit", "upper", "lower")


@dataclass(frozen=True)
class Specialist:
    """One trained visual-family adviser and its global class indices."""

    family: str
    indices: tuple[int, ...]
    model: nn.Module


class FamilyCNN(nn.Module):
    """Tiny CNN for choosing within one visual-twin family."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-family logits."""

        return self.classifier(self.features(images))


def family_indices(family: str, labels: tuple[str, ...] = MIXEDCASE_LABELS) -> tuple[int, ...]:
    """Return global indices for labels in one family string."""

    label_to_index = {label: index for index, label in enumerate(labels)}
    return tuple(label_to_index[label] for label in dict.fromkeys(family) if label in label_to_index)


def parse_families(value: str) -> tuple[str, ...]:
    """Parse requested families, accepting commas or known group names."""

    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    if requested:
        return requested
    return DEFAULT_FAMILIES


def parse_source_groups(value: str) -> tuple[str, ...]:
    """Parse current prediction groups that specialists may rewrite."""

    groups = tuple(part.strip() for part in value.split(",") if part.strip())
    if not groups:
        return SOURCE_GROUPS
    unknown = sorted(set(groups) - set(SOURCE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown source group(s): {', '.join(unknown)}")
    return groups


def source_group_mask(predictions: torch.Tensor, groups: tuple[str, ...]) -> torch.Tensor:
    """Return a mask for predictions whose current labels are in selected groups."""

    if set(groups) == set(SOURCE_GROUPS):
        return torch.ones_like(predictions, dtype=torch.bool)
    mask = torch.zeros_like(predictions, dtype=torch.bool)
    if "digit" in groups:
        mask |= predictions < 10
    if "upper" in groups:
        mask |= (predictions >= 10) & (predictions < 36)
    if "lower" in groups:
        mask |= predictions >= 36
    return mask


def load_split_tensors(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MNIST plus EMNIST ByClass mixed-case tensors."""

    mnist_images, mnist_targets = build_or_load_mnist_cache(train=train)
    byclass_images, byclass_targets = build_or_load_emnist_byclass_mixedcase_cache(train=train)
    return torch.cat([mnist_images, byclass_images]), torch.cat([mnist_targets, byclass_targets])


def append_extra_tensors(
    images: torch.Tensor,
    targets: torch.Tensor,
    extra_roots: list[Path],
    extra_samples_per_class: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append optional capped extra mixed-case caches to train tensors."""

    image_parts = [images]
    target_parts = [targets]
    for extra_index, extra_root in enumerate(extra_roots):
        extra_images, extra_targets = load_mixedcase_extra_cache(extra_root)
        extra_images, extra_targets = limit_mixedcase_extra_cache(
            extra_images,
            extra_targets,
            extra_samples_per_class,
            seed + 10_000 + extra_index,
        )
        image_parts.append(extra_images)
        target_parts.append(extra_targets)
    return torch.cat(image_parts), torch.cat(target_parts)


def capped_family_dataset(
    images: torch.Tensor,
    targets: torch.Tensor,
    indices: tuple[int, ...],
    max_per_label: int | None,
    seed: int,
) -> TensorDataset:
    """Build a local-label dataset for one visual family."""

    generator = torch.Generator().manual_seed(seed)
    selected_indices: list[torch.Tensor] = []
    local_targets: list[torch.Tensor] = []
    for local_index, global_index in enumerate(indices):
        matches = torch.where(targets == global_index)[0]
        if max_per_label is not None and int(matches.numel()) > max_per_label:
            order = torch.randperm(int(matches.numel()), generator=generator)[:max_per_label]
            matches = matches[order]
        selected_indices.append(matches)
        local_targets.append(torch.full((int(matches.numel()),), local_index, dtype=torch.long))
    if not selected_indices:
        return TensorDataset(images[:0], targets[:0])
    selected = torch.cat(selected_indices)
    local = torch.cat(local_targets)
    order = torch.randperm(int(selected.numel()), generator=generator)
    return TensorDataset(images[selected][order], local[order])


def split_holdout(
    images: torch.Tensor,
    targets: torch.Tensor,
    ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split train tensors into fit and validation slices."""

    if ratio <= 0.0 or int(targets.numel()) < 2:
        return images, targets, images[:0], targets[:0]
    validation_count = max(1, int(round(int(targets.numel()) * min(ratio, 0.5))))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(int(targets.numel()), generator=generator)
    validation = order[:validation_count]
    fit = order[validation_count:]
    return images[fit], targets[fit], images[validation], targets[validation]


def train_specialist(
    family: str,
    indices: tuple[int, ...],
    train_images: torch.Tensor,
    train_targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_per_label: int | None,
    seed: int,
) -> Specialist | None:
    """Train one visual-family CNN adviser."""

    dataset = capped_family_dataset(train_images, train_targets, indices, max_per_label, seed)
    if len(dataset) < max(20, len(indices) * 5):
        return None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = FamilyCNN(classes=len(indices)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(max(1, epochs)):
        for batch_images, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images.to(device))
            loss = criterion(logits, batch_targets.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    return Specialist(family=family, indices=indices, model=model)


def deployed_predictions_and_labels(
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """Return current deployed mixed-case predictions and labels for images."""

    model, labels = load_mixedcase_model(device=device)
    if model is None or list(labels or []) != list(MIXEDCASE_LABELS):
        raise RuntimeError("A deployed mixed-case model with expected labels is required.")
    loader = DataLoader(TensorDataset(images), batch_size=batch_size)
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch_images,) in loader:
            predictions.append(model(batch_images.to(device)).argmax(dim=1).cpu())
    return torch.cat(predictions), list(labels)


def deployed_predictions(images: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Return current deployed mixed-case predictions for images."""

    predictions, _labels = deployed_predictions_and_labels(images, batch_size, device)
    return predictions


def apply_specialists(
    base_predictions: torch.Tensor,
    images: torch.Tensor,
    specialists: list[Specialist],
    batch_size: int,
    device: torch.device,
    confidence_threshold: float = 0.0,
    margin_threshold: float = 0.0,
    source_groups: tuple[str, ...] = SOURCE_GROUPS,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Replace predictions only when the deployed label is in a trained family."""

    predictions = base_predictions.clone()
    allowed_sources = source_group_mask(predictions, source_groups)
    reports: list[dict[str, object]] = []
    for specialist in specialists:
        family_mask = torch.zeros_like(predictions, dtype=torch.bool)
        for index in specialist.indices:
            family_mask |= predictions == index
        family_mask &= allowed_sources
        candidate_indices = torch.where(family_mask)[0]
        if not int(candidate_indices.numel()):
            reports.append({"family": specialist.family, "replaced": 0})
            continue
        outputs: list[torch.Tensor] = []
        loader = DataLoader(TensorDataset(images[candidate_indices]), batch_size=batch_size)
        with torch.no_grad():
            for (batch_images,) in loader:
                outputs.append(specialist.model(batch_images.to(device)).cpu())
        logits = torch.cat(outputs)
        probabilities = logits.softmax(dim=1)
        top2 = probabilities.topk(min(2, probabilities.shape[1]), dim=1)
        local = top2.indices[:, 0]
        confidence = top2.values[:, 0]
        margin = (
            top2.values[:, 0] - top2.values[:, 1]
            if top2.values.shape[1] > 1
            else torch.ones_like(top2.values[:, 0])
        )
        override_mask = (confidence >= confidence_threshold) & (margin >= margin_threshold)
        replacement = torch.tensor([specialist.indices[int(index)] for index in local.tolist()], dtype=torch.long)
        changed = int((predictions[candidate_indices][override_mask] != replacement[override_mask]).sum().item())
        predictions[candidate_indices[override_mask]] = replacement[override_mask]
        reports.append({"family": specialist.family, "eligible": int(candidate_indices.numel()), "changed": changed})
    return predictions, reports


def threshold_report(
    base_predictions: torch.Tensor,
    candidate_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[dict[str, float], dict[str, object]]:
    """Return metrics and replacement counts for one threshold candidate."""

    metrics = _metrics(candidate_predictions, targets, list(MIXEDCASE_LABELS))
    changed_mask = candidate_predictions != base_predictions
    fixed_mask = changed_mask & (candidate_predictions == targets) & (base_predictions != targets)
    broken_mask = changed_mask & (candidate_predictions != targets) & (base_predictions == targets)
    return metrics, {
        "changed": int(changed_mask.sum().item()),
        "fixed": int(fixed_mask.sum().item()),
        "broken": int(broken_mask.sum().item()),
    }


def choose_thresholds(
    validation_predictions: torch.Tensor,
    validation_images: torch.Tensor,
    validation_targets: torch.Tensor,
    specialists: list[Specialist],
    batch_size: int,
    device: torch.device,
    confidence_values: tuple[float, ...],
    margin_values: tuple[float, ...],
    source_groups: tuple[str, ...] = SOURCE_GROUPS,
) -> dict[str, object]:
    """Select confidence/margin gates on held-out train data."""

    if not specialists or not int(validation_targets.numel()):
        return {
            "confidence": None,
            "margin": None,
            "base": _metrics(validation_predictions, validation_targets, list(MIXEDCASE_LABELS)),
            "candidate": None,
            "replacement_report": {"changed": 0, "fixed": 0, "broken": 0},
        }
    base_metrics = _metrics(validation_predictions, validation_targets, list(MIXEDCASE_LABELS))
    best: dict[str, object] = {
        "confidence": None,
        "margin": None,
        "base": base_metrics,
        "candidate": base_metrics,
        "replacement_report": {"changed": 0, "fixed": 0, "broken": 0},
    }
    best_score = 0.0
    for confidence in confidence_values:
        for margin in margin_values:
            candidate_predictions, _family_reports = apply_specialists(
                validation_predictions,
                validation_images,
                specialists,
                batch_size,
                device,
                confidence_threshold=confidence,
                margin_threshold=margin,
                source_groups=source_groups,
            )
            candidate_metrics, replacements = threshold_report(
                validation_predictions,
                candidate_predictions,
                validation_targets,
            )
            if (
                candidate_metrics["case_or_ambiguity_aware_test_accuracy"]
                < base_metrics["case_or_ambiguity_aware_test_accuracy"]
                or candidate_metrics["digit_test_accuracy"] < base_metrics["digit_test_accuracy"]
                or candidate_metrics["upper_test_accuracy"] < base_metrics["upper_test_accuracy"]
                or candidate_metrics["lower_test_accuracy"] < base_metrics["lower_test_accuracy"]
            ):
                continue
            score = candidate_metrics["test_accuracy"] - base_metrics["test_accuracy"]
            if score > best_score:
                best_score = score
                best = {
                    "confidence": confidence,
                    "margin": margin,
                    "base": base_metrics,
                    "candidate": candidate_metrics,
                    "replacement_report": replacements,
                }
    return best


def threshold_is_confirmed(
    validation_predictions: torch.Tensor,
    validation_images: torch.Tensor,
    validation_targets: torch.Tensor,
    specialists: list[Specialist],
    batch_size: int,
    device: torch.device,
    confidence: float,
    margin: float,
    min_gain: float = 0.0,
    source_groups: tuple[str, ...] = SOURCE_GROUPS,
) -> tuple[bool, dict[str, object]]:
    """Return whether a selected threshold also improves a second holdout."""

    if not specialists or not int(validation_targets.numel()):
        return False, {
            "base": _metrics(validation_predictions, validation_targets, list(MIXEDCASE_LABELS)),
            "candidate": None,
            "replacement_report": {"changed": 0, "fixed": 0, "broken": 0},
            "gain": 0.0,
        }
    base_metrics = _metrics(validation_predictions, validation_targets, list(MIXEDCASE_LABELS))
    candidate_predictions, _family_reports = apply_specialists(
        validation_predictions,
        validation_images,
        specialists,
        batch_size,
        device,
        confidence_threshold=confidence,
        margin_threshold=margin,
        source_groups=source_groups,
    )
    candidate_metrics, replacements = threshold_report(
        validation_predictions,
        candidate_predictions,
        validation_targets,
    )
    gain = candidate_metrics["test_accuracy"] - base_metrics["test_accuracy"]
    protected_ok = (
        candidate_metrics["case_or_ambiguity_aware_test_accuracy"]
        >= base_metrics["case_or_ambiguity_aware_test_accuracy"]
        and candidate_metrics["digit_test_accuracy"] >= base_metrics["digit_test_accuracy"]
        and candidate_metrics["upper_test_accuracy"] >= base_metrics["upper_test_accuracy"]
        and candidate_metrics["lower_test_accuracy"] >= base_metrics["lower_test_accuracy"]
    )
    return gain >= min_gain and protected_ok, {
        "base": base_metrics,
        "candidate": candidate_metrics,
        "replacement_report": replacements,
        "gain": gain,
    }


def parse_float_grid(value: str) -> tuple[float, ...]:
    """Parse a comma-separated threshold grid."""

    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def probe_family_specialists(
    families: tuple[str, ...],
    batch_size: int,
    epochs: int,
    learning_rate: float,
    max_per_label: int | None,
    train_sample_limit: int | None,
    test_sample_limit: int | None,
    extra_roots: list[Path],
    extra_samples_per_class: int | None,
    specialist_confidence: float,
    specialist_margin: float,
    auto_threshold: bool,
    validation_ratio: float,
    confirmation_ratio: float,
    confidence_grid: tuple[float, ...],
    margin_grid: tuple[float, ...],
    source_groups: tuple[str, ...],
    seed: int,
) -> dict[str, object]:
    """Train and evaluate visual-family specialists without deploying them."""

    torch.manual_seed(seed)
    device = get_device()
    train_images, train_targets = load_split_tensors(train=True)
    test_images, test_targets = load_split_tensors(train=False)
    if train_sample_limit is not None and train_sample_limit < int(train_targets.numel()):
        generator = torch.Generator().manual_seed(seed + 1)
        selected = torch.randperm(int(train_targets.numel()), generator=generator)[:train_sample_limit]
        train_images, train_targets = train_images[selected], train_targets[selected]
    if test_sample_limit is not None and test_sample_limit < int(test_targets.numel()):
        generator = torch.Generator().manual_seed(seed + 2)
        selected = torch.randperm(int(test_targets.numel()), generator=generator)[:test_sample_limit]
        test_images, test_targets = test_images[selected], test_targets[selected]
    fit_images, fit_targets, validation_images, validation_targets = split_holdout(
        train_images,
        train_targets,
        validation_ratio if auto_threshold else 0.0,
        seed + 3,
    )
    selection_images, selection_targets, confirmation_images, confirmation_targets = split_holdout(
        validation_images,
        validation_targets,
        confirmation_ratio if auto_threshold else 0.0,
        seed + 4,
    )
    train_images, train_targets = fit_images, fit_targets
    train_images, train_targets = append_extra_tensors(
        train_images,
        train_targets,
        extra_roots,
        extra_samples_per_class,
        seed,
    )
    trained: list[Specialist] = []
    skipped: list[str] = []
    for family in families:
        indices = family_indices(family)
        if len(indices) < 2:
            skipped.append(family)
            continue
        specialist = train_specialist(
            family,
            indices,
            train_images,
            train_targets,
            device,
            batch_size,
            epochs,
            learning_rate,
            max_per_label,
            seed + len(trained),
        )
        if specialist is None:
            skipped.append(family)
        else:
            trained.append(specialist)
    base_predictions = deployed_predictions(test_images, batch_size, device)
    threshold_selection: dict[str, object] | None = None
    reported_confidence: float | None = specialist_confidence
    reported_margin: float | None = specialist_margin
    if auto_threshold and int(validation_targets.numel()):
        validation_predictions = deployed_predictions(selection_images, batch_size, device)
        threshold_selection = choose_thresholds(
            validation_predictions,
            selection_images,
            selection_targets,
            trained,
            batch_size,
            device,
            confidence_grid,
            margin_grid,
            source_groups,
        )
        if threshold_selection.get("confidence") is not None and threshold_selection.get("margin") is not None:
            selected_confidence = float(threshold_selection["confidence"])
            selected_margin = float(threshold_selection["margin"])
            confirmation_predictions = deployed_predictions(confirmation_images, batch_size, device)
            selected_candidate = threshold_selection.get("candidate", {})
            selected_base = threshold_selection.get("base", {})
            min_gain = max(
                0.0,
                float(selected_candidate.get("test_accuracy", 0.0))
                - float(selected_base.get("test_accuracy", 0.0)),
            )
            confirmed, confirmation_report = threshold_is_confirmed(
                confirmation_predictions,
                confirmation_images,
                confirmation_targets,
                trained,
                batch_size,
                device,
                selected_confidence,
                selected_margin,
                min_gain=min_gain,
                source_groups=source_groups,
            )
            threshold_selection["confirmation"] = confirmation_report
            threshold_selection["confirmed"] = confirmed
            if confirmed:
                specialist_confidence = selected_confidence
                specialist_margin = selected_margin
                reported_confidence = specialist_confidence
                reported_margin = specialist_margin
            else:
                specialist_confidence = float("inf")
                specialist_margin = float("inf")
                reported_confidence = None
                reported_margin = None
        else:
            specialist_confidence = float("inf")
            specialist_margin = float("inf")
            reported_confidence = None
            reported_margin = None
    candidate_predictions, family_reports = apply_specialists(
        base_predictions,
        test_images,
        trained,
        batch_size,
        device,
        confidence_threshold=specialist_confidence,
        margin_threshold=specialist_margin,
        source_groups=source_groups,
    )
    base_metrics = _metrics(base_predictions, test_targets, list(MIXEDCASE_LABELS))
    candidate_metrics = _metrics(candidate_predictions, test_targets, list(MIXEDCASE_LABELS))
    return {
        "families": [specialist.family for specialist in trained],
        "skipped_families": skipped,
        "thresholds": {"confidence": reported_confidence, "margin": reported_margin},
        "threshold_selection": threshold_selection,
        "base": base_metrics,
        "candidate": candidate_metrics,
        "delta": {
            key: float(candidate_metrics.get(key, 0.0)) - float(base_metrics.get(key, 0.0))
            for key in sorted(candidate_metrics)
        },
        "family_reports": family_reports,
        "source_groups": list(source_groups),
        "selection_samples": int(selection_targets.numel()),
        "confirmation_samples": int(confirmation_targets.numel()),
        "promotable": (
            candidate_metrics["test_accuracy"] > base_metrics["test_accuracy"]
            and candidate_metrics["case_or_ambiguity_aware_test_accuracy"]
            >= base_metrics["case_or_ambiguity_aware_test_accuracy"]
            and candidate_metrics["digit_test_accuracy"] >= base_metrics["digit_test_accuracy"]
            and candidate_metrics["upper_test_accuracy"] >= base_metrics["upper_test_accuracy"]
            and candidate_metrics["lower_test_accuracy"] >= base_metrics["lower_test_accuracy"]
        ),
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Probe mixed-case visual-family CNN specialists.")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-per-label", type=int, default=2500)
    parser.add_argument("--train-sample-limit", type=int, default=None)
    parser.add_argument("--test-sample-limit", type=int, default=None)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--extra-samples-per-class", type=int, default=None)
    parser.add_argument("--specialist-confidence", type=float, default=0.85)
    parser.add_argument("--specialist-margin", type=float, default=0.35)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--confirmation-ratio", type=float, default=0.5)
    parser.add_argument("--confidence-grid", default="0.5,0.6,0.7,0.8,0.85,0.9,0.95")
    parser.add_argument("--margin-grid", default="0,0.1,0.2,0.35,0.5,0.7")
    parser.add_argument(
        "--source-groups",
        default="digit,upper,lower",
        help="Comma-separated current prediction groups eligible for specialist rewrites.",
    )
    parser.add_argument("--seed", type=int, default=5150)
    args = parser.parse_args()
    report = probe_family_specialists(
        families=parse_families(args.families),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_per_label=args.max_per_label,
        train_sample_limit=args.train_sample_limit,
        test_sample_limit=args.test_sample_limit,
        extra_roots=args.extra_root,
        extra_samples_per_class=args.extra_samples_per_class,
        specialist_confidence=args.specialist_confidence,
        specialist_margin=args.specialist_margin,
        auto_threshold=args.auto_threshold,
        validation_ratio=args.validation_ratio,
        confirmation_ratio=args.confirmation_ratio,
        confidence_grid=parse_float_grid(args.confidence_grid),
        margin_grid=parse_float_grid(args.margin_grid),
        source_groups=parse_source_groups(args.source_groups),
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
