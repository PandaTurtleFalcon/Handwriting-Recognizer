"""Build a train-only character cache from hard visual-family training samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from character_model import DATASET_ROOT, build_or_load_combined_cache, load_character_model, stratified_split_indices  # noqa: E402
from mnist_model import get_device  # noqa: E402
from scripts.analyze_character_confusions import _metric_extra_roots  # noqa: E402


DEFAULT_FAMILIES = ("!/1Iil|", "0Oo")


def parse_families(value: str | None) -> tuple[str, ...]:
    """Parse comma-separated visual-family labels."""

    if value is None:
        return DEFAULT_FAMILIES
    families = tuple(part.strip() for part in value.split(",") if part.strip())
    return families or DEFAULT_FAMILIES


def family_label_indices(families: tuple[str, ...], labels: list[str]) -> tuple[int, ...]:
    """Return unique label indices selected by visual-family strings."""

    label_to_index = {label: index for index, label in enumerate(labels)}
    selected: list[int] = []
    for family in families:
        for label in family:
            index = label_to_index.get(label)
            if index is not None and index not in selected:
                selected.append(index)
    if not selected:
        raise ValueError("No requested family labels exist in the character labels.")
    return tuple(selected)


def training_split_indices(targets: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Return the benchmark training indices, never the validation indices."""

    train_indices, _validation_indices = stratified_split_indices(
        list(range(int(targets.numel()))),
        test_size=0.15,
        random_state=seed,
        stratify=targets.numpy(),
    )
    return torch.tensor(train_indices, dtype=torch.long)


def deployed_stats(images: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Return deployed predictions, confidence, margin, and labels for images."""

    device = get_device()
    model, labels = load_character_model(device=device)
    loader = DataLoader(TensorDataset(images), batch_size=batch_size)
    prediction_parts: list[torch.Tensor] = []
    confidence_parts: list[torch.Tensor] = []
    margin_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for (batch_images,) in loader:
            outputs = model(batch_images.to(device)).cpu()
            probabilities = outputs.softmax(dim=1)
            top2 = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
            prediction_parts.append(outputs.argmax(dim=1))
            confidence_parts.append(top2[:, 0])
            margin_parts.append(top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else torch.ones_like(top2[:, 0]))
    return torch.cat(prediction_parts), torch.cat(confidence_parts), torch.cat(margin_parts), labels


def hard_family_indices(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    margins: torch.Tensor,
    label_indices: tuple[int, ...],
    max_per_label: int,
    max_confidence: float,
    max_margin: float,
    seed: int,
) -> torch.Tensor:
    """Return balanced hard-example indices within selected target families."""

    generator = torch.Generator().manual_seed(seed)
    selected_parts: list[torch.Tensor] = []
    family_targets = torch.zeros_like(targets, dtype=torch.bool)
    for label_index in label_indices:
        family_targets |= targets == label_index
    hard_mask = family_targets & (
        (predictions != targets)
        | (confidences <= max_confidence)
        | (margins <= max_margin)
    )
    for label_index in label_indices:
        matches = torch.where(hard_mask & (targets == label_index))[0]
        if not int(matches.numel()):
            continue
        order = torch.randperm(int(matches.numel()), generator=generator)
        selected_parts.append(matches[order[:max_per_label]])
    if not selected_parts:
        return torch.empty((0,), dtype=torch.long)
    selected = torch.cat(selected_parts)
    return selected[torch.randperm(int(selected.numel()), generator=generator)]


def build_hard_family_pack(
    families: tuple[str, ...],
    max_per_label: int,
    max_confidence: float,
    max_margin: float,
    seed: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build a train-only hard-family tensor pack from the character training split."""

    images, targets, cache_labels = build_or_load_combined_cache(DATASET_ROOT, _metric_extra_roots())
    train_indices = training_split_indices(targets)
    train_images = images.index_select(0, train_indices)
    train_targets = targets.index_select(0, train_indices)
    predictions, confidences, margins, model_labels = deployed_stats(train_images, batch_size)
    if list(cache_labels) != list(model_labels):
        raise RuntimeError("Character cache labels do not match deployed checkpoint labels.")
    selected_label_indices = family_label_indices(families, list(model_labels))
    selected = hard_family_indices(
        train_targets,
        predictions,
        confidences,
        margins,
        selected_label_indices,
        max_per_label=max_per_label,
        max_confidence=max_confidence,
        max_margin=max_margin,
        seed=seed,
    )
    if not int(selected.numel()):
        raise RuntimeError("No hard family samples matched the requested thresholds.")
    selected_images = train_images.index_select(0, selected).float()
    selected_targets = train_targets.index_select(0, selected).long()
    counts = Counter(int(target) for target in selected_targets.tolist())
    wrong_count = int((predictions.index_select(0, selected) != selected_targets).sum().item())
    metadata = {
        "families": list(families),
        "labels": [model_labels[index] for index in selected_label_indices],
        "cache_labels": list(model_labels),
        "counts": {model_labels[index]: counts.get(index, 0) for index in selected_label_indices},
        "selected_samples": int(selected_targets.numel()),
        "wrong_prediction_samples": wrong_count,
        "low_confidence_samples": int((confidences.index_select(0, selected) <= max_confidence).sum().item()),
        "low_margin_samples": int((margins.index_select(0, selected) <= max_margin).sum().item()),
        "max_per_label": max_per_label,
        "max_confidence": max_confidence,
        "max_margin": max_margin,
        "seed": seed,
    }
    return selected_images, selected_targets, metadata


def save_pack(output: Path, images: torch.Tensor, targets: torch.Tensor, metadata: dict[str, object]) -> dict[str, object]:
    """Write the cache and return a JSON-friendly report."""

    output.parent.mkdir(parents=True, exist_ok=True)
    cache_labels = metadata.get("cache_labels")
    payload = {"images": images, "targets": targets, "metadata": metadata}
    if isinstance(cache_labels, list) and all(isinstance(label, str) for label in cache_labels):
        payload["labels"] = cache_labels
    torch.save(payload, output)
    return {"output": str(output), "samples": int(targets.numel()), **metadata}


def main() -> None:
    """Run the hard-family pack builder."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--max-per-label", type=int, default=500)
    parser.add_argument("--max-confidence", type=float, default=0.70)
    parser.add_argument("--max-margin", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "tmp" / "character_hard_family_pack.pt")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    images, targets, metadata = build_hard_family_pack(
        families=parse_families(args.families),
        max_per_label=args.max_per_label,
        max_confidence=args.max_confidence,
        max_margin=args.max_margin,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    report = save_pack(args.output, images, targets, metadata)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"wrote {args.output} with {targets.numel()} samples")


if __name__ == "__main__":
    main()
