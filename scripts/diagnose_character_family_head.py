"""Diagnose whether a character-family head has usable signal before deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.probe_character_family_reranker import (  # noqa: E402
    _metrics,
    _prediction_change_summary,
    apply_family_probe,
    family_features,
    family_indices,
    parse_label_groups,
    prepare_probe_data,
    train_family_probe,
)


def family_mask(targets: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    """Return rows whose true target is inside the visual family."""

    mask = torch.zeros_like(targets, dtype=torch.bool)
    for index in indices:
        mask |= targets == index
    return mask


def family_head_accuracy(
    probe_model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    indices: tuple[int, ...],
) -> dict[str, object]:
    """Return standalone family-head accuracy on true-family samples."""

    mask = family_mask(targets, indices)
    sample_count = int(mask.sum().item())
    if not sample_count:
        return {"samples": 0, "accuracy": None}
    target_to_local = {target: local for local, target in enumerate(indices)}
    local_targets = torch.tensor([target_to_local[int(target)] for target in targets[mask].tolist()], dtype=torch.long)
    with torch.no_grad():
        local_predictions = probe_model(features[mask]).argmax(dim=1).cpu()
    correct = int((local_predictions == local_targets).sum().item())
    return {"samples": sample_count, "accuracy": 100.0 * correct / sample_count, "correct": correct}


def split_head_report(
    split_name: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    labels: list[str],
    features: torch.Tensor,
    probe_model: torch.nn.Module,
    indices: tuple[int, ...],
) -> dict[str, object]:
    """Return base and standalone-head accuracy for one split."""

    mask = family_mask(targets, indices)
    sample_count = int(mask.sum().item())
    base_correct = int(((predictions == targets) & mask).sum().item())
    head = family_head_accuracy(probe_model, features, targets, indices)
    return {
        "split": split_name,
        "family_samples": sample_count,
        "base_family_accuracy": 100.0 * base_correct / max(sample_count, 1),
        "head_family_accuracy": head["accuracy"],
        "head_correct": head.get("correct"),
        "base_metrics": _metrics(predictions, targets, labels),
    }


def applied_report(
    split_name: str,
    predictions: torch.Tensor,
    images: torch.Tensor,
    outputs: torch.Tensor,
    embeddings: torch.Tensor | None,
    targets: torch.Tensor,
    labels: list[str],
    probe,
    source_groups: tuple[str, ...] | None,
    include_pixel_features: bool,
    probe_confidence: float,
    probe_margin: float,
) -> dict[str, object]:
    """Return the metric impact from actually applying the family probe."""

    candidate = apply_family_probe(
        predictions,
        images,
        outputs,
        probe,
        labels,
        source_groups=source_groups,
        include_pixel_features=include_pixel_features,
        embedding_outputs=embeddings,
        probe_confidence=probe_confidence,
        probe_margin=probe_margin,
    )
    before = _metrics(predictions, targets, labels)
    after = _metrics(candidate, targets, labels)
    return {
        "split": split_name,
        "changes": _prediction_change_summary(predictions, candidate, targets),
        "metric_deltas": {name: after[name] - before[name] for name in sorted(before)},
        "before": before,
        "after": after,
    }


def run_diagnostic(
    family: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    hidden_units: int,
    seed: int,
    train_only_extra_roots: tuple[Path, ...],
    include_pixel_features: bool,
    include_embedding_features: bool,
    max_probe_train_samples: int | None,
    mini_batch_size: int | None,
    source_groups: tuple[str, ...] | None,
    probe_confidence: float,
    probe_margin: float,
) -> dict[str, object]:
    """Train one family head and report standalone plus applied split behavior."""

    data = prepare_probe_data(
        batch_size=batch_size,
        calibration_ratio=0.25,
        confirmation_ratio=0.5,
        seed=seed,
        train_only_extra_roots=train_only_extra_roots,
        include_embedding_features=include_embedding_features,
    )
    indices = family_indices(family, data.labels)
    if len(indices) < 2:
        raise ValueError(f"Family {family!r} has fewer than two known labels.")
    train_features = family_features(
        data.fit_images,
        data.fit_outputs,
        indices,
        include_pixel_features=include_pixel_features,
        embedding_outputs=data.fit_embeddings,
    )
    probe = train_family_probe(
        train_features,
        data.fit_targets,
        indices,
        data.labels,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        max_train_samples=max_probe_train_samples,
        mini_batch_size=mini_batch_size,
        seed=seed,
    )
    if probe is None:
        return {"family": family, "trained": False, "reason": "not_enough_family_samples"}

    split_inputs = (
        (
            "selection",
            data.selection_images,
            data.selection_targets,
            data.selection_outputs,
            data.selection_embeddings,
        ),
        (
            "confirmation",
            data.confirmation_images,
            data.confirmation_targets,
            data.confirmation_outputs,
            data.confirmation_embeddings,
        ),
        (
            "validation",
            data.validation_images,
            data.validation_targets,
            data.validation_outputs,
            data.validation_embeddings,
        ),
    )
    split_reports = []
    application_reports = []
    for split_name, images, targets, outputs, embeddings in split_inputs:
        predictions = outputs.argmax(dim=1)
        features = family_features(
            images,
            outputs,
            indices,
            include_pixel_features=include_pixel_features,
            embedding_outputs=embeddings,
        )
        split_reports.append(
            split_head_report(split_name, predictions, targets, data.labels, features, probe.model, indices)
        )
        application_reports.append(
            applied_report(
                split_name,
                predictions,
                images,
                outputs,
                embeddings,
                targets,
                data.labels,
                probe,
                source_groups,
                include_pixel_features,
                probe_confidence,
                probe_margin,
            )
        )
    return {
        "family": probe.name,
        "trained": True,
        "indices": list(indices),
        "labels": [data.labels[index] for index in indices],
        "fit_samples": int(data.fit_targets.numel()),
        "train_only_extra_samples": data.train_only_count,
        "parameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "hidden_units": hidden_units,
            "include_pixel_features": include_pixel_features,
            "include_embedding_features": include_embedding_features,
            "max_probe_train_samples": max_probe_train_samples,
            "mini_batch_size": mini_batch_size,
            "source_groups": list(source_groups) if source_groups is not None else None,
            "probe_confidence": probe_confidence,
            "probe_margin": probe_margin,
        },
        "split_reports": split_reports,
        "application_reports": application_reports,
    }


def main() -> None:
    """Run the diagnostic CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="!/1Iil|")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--train-only-extra-root", action="append", default=[])
    parser.add_argument("--include-pixel-features", action="store_true")
    parser.add_argument("--include-embedding-features", action="store_true")
    parser.add_argument("--max-probe-train-samples", type=int, default=None)
    parser.add_argument("--mini-batch-size", type=int, default=None)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--probe-confidence", type=float, default=0.0)
    parser.add_argument("--probe-margin", type=float, default=0.0)
    args = parser.parse_args()

    print(
        json.dumps(
            run_diagnostic(
                family=args.family,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                hidden_units=args.hidden_units,
                seed=args.seed,
                train_only_extra_roots=tuple(Path(root) for root in args.train_only_extra_root),
                include_pixel_features=args.include_pixel_features,
                include_embedding_features=args.include_embedding_features,
                max_probe_train_samples=args.max_probe_train_samples,
                mini_batch_size=args.mini_batch_size,
                source_groups=parse_label_groups(args.source_groups),
                probe_confidence=args.probe_confidence,
                probe_margin=args.probe_margin,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
