"""Create an optional character-model logit-bias calibration artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from character_model import (  # noqa: E402
    DATASET_ROOT,
    LOGIT_BIAS_PATH,
    build_or_load_combined_cache,
    labels_match_with_ambiguity,
    load_character_model,
)
from mnist_model import get_device  # noqa: E402
from scripts.analyze_character_confusions import _metric_extra_roots  # noqa: E402


def _validation_logits(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Return logits, validation targets, training targets, and labels."""

    device = get_device()
    model, labels = load_character_model(device=device, logit_bias_path=None)
    images, targets, cache_labels = build_or_load_combined_cache(DATASET_ROOT, _metric_extra_roots())
    if list(cache_labels) != list(labels):
        raise RuntimeError("Character cache labels do not match deployed checkpoint labels.")
    indices = list(range(len(targets)))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.15,
        random_state=42,
        stratify=targets.numpy(),
    )
    validation_index_tensor = torch.tensor(validation_indices, dtype=torch.long)
    loader = DataLoader(TensorDataset(images[validation_index_tensor], targets[validation_index_tensor]), batch_size=batch_size)
    outputs = []
    validation_targets = []
    with torch.no_grad():
        for batch_images, batch_targets in loader:
            outputs.append(model(batch_images.to(device)).cpu())
            validation_targets.append(batch_targets)
    train_target_tensor = targets[torch.tensor(train_indices, dtype=torch.long)]
    return torch.cat(outputs), torch.cat(validation_targets), train_target_tensor, list(labels)


def _breakdown(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    labels: list[str],
) -> dict[str, float]:
    """Return calibrated validation metrics in character-training format."""

    group_total = {"digits": 0, "letters": 0, "punctuation": 0}
    group_correct = {"digits": 0, "letters": 0, "punctuation": 0}
    group_ambiguity = {"digits": 0, "letters": 0, "punctuation": 0}
    exact = 0
    ambiguity = 0
    total = 0
    for expected_index, predicted_index in zip(targets.tolist(), predictions.tolist()):
        expected = labels[int(expected_index)]
        predicted = labels[int(predicted_index)]
        if expected.isdigit():
            group = "digits"
        elif expected.isalpha():
            group = "letters"
        else:
            group = "punctuation"
        exact_match = expected == predicted
        ambiguity_match = labels_match_with_ambiguity(expected, predicted)
        exact += int(exact_match)
        ambiguity += int(ambiguity_match)
        total += 1
        group_total[group] += 1
        group_correct[group] += int(exact_match)
        group_ambiguity[group] += int(ambiguity_match)
    return {
        "validation_accuracy": 100.0 * exact / max(total, 1),
        "ambiguity_aware_validation_accuracy": 100.0 * ambiguity / max(total, 1),
        "digit_validation_accuracy": 100.0 * group_correct["digits"] / max(group_total["digits"], 1),
        "letter_validation_accuracy": 100.0 * group_correct["letters"] / max(group_total["letters"], 1),
        "punctuation_validation_accuracy": 100.0 * group_correct["punctuation"] / max(group_total["punctuation"], 1),
        "punctuation_ambiguity_aware_validation_accuracy": 100.0
        * group_ambiguity["punctuation"]
        / max(group_total["punctuation"], 1),
    }


def calibrate_character_logits(
    output_path: Path = LOGIT_BIAS_PATH,
    batch_size: int = 2048,
    max_scale: float = 1.5,
    step: float = 0.05,
    min_improvement: float = 0.01,
    fixed_scale: float | None = None,
    write: bool = True,
) -> dict[str, object]:
    """Fit a simple train-prior logit bias and optionally save it."""

    logits, targets, train_targets, labels = _validation_logits(batch_size)
    counts = torch.bincount(train_targets, minlength=len(labels)).float().clamp_min(1.0)
    log_prior = torch.log(counts / counts.sum())
    centered_prior = log_prior - log_prior.mean()
    base_predictions = logits.argmax(dim=1)
    base_accuracy = 100.0 * (base_predictions == targets).float().mean().item()
    best_accuracy = base_accuracy
    best_scale = 0.0
    if fixed_scale is None:
        steps = int(round((max_scale * 2) / step))
        for index in range(steps + 1):
            scale = -max_scale + index * step
            predictions = (logits + centered_prior * scale).argmax(dim=1)
            accuracy = 100.0 * (predictions == targets).float().mean().item()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_scale = scale
    else:
        best_scale = fixed_scale
        best_accuracy = 100.0 * ((logits + centered_prior * best_scale).argmax(dim=1) == targets).float().mean().item()
    bias = centered_prior * best_scale
    improved = best_accuracy >= base_accuracy + min_improvement
    calibrated_predictions = (logits + bias).argmax(dim=1)
    calibrated_breakdown = _breakdown(calibrated_predictions, targets, labels)
    if write and improved:
        torch.save(
            {
                "labels": labels,
                "bias": bias,
                "scale": best_scale,
                "base_accuracy": base_accuracy,
                "calibrated_accuracy": best_accuracy,
                "best_checkpoint": calibrated_breakdown,
                "source": "train_prior_validation_sweep",
            },
            output_path,
        )
    return {
        "base_accuracy": base_accuracy,
        "calibrated_accuracy": best_accuracy,
        "best_scale": best_scale,
        "improvement": best_accuracy - base_accuracy,
        "best_checkpoint": calibrated_breakdown,
        "wrote": bool(write and improved),
        "output_path": str(output_path),
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Calibrate character-model logits with train-prior bias.")
    parser.add_argument("--output-path", type=Path, default=LOGIT_BIAS_PATH)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-scale", type=float, default=1.5)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=None, help="Write this fixed scale instead of the validation optimum.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = calibrate_character_logits(
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_scale=args.max_scale,
        step=args.step,
        min_improvement=args.min_improvement,
        fixed_scale=args.scale,
        write=not args.dry_run,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
