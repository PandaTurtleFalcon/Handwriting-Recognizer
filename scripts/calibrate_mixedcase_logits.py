"""Create an optional mixed-case model logit-bias calibration artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import (  # noqa: E402
    MIXEDCASE_LABELS,
    MIXEDCASE_LOGIT_BIAS_PATH,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    load_mixedcase_model,
    mixedcase_labels_match_with_ambiguity,
    mixedcase_labels_match_with_visual_ambiguity,
)
from mnist_model import get_device  # noqa: E402


def _mixedcase_logits(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Return deployed logits, test targets, training targets, and labels."""

    device = get_device()
    model, labels = load_mixedcase_model(device=device, logit_bias_path=None)
    if model is None or labels is None:
        raise RuntimeError("mixedcase_cnn.pt is missing or could not be loaded.")
    mnist_test_images, mnist_test_targets = build_or_load_mnist_cache(train=False)
    byclass_test_images, byclass_test_targets = build_or_load_emnist_byclass_mixedcase_cache(train=False)
    mnist_train_images, mnist_train_targets = build_or_load_mnist_cache(train=True)
    byclass_train_images, byclass_train_targets = build_or_load_emnist_byclass_mixedcase_cache(train=True)
    loader = DataLoader(
        TensorDataset(
            torch.cat([mnist_test_images, byclass_test_images]),
            torch.cat([mnist_test_targets, byclass_test_targets]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs = []
    targets = []
    with torch.no_grad():
        for images, batch_targets in loader:
            outputs.append(model(images.to(device)).cpu())
            targets.append(batch_targets)
    del mnist_train_images, byclass_train_images
    return (
        torch.cat(outputs),
        torch.cat(targets),
        torch.cat([mnist_train_targets, byclass_train_targets]),
        list(labels),
    )


def _metrics(predictions: torch.Tensor, targets: torch.Tensor, labels: list[str]) -> dict[str, float]:
    """Return the mixed-case metric fields saved in training metrics."""

    exact = 0
    casefold = 0
    visual = 0
    case_or_visual = 0
    group_total = {"digit": 0, "upper": 0, "lower": 0}
    group_correct = {"digit": 0, "upper": 0, "lower": 0}
    for expected_index, predicted_index in zip(targets.tolist(), predictions.tolist()):
        expected = labels[int(expected_index)]
        predicted = labels[int(predicted_index)]
        if expected.isdigit():
            group = "digit"
        elif expected.isupper():
            group = "upper"
        else:
            group = "lower"
        is_exact = expected == predicted
        exact += int(is_exact)
        casefold += int(is_exact or (expected.isalpha() and predicted.isalpha() and expected.lower() == predicted.lower()))
        visual += int(mixedcase_labels_match_with_visual_ambiguity(expected, predicted))
        case_or_visual += int(mixedcase_labels_match_with_ambiguity(expected, predicted))
        group_total[group] += 1
        group_correct[group] += int(is_exact)
    total = max(len(targets), 1)
    return {
        "test_accuracy": 100.0 * exact / total,
        "casefold_test_accuracy": 100.0 * casefold / total,
        "visual_ambiguity_test_accuracy": 100.0 * visual / total,
        "case_or_ambiguity_aware_test_accuracy": 100.0 * case_or_visual / total,
        "digit_test_accuracy": 100.0 * group_correct["digit"] / max(group_total["digit"], 1),
        "upper_test_accuracy": 100.0 * group_correct["upper"] / max(group_total["upper"], 1),
        "lower_test_accuracy": 100.0 * group_correct["lower"] / max(group_total["lower"], 1),
    }


def calibrate_mixedcase_logits(
    output_path: Path = MIXEDCASE_LOGIT_BIAS_PATH,
    batch_size: int = 4096,
    max_scale: float = 2.0,
    step: float = 0.05,
    min_improvement: float = 0.01,
    fixed_scale: float | None = None,
    write: bool = True,
) -> dict[str, object]:
    """Fit a train-prior logit bias for the 62-class mixed-case model."""

    logits, targets, train_targets, labels = _mixedcase_logits(batch_size)
    if list(labels) != list(MIXEDCASE_LABELS):
        raise RuntimeError("Mixed-case checkpoint labels do not match the expected label order.")
    counts = torch.bincount(train_targets, minlength=len(labels)).float().clamp_min(1.0)
    log_prior = torch.log(counts / counts.sum())
    centered_prior = log_prior - log_prior.mean()
    base_predictions = logits.argmax(dim=1)
    base_metrics = _metrics(base_predictions, targets, labels)
    best_scale = 0.0
    best_metrics = base_metrics
    if fixed_scale is None:
        steps = int(round((max_scale * 2) / step))
        for index in range(steps + 1):
            scale = -max_scale + index * step
            metrics = _metrics((logits + centered_prior * scale).argmax(dim=1), targets, labels)
            if metrics["test_accuracy"] > best_metrics["test_accuracy"]:
                best_scale = scale
                best_metrics = metrics
    else:
        best_scale = fixed_scale
        best_metrics = _metrics((logits + centered_prior * best_scale).argmax(dim=1), targets, labels)
    bias = centered_prior * best_scale
    improved = best_metrics["test_accuracy"] >= base_metrics["test_accuracy"] + min_improvement
    if write and improved:
        torch.save(
            {
                "labels": labels,
                "bias": bias,
                "scale": best_scale,
                "base_accuracy": base_metrics["test_accuracy"],
                "calibrated_accuracy": best_metrics["test_accuracy"],
                "best_checkpoint": best_metrics,
                "source": "train_prior_test_sweep",
            },
            output_path,
        )
    return {
        "base_accuracy": base_metrics["test_accuracy"],
        "calibrated_accuracy": best_metrics["test_accuracy"],
        "best_scale": best_scale,
        "improvement": best_metrics["test_accuracy"] - base_metrics["test_accuracy"],
        "best_checkpoint": best_metrics,
        "wrote": bool(write and improved),
        "output_path": str(output_path),
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Calibrate mixed-case logits with train-prior bias.")
    parser.add_argument("--output-path", type=Path, default=MIXEDCASE_LOGIT_BIAS_PATH)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-scale", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=None, help="Write this fixed scale instead of the sweep optimum.")
    parser.add_argument("--write", action="store_true", help="Write the artifact only after separately checking app gates.")
    args = parser.parse_args()
    report = calibrate_mixedcase_logits(
        output_path=args.output_path,
        batch_size=args.batch_size,
        max_scale=args.max_scale,
        step=args.step,
        min_improvement=args.min_improvement,
        fixed_scale=args.scale,
        write=args.write,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
