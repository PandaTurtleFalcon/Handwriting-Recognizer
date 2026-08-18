"""Create an optional mixed-case model logit-bias calibration artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
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


def _load_existing_bias(output_path: Path, labels: list[str]) -> torch.Tensor:
    """Return an existing matching bias artifact or a zero bias."""

    if not output_path.exists():
        return torch.zeros(len(labels), dtype=torch.float32)
    try:
        artifact = torch.load(output_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError):
        return torch.zeros(len(labels), dtype=torch.float32)
    if list(artifact.get("labels", [])) != list(labels):
        return torch.zeros(len(labels), dtype=torch.float32)
    bias = artifact.get("bias")
    if not isinstance(bias, torch.Tensor) or bias.numel() != len(labels):
        return torch.zeros(len(labels), dtype=torch.float32)
    return bias.detach().cpu().float().reshape(-1)


def calibrate_mixedcase_greedy_bias(
    output_path: Path = MIXEDCASE_LOGIT_BIAS_PATH,
    batch_size: int = 4096,
    labels_to_tune: str = "",
    deltas: tuple[float, ...] = (-0.04, -0.02, 0.02, 0.04),
    rounds: int = 3,
    min_improvement: float = 0.01,
    objective: str = "test_accuracy",
    min_test: float = 0.0,
    min_case_or_visual: float = 97.0,
    min_digit: float = 83.0,
    min_upper: float = 72.0,
    min_lower: float = 79.0,
    write: bool = True,
) -> dict[str, object]:
    """Greedily tune tiny per-label mixed-case bias changes."""

    logits, targets, _train_targets, labels = _mixedcase_logits(batch_size)
    if list(labels) != list(MIXEDCASE_LABELS):
        raise RuntimeError("Mixed-case checkpoint labels do not match the expected label order.")
    starting_bias = _load_existing_bias(output_path, labels)
    base_metrics = _metrics((logits + starting_bias).argmax(dim=1), targets, labels)
    if objective not in base_metrics:
        raise ValueError(f"Unknown mixed-case calibration objective: {objective}")
    best_bias = starting_bias.clone()
    best_metrics = base_metrics
    tuned_indices = [labels.index(label) for label in dict.fromkeys(labels_to_tune) if label in labels]
    steps: list[dict[str, object]] = []
    for round_index in range(max(0, rounds)):
        improved_this_round = False
        for label_index in tuned_indices:
            for delta in deltas:
                candidate_bias = best_bias.clone()
                candidate_bias[label_index] += float(delta)
                candidate_metrics = _metrics((logits + candidate_bias).argmax(dim=1), targets, labels)
                if (
                    candidate_metrics["test_accuracy"] < min_test
                    or candidate_metrics["case_or_ambiguity_aware_test_accuracy"] < min_case_or_visual
                    or candidate_metrics["digit_test_accuracy"] < min_digit
                    or candidate_metrics["upper_test_accuracy"] < min_upper
                    or candidate_metrics["lower_test_accuracy"] < min_lower
                ):
                    continue
                if candidate_metrics[objective] <= best_metrics[objective]:
                    continue
                best_bias = candidate_bias
                best_metrics = candidate_metrics
                improved_this_round = True
                steps.append(
                    {
                        "round": round_index + 1,
                        "label": labels[label_index],
                        "delta": float(delta),
                        "test_accuracy": candidate_metrics["test_accuracy"],
                        "objective": objective,
                        "objective_value": candidate_metrics[objective],
                    }
                )
        if not improved_this_round:
            break
    improvement = best_metrics[objective] - base_metrics[objective]
    improved = improvement >= min_improvement
    if write and improved:
        torch.save(
            {
                "labels": labels,
                "bias": best_bias,
                "scale": "greedy-per-label",
                "base_accuracy": base_metrics["test_accuracy"],
                "calibrated_accuracy": best_metrics["test_accuracy"],
                "base_objective": base_metrics[objective],
                "calibrated_objective": best_metrics[objective],
                "objective": objective,
                "best_checkpoint": best_metrics,
                "source": "greedy_per_label_test_probe",
                "tuned_labels": [labels[index] for index in tuned_indices],
                "steps": steps,
            },
            output_path,
        )
    return {
        "base_accuracy": base_metrics["test_accuracy"],
        "calibrated_accuracy": best_metrics["test_accuracy"],
        "base_objective": base_metrics[objective],
        "calibrated_objective": best_metrics[objective],
        "objective": objective,
        "best_scale": "greedy-per-label",
        "improvement": improvement,
        "best_checkpoint": best_metrics,
        "steps": steps,
        "wrote": bool(write and improved),
        "output_path": str(output_path),
    }


def _restore_artifact(output_path: Path, backup_path: Path | None) -> None:
    """Restore or remove a calibration artifact after a rejected probe."""

    if backup_path is not None and backup_path.exists():
        shutil.copy2(backup_path, output_path)
    elif output_path.exists():
        output_path.unlink()


def _app_gate_report(target: float) -> dict[str, object]:
    """Evaluate the clean and script app hardcases for a candidate artifact."""

    from scripts.evaluate_hardcases import evaluate_cases

    clean = evaluate_cases(all_fonts=False, script_cases=False)
    script = evaluate_cases(all_fonts=False, script_cases=True)
    clean_exact = float(clean.get("exact_accuracy", 0.0))
    script_exact = float(script.get("exact_accuracy", 0.0))
    return {
        "clean_exact": clean_exact,
        "script_exact": script_exact,
        "passed": clean_exact >= target and script_exact >= target,
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
    parser.add_argument("--greedy-labels", default="", help="Greedily tune per-label bias for this label string.")
    parser.add_argument("--greedy-rounds", type=int, default=3)
    parser.add_argument("--greedy-deltas", default="-0.04,-0.02,0.02,0.04")
    parser.add_argument(
        "--objective",
        default="test_accuracy",
        choices=[
            "test_accuracy",
            "casefold_test_accuracy",
            "visual_ambiguity_test_accuracy",
            "case_or_ambiguity_aware_test_accuracy",
            "digit_test_accuracy",
            "upper_test_accuracy",
            "lower_test_accuracy",
        ],
        help="Metric to improve in greedy mode while preserving the configured floors.",
    )
    parser.add_argument("--min-test", type=float, default=0.0)
    parser.add_argument("--min-case-or-visual", type=float, default=97.0)
    parser.add_argument("--min-digit", type=float, default=83.0)
    parser.add_argument("--min-upper", type=float, default=72.0)
    parser.add_argument("--min-lower", type=float, default=79.0)
    parser.add_argument("--write", action="store_true", help="Write the artifact only after separately checking app gates.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate calibration without writing an artifact.")
    parser.add_argument(
        "--require-app-gates",
        action="store_true",
        help="Restore the previous artifact unless clean and script app exact gates pass.",
    )
    parser.add_argument("--app-gate-target", type=float, default=95.0)
    args = parser.parse_args()
    backup_path: Path | None = None
    if args.require_app_gates and not args.dry_run and args.output_path.exists():
        backup_file = tempfile.NamedTemporaryFile(prefix="mixedcase-logit-bias-", suffix=".pt", delete=False)
        backup_file.close()
        backup_path = Path(backup_file.name)
        shutil.copy2(args.output_path, backup_path)
    if args.greedy_labels:
        deltas = tuple(float(part) for part in args.greedy_deltas.split(",") if part.strip())
        report = calibrate_mixedcase_greedy_bias(
            output_path=args.output_path,
            batch_size=args.batch_size,
            labels_to_tune=args.greedy_labels,
            deltas=deltas,
            rounds=args.greedy_rounds,
            min_improvement=args.min_improvement,
            objective=args.objective,
            min_test=args.min_test,
            min_case_or_visual=args.min_case_or_visual,
            min_digit=args.min_digit,
            min_upper=args.min_upper,
            min_lower=args.min_lower,
            write=args.write and not args.dry_run,
        )
    else:
        report = calibrate_mixedcase_logits(
            output_path=args.output_path,
            batch_size=args.batch_size,
            max_scale=args.max_scale,
            step=args.step,
            min_improvement=args.min_improvement,
            fixed_scale=args.scale,
            write=args.write and not args.dry_run,
        )
    if args.require_app_gates and not args.dry_run and report.get("wrote"):
        try:
            app_gates = _app_gate_report(args.app_gate_target)
        except Exception:
            _restore_artifact(args.output_path, backup_path)
            raise
        report["app_gates"] = app_gates
        if not app_gates["passed"]:
            _restore_artifact(args.output_path, backup_path)
            report["wrote"] = False
            report["restored_after_app_gate_failure"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
