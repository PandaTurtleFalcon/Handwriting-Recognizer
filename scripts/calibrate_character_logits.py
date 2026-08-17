"""Create an optional character-model logit-bias calibration artifact."""

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

from character_model import (  # noqa: E402
    DATASET_ROOT,
    LOGIT_BIAS_PATH,
    build_or_load_combined_cache,
    labels_match_with_ambiguity,
    load_character_model,
    stratified_split_indices,
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
    train_indices, validation_indices = stratified_split_indices(
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


def calibrate_character_greedy_bias(
    output_path: Path = LOGIT_BIAS_PATH,
    batch_size: int = 2048,
    labels_to_tune: str = "",
    deltas: tuple[float, ...] = (-0.12, -0.08, -0.04, 0.04, 0.08, 0.12),
    rounds: int = 6,
    min_improvement: float = 0.01,
    min_ambiguity: float = 98.8,
    min_punctuation: float = 95.0,
    write: bool = True,
) -> dict[str, object]:
    """Greedily tune tiny per-label bias changes from the current artifact."""

    logits, targets, _train_targets, labels = _validation_logits(batch_size)
    starting_bias = _load_existing_bias(output_path, labels)
    base_breakdown = _breakdown((logits + starting_bias).argmax(dim=1), targets, labels)
    best_bias = starting_bias.clone()
    best_breakdown = base_breakdown
    tuned_indices = [labels.index(label) for label in dict.fromkeys(labels_to_tune) if label in labels]
    steps: list[dict[str, object]] = []
    for round_index in range(max(0, rounds)):
        improved_this_round = False
        for label_index in tuned_indices:
            for delta in deltas:
                candidate_bias = best_bias.clone()
                candidate_bias[label_index] += float(delta)
                candidate_breakdown = _breakdown((logits + candidate_bias).argmax(dim=1), targets, labels)
                if (
                    candidate_breakdown["punctuation_validation_accuracy"] < min_punctuation
                    or candidate_breakdown["ambiguity_aware_validation_accuracy"] < min_ambiguity
                ):
                    continue
                if candidate_breakdown["validation_accuracy"] <= best_breakdown["validation_accuracy"]:
                    continue
                best_bias = candidate_bias
                best_breakdown = candidate_breakdown
                improved_this_round = True
                steps.append(
                    {
                        "round": round_index + 1,
                        "label": labels[label_index],
                        "delta": float(delta),
                        "validation_accuracy": candidate_breakdown["validation_accuracy"],
                    }
                )
        if not improved_this_round:
            break
    improvement = best_breakdown["validation_accuracy"] - base_breakdown["validation_accuracy"]
    improved = improvement >= min_improvement
    if write and improved:
        torch.save(
            {
                "labels": labels,
                "bias": best_bias,
                "scale": "greedy-per-label",
                "base_accuracy": base_breakdown["validation_accuracy"],
                "calibrated_accuracy": best_breakdown["validation_accuracy"],
                "best_checkpoint": best_breakdown,
                "source": "greedy_per_label_validation_probe",
                "tuned_labels": [labels[index] for index in tuned_indices],
                "steps": steps,
            },
            output_path,
        )
    return {
        "base_accuracy": base_breakdown["validation_accuracy"],
        "calibrated_accuracy": best_breakdown["validation_accuracy"],
        "best_scale": "greedy-per-label",
        "improvement": improvement,
        "best_checkpoint": best_breakdown,
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

    parser = argparse.ArgumentParser(description="Calibrate character-model logits with train-prior bias.")
    parser.add_argument("--output-path", type=Path, default=LOGIT_BIAS_PATH)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-scale", type=float, default=1.5)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=None, help="Write this fixed scale instead of the validation optimum.")
    parser.add_argument("--greedy-labels", default="", help="Greedily tune per-label bias for this label string.")
    parser.add_argument("--greedy-rounds", type=int, default=6)
    parser.add_argument("--greedy-deltas", default="-0.12,-0.08,-0.04,0.04,0.08,0.12")
    parser.add_argument("--min-ambiguity", type=float, default=98.8)
    parser.add_argument("--min-punctuation", type=float, default=95.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-app-gates",
        action="store_true",
        help="Restore the previous artifact unless clean and script app exact gates pass.",
    )
    parser.add_argument("--app-gate-target", type=float, default=95.0)
    args = parser.parse_args()
    backup_path: Path | None = None
    if args.require_app_gates and not args.dry_run and args.output_path.exists():
        backup_file = tempfile.NamedTemporaryFile(prefix="character-logit-bias-", suffix=".pt", delete=False)
        backup_file.close()
        backup_path = Path(backup_file.name)
        shutil.copy2(args.output_path, backup_path)
    if args.greedy_labels:
        deltas = tuple(float(part) for part in args.greedy_deltas.split(",") if part.strip())
        report = calibrate_character_greedy_bias(
            output_path=args.output_path,
            batch_size=args.batch_size,
            labels_to_tune=args.greedy_labels,
            deltas=deltas,
            rounds=args.greedy_rounds,
            min_improvement=args.min_improvement,
            min_ambiguity=args.min_ambiguity,
            min_punctuation=args.min_punctuation,
            write=not args.dry_run,
        )
    else:
        report = calibrate_character_logits(
            output_path=args.output_path,
            batch_size=args.batch_size,
            max_scale=args.max_scale,
            step=args.step,
            min_improvement=args.min_improvement,
            fixed_scale=args.scale,
            write=not args.dry_run,
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
