"""Create an optional character-model logit-bias calibration artifact."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
    PAIR_RULES_PATH,
    WEIGHTS_PATH,
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
    model, labels = load_character_model(device=device, logit_bias_path=None, pair_rules_path=None)
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


def _checkpoint_sha256() -> str | None:
    """Return the character checkpoint fingerprint for calibration artifacts."""

    try:
        digest = hashlib.sha256()
        with WEIGHTS_PATH.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _file_sha256(path: Path) -> str | None:
    """Return a stable file digest for dependent calibration artifacts."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


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


def calibrate_character_pair_rules(
    output_path: Path = PAIR_RULES_PATH,
    batch_size: int = 2048,
    families: tuple[str, ...] = ("0Oo", "1Ili|!/", "5Ss", "2Zz", "9qg", "UuVv", "NnMm", "Cc", "Pp", "Ff"),
    thresholds: tuple[float, ...] = (-2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.85, -0.7, -0.5, -0.32, -0.18),
    rounds: int = 10,
    min_improvement: float = 0.01,
    objective: str = "letter_validation_accuracy",
    min_validation: float = 0.0,
    min_ambiguity: float = 98.8,
    min_digit: float = 0.0,
    min_letter: float = 0.0,
    min_punctuation: float = 95.0,
    write: bool = True,
) -> dict[str, object]:
    """Greedily tune ordered pairwise visual-twin rules for character logits."""

    logits, targets, _train_targets, labels = _validation_logits(batch_size)
    starting_bias = _load_existing_bias(LOGIT_BIAS_PATH, labels)
    scores = logits + starting_bias
    raw_predictions = scores.argmax(dim=1)
    existing_rules = _load_existing_pair_rules(output_path, labels)
    starting_predictions = _apply_pair_rules_to_predictions(scores, raw_predictions, labels, existing_rules)
    base_breakdown = _breakdown(starting_predictions, targets, labels)
    if objective not in base_breakdown:
        raise ValueError(f"Unknown character calibration objective: {objective}")
    best_predictions = starting_predictions.clone()
    best_breakdown = base_breakdown
    label_to_index = {label: index for index, label in enumerate(labels)}
    candidate_pairs = [
        (left, right)
        for family in families
        for left, right in itertools.permutations(dict.fromkeys(label for label in family if label in label_to_index), 2)
    ]
    steps: list[dict[str, object]] = list(existing_rules)
    new_steps: list[dict[str, object]] = []
    for round_index in range(max(0, rounds)):
        best_candidate: tuple[tuple[float, float], str, str, float, int, dict[str, float], torch.Tensor] | None = None
        for from_label, to_label in candidate_pairs:
            from_index = label_to_index[from_label]
            to_index = label_to_index[to_label]
            current_mask = best_predictions == from_index
            if not bool(current_mask.any()):
                continue
            margin = scores[:, to_index] - scores[:, from_index]
            for threshold in thresholds:
                flip_mask = current_mask & (margin >= threshold)
                if not bool(flip_mask.any()):
                    continue
                candidate_predictions = best_predictions.clone()
                candidate_predictions[flip_mask] = to_index
                candidate_breakdown = _breakdown(candidate_predictions, targets, labels)
                if (
                    candidate_breakdown["validation_accuracy"] < min_validation
                    or candidate_breakdown["ambiguity_aware_validation_accuracy"] < min_ambiguity
                    or candidate_breakdown["digit_validation_accuracy"] < min_digit
                    or candidate_breakdown["letter_validation_accuracy"] < min_letter
                    or candidate_breakdown["punctuation_validation_accuracy"] < min_punctuation
                ):
                    continue
                objective_gain = candidate_breakdown[objective] - best_breakdown[objective]
                validation_gain = candidate_breakdown["validation_accuracy"] - best_breakdown["validation_accuracy"]
                if objective_gain <= 0:
                    continue
                score = (objective_gain, validation_gain)
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (
                        score,
                        from_label,
                        to_label,
                        float(threshold),
                        int(flip_mask.sum().item()),
                        candidate_breakdown,
                        candidate_predictions,
                    )
        if best_candidate is None:
            break
        score, from_label, to_label, threshold, flips, best_breakdown, best_predictions = best_candidate
        steps.append(
            {
                "round": round_index + 1,
                "from": from_label,
                "to": to_label,
                "threshold": threshold,
                "flips": flips,
                "objective": objective,
                "objective_gain": score[0],
                "validation_accuracy": best_breakdown["validation_accuracy"],
                "objective_value": best_breakdown[objective],
            }
        )
        new_steps.append(steps[-1])
    improvement = best_breakdown[objective] - base_breakdown[objective]
    improved = improvement >= min_improvement
    if write and improved:
        output_path.write_text(
            json.dumps(
                {
                    "labels": labels,
                    "rules": steps,
                    "checkpoint_sha256": _checkpoint_sha256(),
                    "base_accuracy": base_breakdown["validation_accuracy"],
                    "calibrated_accuracy": best_breakdown["validation_accuracy"],
                    "base_objective": base_breakdown[objective],
                    "calibrated_objective": best_breakdown[objective],
                    "objective": objective,
                    "best_checkpoint": best_breakdown,
                    "source": "greedy_pair_rule_validation_probe",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "base_accuracy": base_breakdown["validation_accuracy"],
        "calibrated_accuracy": best_breakdown["validation_accuracy"],
        "base_objective": base_breakdown[objective],
        "calibrated_objective": best_breakdown[objective],
        "objective": objective,
        "best_scale": "greedy-pair-rules",
        "improvement": improvement,
        "best_checkpoint": best_breakdown,
        "steps": steps,
        "new_steps": new_steps,
        "wrote": bool(write and improved),
        "output_path": str(output_path),
    }


def _apply_pair_rules_to_predictions(
    scores: torch.Tensor,
    starting_predictions: torch.Tensor,
    labels: list[str],
    rules: list[dict[str, object]],
) -> torch.Tensor:
    """Apply character pair rules to predictions in serving order."""

    label_to_index = {label: index for index, label in enumerate(labels)}
    predictions = starting_predictions.clone()
    for rule in rules:
        from_label = str(rule.get("from", ""))
        to_label = str(rule.get("to", ""))
        if from_label not in label_to_index or to_label not in label_to_index:
            continue
        try:
            threshold = float(rule["threshold"])
        except (KeyError, TypeError, ValueError):
            continue
        from_index = label_to_index[from_label]
        to_index = label_to_index[to_label]
        margin = scores[:, to_index] - scores[:, from_index]
        predictions[(predictions == from_index) & (margin >= threshold)] = to_index
    return predictions


def _load_existing_pair_rules(output_path: Path, labels: list[str]) -> list[dict[str, object]]:
    """Return existing matching character pair rules for continuation runs."""

    if not output_path.exists():
        return []
    try:
        artifact = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(artifact, dict) or list(artifact.get("labels", [])) != list(labels):
        return []
    rules = artifact.get("rules", [])
    return [rule for rule in rules if isinstance(rule, dict)]


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
                "checkpoint_sha256": _checkpoint_sha256(),
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
    objective: str = "validation_accuracy",
    min_validation: float = 0.0,
    min_ambiguity: float = 98.8,
    min_digit: float = 0.0,
    min_letter: float = 0.0,
    min_punctuation: float = 95.0,
    include_pair_rules: bool = False,
    write: bool = True,
) -> dict[str, object]:
    """Greedily tune tiny per-label bias changes from the current artifact."""

    logits, targets, _train_targets, labels = _validation_logits(batch_size)
    starting_bias = _load_existing_bias(output_path, labels)
    pair_rules = _load_existing_pair_rules(PAIR_RULES_PATH, labels) if include_pair_rules else []
    base_scores = logits + starting_bias
    base_predictions = _apply_pair_rules_to_predictions(base_scores, base_scores.argmax(dim=1), labels, pair_rules)
    base_breakdown = _breakdown(base_predictions, targets, labels)
    if objective not in base_breakdown:
        raise ValueError(f"Unknown character calibration objective: {objective}")
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
                candidate_scores = logits + candidate_bias
                candidate_predictions = _apply_pair_rules_to_predictions(
                    candidate_scores,
                    candidate_scores.argmax(dim=1),
                    labels,
                    pair_rules,
                )
                candidate_breakdown = _breakdown(candidate_predictions, targets, labels)
                if (
                    candidate_breakdown["validation_accuracy"] < min_validation
                    or candidate_breakdown["punctuation_validation_accuracy"] < min_punctuation
                    or candidate_breakdown["ambiguity_aware_validation_accuracy"] < min_ambiguity
                    or candidate_breakdown["digit_validation_accuracy"] < min_digit
                    or candidate_breakdown["letter_validation_accuracy"] < min_letter
                ):
                    continue
                if candidate_breakdown[objective] <= best_breakdown[objective]:
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
                        "objective": objective,
                        "objective_value": candidate_breakdown[objective],
                    }
                )
        if not improved_this_round:
            break
    improvement = best_breakdown[objective] - base_breakdown[objective]
    improved = improvement >= min_improvement
    if write and improved:
        torch.save(
            {
                "labels": labels,
                "bias": best_bias,
                "checkpoint_sha256": _checkpoint_sha256(),
                "scale": "greedy-per-label",
                "base_accuracy": base_breakdown["validation_accuracy"],
                "calibrated_accuracy": best_breakdown["validation_accuracy"],
                "base_objective": base_breakdown[objective],
                "calibrated_objective": best_breakdown[objective],
                "objective": objective,
                "best_checkpoint": best_breakdown,
                "source": "greedy_per_label_validation_probe",
                "includes_pair_rules": include_pair_rules,
                "pair_rules_sha256": _file_sha256(PAIR_RULES_PATH) if include_pair_rules else None,
                "tuned_labels": [labels[index] for index in tuned_indices],
                "steps": steps,
            },
            output_path,
        )
    return {
        "base_accuracy": base_breakdown["validation_accuracy"],
        "calibrated_accuracy": best_breakdown["validation_accuracy"],
        "base_objective": base_breakdown[objective],
        "calibrated_objective": best_breakdown[objective],
        "objective": objective,
        "best_scale": "greedy-per-label",
        "improvement": improvement,
        "best_checkpoint": best_breakdown,
        "steps": steps,
        "includes_pair_rules": include_pair_rules,
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
    parser.add_argument(
        "--include-pair-rules",
        action="store_true",
        help="Evaluate greedy bias candidates after applying current character pair rules.",
    )
    parser.add_argument("--pair-rules", action="store_true", help="Tune ordered visual-twin pair rules instead of bias.")
    parser.add_argument(
        "--pair-families",
        default="0Oo,1Ili|!/,5Ss,2Zz,9qg,UuVv,NnMm,Cc,Pp,Ff,Kk,Xx,Ww,Yy4,Tt7,Jj,8B,-_,.'`,:;i!,+t",
        help="Comma-separated visual families considered by --pair-rules.",
    )
    parser.add_argument(
        "--pair-thresholds",
        default="-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.85,-0.7,-0.5,-0.32,-0.18",
    )
    parser.add_argument(
        "--objective",
        default="validation_accuracy",
        choices=[
            "validation_accuracy",
            "ambiguity_aware_validation_accuracy",
            "digit_validation_accuracy",
            "letter_validation_accuracy",
            "punctuation_validation_accuracy",
            "punctuation_ambiguity_aware_validation_accuracy",
        ],
        help="Metric to improve in greedy mode while preserving the configured floors.",
    )
    parser.add_argument("--min-validation", type=float, default=0.0)
    parser.add_argument("--min-ambiguity", type=float, default=98.8)
    parser.add_argument("--min-digit", type=float, default=0.0)
    parser.add_argument("--min-letter", type=float, default=0.0)
    parser.add_argument("--min-punctuation", type=float, default=95.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-app-gates",
        action="store_true",
        help="Restore the previous artifact unless clean and script app exact gates pass.",
    )
    parser.add_argument("--app-gate-target", type=float, default=95.0)
    args = parser.parse_args()
    if args.pair_rules and args.output_path == LOGIT_BIAS_PATH:
        args.output_path = PAIR_RULES_PATH
    backup_path: Path | None = None
    if args.require_app_gates and not args.dry_run and args.output_path.exists():
        backup_file = tempfile.NamedTemporaryFile(prefix="character-logit-bias-", suffix=".pt", delete=False)
        backup_file.close()
        backup_path = Path(backup_file.name)
        shutil.copy2(args.output_path, backup_path)
    if args.pair_rules:
        thresholds = tuple(float(part) for part in args.pair_thresholds.split(",") if part.strip())
        families = tuple(part for part in args.pair_families.split(",") if part)
        report = calibrate_character_pair_rules(
            output_path=args.output_path,
            batch_size=args.batch_size,
            families=families,
            thresholds=thresholds,
            rounds=args.greedy_rounds,
            min_improvement=args.min_improvement,
            objective=args.objective,
            min_validation=args.min_validation,
            min_ambiguity=args.min_ambiguity,
            min_digit=args.min_digit,
            min_letter=args.min_letter,
            min_punctuation=args.min_punctuation,
            write=not args.dry_run,
        )
    elif args.greedy_labels:
        deltas = tuple(float(part) for part in args.greedy_deltas.split(",") if part.strip())
        report = calibrate_character_greedy_bias(
            output_path=args.output_path,
            batch_size=args.batch_size,
            labels_to_tune=args.greedy_labels,
            deltas=deltas,
            rounds=args.greedy_rounds,
            min_improvement=args.min_improvement,
            objective=args.objective,
            min_validation=args.min_validation,
            min_ambiguity=args.min_ambiguity,
            min_digit=args.min_digit,
            min_letter=args.min_letter,
            min_punctuation=args.min_punctuation,
            include_pair_rules=args.include_pair_rules,
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
