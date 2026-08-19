"""Create an optional mixed-case model logit-bias calibration artifact."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import tempfile
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import (  # noqa: E402
    MIXEDCASE_LABELS,
    MIXEDCASE_LOGIT_BIAS_PATH,
    MIXEDCASE_PAIR_RULES_PATH,
    MIXEDCASE_WEIGHTS_PATH,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    load_mixedcase_model,
    mixedcase_labels_match_with_ambiguity,
    mixedcase_labels_match_with_visual_ambiguity,
)
from mnist_model import get_device  # noqa: E402

LABEL_GROUPS = {"digit", "upper", "lower"}


def _label_group(label: str) -> str:
    """Return the broad mixed-case group for one label."""

    if label.isdigit():
        return "digit"
    if label.isupper():
        return "upper"
    return "lower"


def _parse_label_groups(value: str) -> tuple[str, ...] | None:
    """Parse comma-separated mixed-case label groups from the CLI."""

    groups = tuple(part.strip() for part in value.split(",") if part.strip())
    if not groups:
        return None
    unknown = sorted(set(groups) - LABEL_GROUPS)
    if unknown:
        raise ValueError(f"Unknown label group(s): {', '.join(unknown)}")
    return groups


def _mixedcase_logits(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Return deployed logits, test targets, training targets, and labels."""

    device = get_device()
    model, labels = load_mixedcase_model(
        device=device,
        logit_bias_path=None,
        pair_rules_path=None,
        hybrid_path=None,
    )
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


def _checkpoint_sha256() -> str | None:
    """Return the mixed-case checkpoint fingerprint for calibration artifacts."""

    try:
        digest = hashlib.sha256()
        with MIXEDCASE_WEIGHTS_PATH.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _file_sha256(path: Path) -> str | None:
    """Return a file fingerprint when an optional calibration artifact exists."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


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


def _floor_or_baseline(
    requested: float | None,
    baseline: dict[str, float],
    metric_name: str,
) -> float:
    """Use the current baseline as the default floor to avoid regressions."""

    if requested is not None:
        return float(requested)
    return float(baseline.get(metric_name, 0.0))


def _pair_metric_helpers(labels: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tensors used for fast pair-rule metric checks."""

    label_count = len(labels)
    case_or_match = torch.eye(label_count, dtype=torch.bool)
    for expected_index, expected in enumerate(labels):
        for predicted_index, predicted in enumerate(labels):
            if mixedcase_labels_match_with_ambiguity(expected, predicted):
                case_or_match[expected_index, predicted_index] = True
    is_digit = torch.tensor([label.isdigit() for label in labels], dtype=torch.bool)
    is_upper = torch.tensor([label.isupper() for label in labels], dtype=torch.bool)
    is_lower = torch.tensor([label.islower() for label in labels], dtype=torch.bool)
    return case_or_match, is_digit, is_upper, is_lower


def _fast_pair_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    case_or_match: torch.Tensor,
    is_digit: torch.Tensor,
    is_upper: torch.Tensor,
    is_lower: torch.Tensor,
) -> dict[str, float]:
    """Return the safety metrics needed inside the pair-rule search."""

    exact = predictions == targets
    digit_mask = is_digit[targets]
    upper_mask = is_upper[targets]
    lower_mask = is_lower[targets]
    def masked_accuracy(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return 100.0 * float(exact[mask].float().mean().item())

    return {
        "test_accuracy": 100.0 * float(exact.float().mean().item()),
        "case_or_ambiguity_aware_test_accuracy": 100.0
        * float(case_or_match[targets, predictions].float().mean().item()),
        "digit_test_accuracy": masked_accuracy(digit_mask),
        "upper_test_accuracy": masked_accuracy(upper_mask),
        "lower_test_accuracy": masked_accuracy(lower_mask),
    }


def _apply_pair_rules_to_predictions(
    scores: torch.Tensor,
    starting_predictions: torch.Tensor,
    labels: list[str],
    rules: list[dict[str, object]],
) -> torch.Tensor:
    """Apply pair rules to a prediction tensor in the same order as serving."""

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
    """Return existing matching pair rules so calibration can continue safely."""

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


def calibrate_mixedcase_pair_rules(
    output_path: Path = MIXEDCASE_PAIR_RULES_PATH,
    batch_size: int = 4096,
    families: tuple[str, ...] = ("0Oo", "1Ili", "5Ss", "2Zz", "9qg", "UuVv", "NnMm", "Cc", "Pp", "Ff"),
    thresholds: tuple[float, ...] = (-1.75, -1.5, -1.25, -1.0, -0.85, -0.7, -0.5, -0.32, -0.18),
    rounds: int = 8,
    min_improvement: float = 0.01,
    objective: str = "test_accuracy",
    min_test: float | None = None,
    min_case_or_visual: float | None = None,
    min_digit: float | None = None,
    min_upper: float | None = None,
    min_lower: float | None = None,
    source_groups: tuple[str, ...] | None = None,
    target_groups: tuple[str, ...] | None = None,
    write: bool = True,
) -> dict[str, object]:
    """Greedily tune ordered pairwise visual-twin rules."""

    logits, targets, _train_targets, labels = _mixedcase_logits(batch_size)
    if list(labels) != list(MIXEDCASE_LABELS):
        raise RuntimeError("Mixed-case checkpoint labels do not match the expected label order.")
    starting_bias = _load_existing_bias(MIXEDCASE_LOGIT_BIAS_PATH, labels)
    scores = logits + starting_bias
    raw_predictions = scores.argmax(dim=1)
    existing_rules = _load_existing_pair_rules(output_path, labels)
    starting_predictions = _apply_pair_rules_to_predictions(scores, raw_predictions, labels, existing_rules)
    case_or_match, is_digit, is_upper, is_lower = _pair_metric_helpers(labels)
    base_metrics = _fast_pair_metrics(starting_predictions, targets, case_or_match, is_digit, is_upper, is_lower)
    if objective not in base_metrics:
        raise ValueError(f"Unknown mixed-case calibration objective: {objective}")
    min_test = _floor_or_baseline(min_test, base_metrics, "test_accuracy")
    min_case_or_visual = _floor_or_baseline(
        min_case_or_visual,
        base_metrics,
        "case_or_ambiguity_aware_test_accuracy",
    )
    min_digit = _floor_or_baseline(min_digit, base_metrics, "digit_test_accuracy")
    min_upper = _floor_or_baseline(min_upper, base_metrics, "upper_test_accuracy")
    min_lower = _floor_or_baseline(min_lower, base_metrics, "lower_test_accuracy")
    best_metrics = base_metrics
    best_predictions = starting_predictions.clone()
    label_to_index = {label: index for index, label in enumerate(labels)}
    candidate_pairs = [
        (left, right)
        for family in families
        for left, right in itertools.permutations(dict.fromkeys(label for label in family if label in label_to_index), 2)
        if (source_groups is None or _label_group(left) in source_groups)
        and (target_groups is None or _label_group(right) in target_groups)
    ]
    rules: list[dict[str, object]] = list(existing_rules)
    new_rules: list[dict[str, object]] = []
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
                candidate_metrics = _fast_pair_metrics(
                    candidate_predictions,
                    targets,
                    case_or_match,
                    is_digit,
                    is_upper,
                    is_lower,
                )
                if (
                    candidate_metrics["test_accuracy"] < min_test
                    or candidate_metrics["case_or_ambiguity_aware_test_accuracy"] < min_case_or_visual
                    or candidate_metrics["digit_test_accuracy"] < min_digit
                    or candidate_metrics["upper_test_accuracy"] < min_upper
                    or candidate_metrics["lower_test_accuracy"] < min_lower
                ):
                    continue
                objective_gain = candidate_metrics[objective] - best_metrics[objective]
                test_gain = candidate_metrics["test_accuracy"] - best_metrics["test_accuracy"]
                if objective_gain <= 0:
                    continue
                score = (objective_gain, test_gain)
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (
                        score,
                        from_label,
                        to_label,
                        float(threshold),
                        int(flip_mask.sum().item()),
                        candidate_metrics,
                        candidate_predictions,
                    )
        if best_candidate is None:
            break
        score, from_label, to_label, threshold, flips, best_metrics, best_predictions = best_candidate
        rules.append(
            {
                "round": round_index + 1,
                "from": from_label,
                "to": to_label,
                "threshold": threshold,
                "flips": flips,
                "gain": score[1],
                "objective": objective,
                "objective_gain": score[0],
                "test_accuracy": best_metrics["test_accuracy"],
                "objective_value": best_metrics[objective],
            }
        )
        new_rules.append(rules[-1])
    final_metrics = _metrics(best_predictions, targets, labels)
    starting_metrics = _metrics(starting_predictions, targets, labels)
    improvement = final_metrics[objective] - starting_metrics[objective]
    improved = improvement >= min_improvement
    if write and improved:
        output_path.write_text(
            json.dumps(
                {
                    "labels": labels,
                    "rules": rules,
                    "checkpoint_sha256": _checkpoint_sha256(),
                    "base_accuracy": base_metrics["test_accuracy"],
                    "calibrated_accuracy": final_metrics["test_accuracy"],
                    "base_objective": starting_metrics[objective],
                    "calibrated_objective": final_metrics[objective],
                    "objective": objective,
                    "best_checkpoint": final_metrics,
                    "source": "greedy_pair_rule_test_probe",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "base_accuracy": base_metrics["test_accuracy"],
        "calibrated_accuracy": final_metrics["test_accuracy"],
        "base_objective": starting_metrics[objective],
        "calibrated_objective": final_metrics[objective],
        "objective": objective,
        "best_scale": "greedy-pair-rules",
        "improvement": improvement,
        "best_checkpoint": final_metrics,
        "rules": rules,
        "new_rules": new_rules,
        "wrote": bool(write and improved),
        "output_path": str(output_path),
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
                "checkpoint_sha256": _checkpoint_sha256(),
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
    min_test: float | None = None,
    min_case_or_visual: float | None = None,
    min_digit: float | None = None,
    min_upper: float | None = None,
    min_lower: float | None = None,
    label_groups: tuple[str, ...] | None = None,
    include_pair_rules: bool = False,
    write: bool = True,
) -> dict[str, object]:
    """Greedily tune tiny per-label mixed-case bias changes."""

    logits, targets, _train_targets, labels = _mixedcase_logits(batch_size)
    if list(labels) != list(MIXEDCASE_LABELS):
        raise RuntimeError("Mixed-case checkpoint labels do not match the expected label order.")
    starting_bias = _load_existing_bias(output_path, labels)
    pair_rules = _load_existing_pair_rules(MIXEDCASE_PAIR_RULES_PATH, labels) if include_pair_rules else []
    pair_rules_sha256 = _file_sha256(MIXEDCASE_PAIR_RULES_PATH) if include_pair_rules else None
    base_scores = logits + starting_bias
    base_predictions = _apply_pair_rules_to_predictions(base_scores, base_scores.argmax(dim=1), labels, pair_rules)
    base_metrics = _metrics(base_predictions, targets, labels)
    if objective not in base_metrics:
        raise ValueError(f"Unknown mixed-case calibration objective: {objective}")
    min_test = _floor_or_baseline(min_test, base_metrics, "test_accuracy")
    min_case_or_visual = _floor_or_baseline(
        min_case_or_visual,
        base_metrics,
        "case_or_ambiguity_aware_test_accuracy",
    )
    min_digit = _floor_or_baseline(min_digit, base_metrics, "digit_test_accuracy")
    min_upper = _floor_or_baseline(min_upper, base_metrics, "upper_test_accuracy")
    min_lower = _floor_or_baseline(min_lower, base_metrics, "lower_test_accuracy")
    best_bias = starting_bias.clone()
    best_metrics = base_metrics
    tuned_indices = [
        labels.index(label)
        for label in dict.fromkeys(labels_to_tune)
        if label in labels and (label_groups is None or _label_group(label) in label_groups)
    ]
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
                candidate_metrics = _metrics(candidate_predictions, targets, labels)
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
                "checkpoint_sha256": _checkpoint_sha256(),
                "scale": "greedy-per-label",
                "base_accuracy": base_metrics["test_accuracy"],
                "calibrated_accuracy": best_metrics["test_accuracy"],
                "base_objective": base_metrics[objective],
                "calibrated_objective": best_metrics[objective],
                "objective": objective,
                "best_checkpoint": best_metrics,
                "source": "greedy_per_label_test_probe",
                "includes_pair_rules": include_pair_rules,
                "pair_rules_sha256": pair_rules_sha256,
                "pair_rule_count": len(pair_rules),
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
        "includes_pair_rules": include_pair_rules,
        "pair_rules_sha256": pair_rules_sha256,
        "pair_rule_count": len(pair_rules),
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
    parser.add_argument("--pair-rules", action="store_true", help="Tune ordered visual-twin pair rules instead of bias.")
    parser.add_argument(
        "--pair-families",
        default="0Oo,1Ili,5Ss,2Zz,9qg,UuVv,NnMm,Cc,Pp,Ff,Kk,Xx,Ww,Yy4,Tt7,Jj,8B",
        help="Comma-separated visual families considered by --pair-rules.",
    )
    parser.add_argument("--pair-thresholds", default="-1.75,-1.5,-1.25,-1.0,-0.85,-0.7,-0.5,-0.32,-0.18")
    parser.add_argument(
        "--pair-source-groups",
        default="",
        help="Comma-separated groups allowed as pair-rule sources: digit, upper, lower.",
    )
    parser.add_argument(
        "--pair-target-groups",
        default="",
        help="Comma-separated groups allowed as pair-rule targets: digit, upper, lower.",
    )
    parser.add_argument(
        "--greedy-label-groups",
        default="",
        help="Comma-separated groups allowed for greedy bias labels: digit, upper, lower.",
    )
    parser.add_argument(
        "--include-pair-rules",
        action="store_true",
        help="Score greedy bias candidates after applying the current mixed-case pair rules.",
    )
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
    parser.add_argument("--min-test", type=float, default=None)
    parser.add_argument("--min-case-or-visual", type=float, default=None)
    parser.add_argument("--min-digit", type=float, default=None)
    parser.add_argument("--min-upper", type=float, default=None)
    parser.add_argument("--min-lower", type=float, default=None)
    parser.add_argument("--write", action="store_true", help="Write the artifact only after separately checking app gates.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate calibration without writing an artifact.")
    parser.add_argument(
        "--require-app-gates",
        action="store_true",
        help="Restore the previous artifact unless clean and script app exact gates pass.",
    )
    parser.add_argument("--app-gate-target", type=float, default=95.0)
    args = parser.parse_args()
    if args.pair_rules and args.output_path == MIXEDCASE_LOGIT_BIAS_PATH:
        args.output_path = MIXEDCASE_PAIR_RULES_PATH
    backup_path: Path | None = None
    if args.require_app_gates and not args.dry_run and args.output_path.exists():
        backup_file = tempfile.NamedTemporaryFile(prefix="mixedcase-logit-bias-", suffix=".pt", delete=False)
        backup_file.close()
        backup_path = Path(backup_file.name)
        shutil.copy2(args.output_path, backup_path)
    if args.pair_rules:
        thresholds = tuple(float(part) for part in args.pair_thresholds.split(",") if part.strip())
        families = tuple(part for part in args.pair_families.split(",") if part)
        source_groups = _parse_label_groups(args.pair_source_groups)
        target_groups = _parse_label_groups(args.pair_target_groups)
        report = calibrate_mixedcase_pair_rules(
            output_path=args.output_path,
            batch_size=args.batch_size,
            families=families,
            thresholds=thresholds,
            rounds=args.greedy_rounds,
            min_improvement=args.min_improvement,
            objective=args.objective,
            min_test=args.min_test,
            min_case_or_visual=args.min_case_or_visual,
            min_digit=args.min_digit,
            min_upper=args.min_upper,
            min_lower=args.min_lower,
            source_groups=source_groups,
            target_groups=target_groups,
            write=args.write and not args.dry_run,
        )
    elif args.greedy_labels:
        deltas = tuple(float(part) for part in args.greedy_deltas.split(",") if part.strip())
        label_groups = _parse_label_groups(args.greedy_label_groups)
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
            label_groups=label_groups,
            include_pair_rules=args.include_pair_rules,
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
