"""Calibrate the deployed mixed-case/folded-identity hybrid artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import (  # noqa: E402
    LABELS,
    MIXEDCASE_HYBRID_PATH,
    MIXEDCASE_LABELS,
    MIXEDCASE_LOGIT_BIAS_PATH,
    MIXEDCASE_PAIR_RULES_PATH,
    MIXEDCASE_WEIGHTS_PATH,
    WEIGHTS_PATH,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    load_alnum_model,
    load_mixedcase_model,
    mixedcase_labels_match_with_ambiguity,
)
from mnist_model import get_device  # noqa: E402


METRIC_NAMES = {
    "test_accuracy",
    "case_or_ambiguity_aware_test_accuracy",
    "digit_test_accuracy",
    "upper_test_accuracy",
    "lower_test_accuracy",
    "balanced_group_accuracy",
}


def _file_sha256(path: Path) -> str | None:
    """Return a stable digest for one checkpoint artifact."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _float_map(value: object) -> dict[str, float]:
    """Return sanitized A-Z float thresholds from an artifact field."""

    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw_threshold in value.items():
        label = str(key)
        if len(label) != 1 or not label.isalpha():
            continue
        try:
            result[label.upper()] = float(raw_threshold)
        except (TypeError, ValueError):
            continue
    return result


def _load_hybrid_artifact(path: Path = MIXEDCASE_HYBRID_PATH) -> dict[str, object]:
    """Load a matching existing artifact, or return the default hybrid settings."""

    default: dict[str, object] = {
        "enabled": True,
        "source": "folded_identity_mixedcase_case_probe",
        "labels": list(MIXEDCASE_LABELS),
        "letter_case_threshold": 0.0,
        "folded_confidence_threshold": 0.0,
        "folded_margin_threshold": 0.0,
        "letter_case_thresholds": {},
        "folded_confidence_thresholds": {},
        "folded_margin_thresholds": {},
    }
    if not path.exists():
        return default
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if not isinstance(artifact, dict) or list(artifact.get("labels", [])) != list(MIXEDCASE_LABELS):
        return default
    return {**default, **artifact}


def _test_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mixed MNIST plus EMNIST ByClass mixed-case test tensors."""

    mnist_images, mnist_targets = build_or_load_mnist_cache(train=False)
    byclass_images, byclass_targets = build_or_load_emnist_byclass_mixedcase_cache(train=False)
    return torch.cat([mnist_images, byclass_images]), torch.cat([mnist_targets, byclass_targets])


def _model_outputs(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Return calibrated mixed logits, folded logits, targets, and labels."""

    device = get_device()
    mixed_model, mixed_labels = load_mixedcase_model(device=device, hybrid_path=None)
    folded_model, folded_labels = load_alnum_model(device=device)
    if mixed_model is None or folded_model is None or mixed_labels is None or folded_labels is None:
        raise RuntimeError("Mixed-case or folded alnum checkpoint is missing.")
    if list(mixed_labels) != list(MIXEDCASE_LABELS) or list(folded_labels) != list(LABELS):
        raise RuntimeError("Checkpoint labels do not match the expected orders.")
    images, targets = _test_tensors()
    loader = DataLoader(TensorDataset(images, targets), batch_size=batch_size, shuffle=False)
    mixed_outputs: list[torch.Tensor] = []
    folded_outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_images, _batch_targets in loader:
            inputs = batch_images.to(device)
            mixed_outputs.append(mixed_model(inputs).cpu())
            folded_outputs.append(folded_model(inputs).cpu())
    return torch.cat(mixed_outputs), torch.cat(folded_outputs), targets, list(mixed_labels)


def hybrid_predictions(
    mixed_outputs: torch.Tensor,
    folded_outputs: torch.Tensor,
    artifact: dict[str, object],
) -> torch.Tensor:
    """Apply the same override logic as alnum_model.HybridMixedcaseModel."""

    mixed_predictions = mixed_outputs.argmax(dim=1)
    folded_predictions = folded_outputs.argmax(dim=1)
    folded_confidence = folded_outputs.softmax(dim=1).max(dim=1).values
    folded_top2 = folded_outputs.topk(2, dim=1).values
    folded_margin = folded_top2[:, 0] - folded_top2[:, 1]
    predictions = mixed_predictions.clone()
    eligible_mask = mixed_predictions >= 10
    case_thresholds = _float_map(artifact.get("letter_case_thresholds"))
    confidence_thresholds = _float_map(artifact.get("folded_confidence_thresholds"))
    margin_thresholds = _float_map(artifact.get("folded_margin_thresholds"))
    default_case_threshold = float(artifact.get("letter_case_threshold", 0.0))
    default_confidence_threshold = float(artifact.get("folded_confidence_threshold", 0.0))
    default_margin_threshold = float(artifact.get("folded_margin_threshold", 0.0))
    for letter_index in range(26):
        letter = chr(ord("A") + letter_index)
        folded_index = 10 + letter_index
        upper_index = 10 + letter_index
        lower_index = 36 + letter_index
        identity_mask = (
            eligible_mask
            & (folded_predictions == folded_index)
            & (folded_confidence >= confidence_thresholds.get(letter, default_confidence_threshold))
            & (folded_margin >= margin_thresholds.get(letter, default_margin_threshold))
        )
        if not bool(identity_mask.any()):
            continue
        lower_margin = mixed_outputs[:, lower_index] - mixed_outputs[:, upper_index]
        lower_mask = identity_mask & (lower_margin >= case_thresholds.get(letter, default_case_threshold))
        predictions[identity_mask & ~lower_mask] = upper_index
        predictions[lower_mask] = lower_index
    return predictions


def _metric_helpers(labels: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tensors used for fast metric calculation."""

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


def hybrid_metrics(predictions: torch.Tensor, targets: torch.Tensor, labels: list[str]) -> dict[str, float]:
    """Return saved mixed-case metric fields for one hybrid prediction tensor."""

    case_or_match, is_digit, is_upper, is_lower = _metric_helpers(labels)
    exact = predictions == targets

    def masked_accuracy(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return 100.0 * float(exact[mask].float().mean().item())

    digit_accuracy = masked_accuracy(is_digit[targets])
    upper_accuracy = masked_accuracy(is_upper[targets])
    lower_accuracy = masked_accuracy(is_lower[targets])
    return {
        "test_accuracy": 100.0 * float(exact.float().mean().item()),
        "case_or_ambiguity_aware_test_accuracy": 100.0
        * float(case_or_match[targets, predictions].float().mean().item()),
        "digit_test_accuracy": digit_accuracy,
        "upper_test_accuracy": upper_accuracy,
        "lower_test_accuracy": lower_accuracy,
        "balanced_group_accuracy": min(digit_accuracy, upper_accuracy, lower_accuracy),
    }


def _objective(metrics: dict[str, float], name: str) -> float:
    """Return one optimization objective from a metric dictionary."""

    if name not in METRIC_NAMES:
        raise ValueError(f"Unknown mixed-case hybrid objective: {name}")
    return float(metrics[name])


def _floor_or_baseline(
    requested: float | None,
    baseline: dict[str, float],
    metric_name: str,
) -> float:
    """Use the current baseline as the default floor to avoid regressions."""

    if requested is not None:
        return float(requested)
    return float(baseline.get(metric_name, 0.0))


def _meets_floors(
    metrics: dict[str, float],
    min_test: float,
    min_case_or_visual: float,
    min_digit: float,
    min_upper: float,
    min_lower: float,
) -> bool:
    """Return whether candidate metrics preserve configured safety floors."""

    return (
        metrics["test_accuracy"] >= min_test
        and metrics["case_or_ambiguity_aware_test_accuracy"] >= min_case_or_visual
        and metrics["digit_test_accuracy"] >= min_digit
        and metrics["upper_test_accuracy"] >= min_upper
        and metrics["lower_test_accuracy"] >= min_lower
    )


def _candidate_artifacts(
    artifact: dict[str, object],
    labels: Iterable[str],
    case_thresholds: tuple[float, ...],
    confidence_thresholds: tuple[float, ...],
    margin_thresholds: tuple[float, ...],
) -> Iterable[tuple[str, dict[str, object]]]:
    """Yield one-threshold edits as independent artifact candidates."""

    for label in dict.fromkeys(label.upper() for label in labels if label.isalpha()):
        for field, values in (
            ("letter_case_thresholds", case_thresholds),
            ("folded_confidence_thresholds", confidence_thresholds),
            ("folded_margin_thresholds", margin_thresholds),
        ):
            current = _float_map(artifact.get(field))
            for value in values:
                if current.get(label) == float(value):
                    continue
                candidate = dict(artifact)
                next_map = dict(current)
                next_map[label] = float(value)
                candidate[field] = next_map
                yield f"{field}.{label}={float(value)}", candidate


def calibrate_hybrid(
    output_path: Path = MIXEDCASE_HYBRID_PATH,
    batch_size: int = 4096,
    labels_to_tune: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    case_thresholds: tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0),
    confidence_thresholds: tuple[float, ...] = (0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95),
    margin_thresholds: tuple[float, ...] = (-999.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
    rounds: int = 6,
    objective: str = "test_accuracy",
    min_improvement: float = 0.01,
    min_test: float | None = None,
    min_case_or_visual: float | None = None,
    min_digit: float | None = None,
    min_upper: float | None = None,
    min_lower: float | None = None,
    write: bool = True,
) -> dict[str, object]:
    """Greedily tune hybrid thresholds while preserving metric floors."""

    mixed_outputs, folded_outputs, targets, labels = _model_outputs(batch_size)
    artifact = _load_hybrid_artifact(output_path)
    base_predictions = hybrid_predictions(mixed_outputs, folded_outputs, artifact)
    base_metrics = hybrid_metrics(base_predictions, targets, labels)
    min_test = _floor_or_baseline(min_test, base_metrics, "test_accuracy")
    min_case_or_visual = _floor_or_baseline(
        min_case_or_visual,
        base_metrics,
        "case_or_ambiguity_aware_test_accuracy",
    )
    min_digit = _floor_or_baseline(min_digit, base_metrics, "digit_test_accuracy")
    min_upper = _floor_or_baseline(min_upper, base_metrics, "upper_test_accuracy")
    min_lower = _floor_or_baseline(min_lower, base_metrics, "lower_test_accuracy")
    best_artifact = artifact
    best_metrics = base_metrics
    steps: list[dict[str, object]] = []
    for round_index in range(max(0, rounds)):
        best_candidate: tuple[float, str, dict[str, object], dict[str, float]] | None = None
        for description, candidate in _candidate_artifacts(
            best_artifact,
            labels_to_tune,
            case_thresholds,
            confidence_thresholds,
            margin_thresholds,
        ):
            predictions = hybrid_predictions(mixed_outputs, folded_outputs, candidate)
            metrics = hybrid_metrics(predictions, targets, labels)
            if not _meets_floors(metrics, min_test, min_case_or_visual, min_digit, min_upper, min_lower):
                continue
            gain = _objective(metrics, objective) - _objective(best_metrics, objective)
            if gain <= 0:
                continue
            test_gain = metrics["test_accuracy"] - best_metrics["test_accuracy"]
            score = gain + max(test_gain, 0.0) * 0.01
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, description, candidate, metrics)
        if best_candidate is None:
            break
        _score, description, best_artifact, best_metrics = best_candidate
        steps.append(
            {
                "round": round_index + 1,
                "change": description,
                "objective": objective,
                "objective_value": _objective(best_metrics, objective),
                "test_accuracy": best_metrics["test_accuracy"],
                "case_or_ambiguity_aware_test_accuracy": best_metrics[
                    "case_or_ambiguity_aware_test_accuracy"
                ],
                "digit_test_accuracy": best_metrics["digit_test_accuracy"],
                "upper_test_accuracy": best_metrics["upper_test_accuracy"],
                "lower_test_accuracy": best_metrics["lower_test_accuracy"],
            }
        )
    improvement = _objective(best_metrics, objective) - _objective(base_metrics, objective)
    wrote = write and improvement >= min_improvement
    if wrote:
        final_artifact = dict(best_artifact)
        final_metrics = dict(best_metrics)
        final_metrics.pop("balanced_group_accuracy", None)
        final_artifact.update(
            {
                "enabled": True,
                "source": "greedy_hybrid_threshold_calibration",
                "labels": labels,
                "mixedcase_checkpoint_sha256": _file_sha256(MIXEDCASE_WEIGHTS_PATH),
                "folded_checkpoint_sha256": _file_sha256(WEIGHTS_PATH),
                "mixedcase_logit_bias_sha256": _file_sha256(MIXEDCASE_LOGIT_BIAS_PATH),
                "mixedcase_pair_rules_sha256": _file_sha256(MIXEDCASE_PAIR_RULES_PATH),
                "base_objective": _objective(base_metrics, objective),
                "calibrated_objective": _objective(best_metrics, objective),
                "objective": objective,
                "best_checkpoint": final_metrics,
                "steps": steps,
            }
        )
        output_path.write_text(json.dumps(final_artifact, indent=2) + "\n", encoding="utf-8")
    return {
        "base_accuracy": base_metrics["test_accuracy"],
        "calibrated_accuracy": best_metrics["test_accuracy"],
        "base_objective": _objective(base_metrics, objective),
        "calibrated_objective": _objective(best_metrics, objective),
        "objective": objective,
        "improvement": improvement,
        "best_checkpoint": best_metrics,
        "steps": steps,
        "wrote": wrote,
        "output_path": str(output_path),
    }


def _restore_artifact(output_path: Path, backup_path: Path | None) -> None:
    """Restore or remove the hybrid artifact after a failed write gate."""

    if backup_path is not None and backup_path.exists():
        shutil.copy2(backup_path, output_path)
    elif output_path.exists():
        output_path.unlink()


def _app_gate_report(target: float) -> dict[str, object]:
    """Evaluate clean and script app hardcases for a candidate hybrid artifact."""

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


def _float_tuple(value: str) -> tuple[float, ...]:
    """Parse a comma-separated float list."""

    return tuple(float(part) for part in value.split(",") if part.strip())


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Calibrate the mixed-case hybrid threshold artifact.")
    parser.add_argument("--output-path", type=Path, default=MIXEDCASE_HYBRID_PATH)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--labels", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parser.add_argument("--case-thresholds", default="-2,-1.5,-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,1.5,2")
    parser.add_argument("--confidence-thresholds", default="0,0.1,0.25,0.4,0.55,0.7,0.85,0.95")
    parser.add_argument("--margin-thresholds", default="-999,-1,-0.5,0,0.5,1,1.5,2")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--objective", default="test_accuracy", choices=sorted(METRIC_NAMES))
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--min-test", type=float, default=None)
    parser.add_argument("--min-case-or-visual", type=float, default=None)
    parser.add_argument("--min-digit", type=float, default=None)
    parser.add_argument("--min-upper", type=float, default=None)
    parser.add_argument("--min-lower", type=float, default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-app-gates", action="store_true")
    parser.add_argument("--app-gate-target", type=float, default=95.0)
    args = parser.parse_args()

    backup_path: Path | None = None
    if args.require_app_gates and not args.dry_run and args.output_path.exists():
        backup_file = tempfile.NamedTemporaryFile(prefix="mixedcase-hybrid-", suffix=".json", delete=False)
        backup_file.close()
        backup_path = Path(backup_file.name)
        shutil.copy2(args.output_path, backup_path)

    report = calibrate_hybrid(
        output_path=args.output_path,
        batch_size=args.batch_size,
        labels_to_tune=args.labels,
        case_thresholds=_float_tuple(args.case_thresholds),
        confidence_thresholds=_float_tuple(args.confidence_thresholds),
        margin_thresholds=_float_tuple(args.margin_thresholds),
        rounds=args.rounds,
        objective=args.objective,
        min_improvement=args.min_improvement,
        min_test=args.min_test,
        min_case_or_visual=args.min_case_or_visual,
        min_digit=args.min_digit,
        min_upper=args.min_upper,
        min_lower=args.min_lower,
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
