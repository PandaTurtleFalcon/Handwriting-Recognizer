"""Summarize the recognizer's saved benchmark gates against a target."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _read_json(path: Path) -> Any:
    """Read a JSON object, returning an empty dict when it is absent."""

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _gate(name: str, value: float | None, target: float) -> dict[str, object]:
    """Create one pass/fail benchmark row."""

    return {
        "name": name,
        "value": value,
        "target": target,
        "passed": value is not None and value >= target,
    }


def _counted_gate(
    name: str,
    value: float | None,
    target: float,
    correct: object,
    total: object,
) -> dict[str, object]:
    """Create a benchmark row that also carries numerator/denominator counts."""

    row = _gate(name, value, target)
    row["correct"] = int(correct)
    row["total"] = int(total)
    return row


def summarize_saved_metrics(project_dir: Path = PROJECT_DIR, target: float = 95.0) -> list[dict[str, object]]:
    """Return saved model-metric gates for the current checkpoints."""

    digit_metrics = _read_json(project_dir / "training_metrics.json")
    folded_metrics = _read_json(project_dir / "alnum_training_metrics.json")
    mixed_metrics = _read_json(project_dir / "mixedcase_training_metrics.json")
    character_metrics = _read_json(project_dir / "character_training_metrics.json")
    mixed_calibration = _read_mixedcase_calibration(project_dir)
    mixed_pair_rules = _read_mixedcase_pair_rules(project_dir)
    mixed_hybrid = _read_mixedcase_hybrid(project_dir)
    mixed_family_reranker = _read_mixedcase_family_reranker(project_dir)
    character_calibration = _read_character_calibration(project_dir)
    character_pair_rules = _read_character_pair_rules(project_dir)

    digit_best = _best_checkpoint(digit_metrics)
    folded_best = folded_metrics.get("best_checkpoint", {})
    mixed_best = mixed_metrics.get("best_checkpoint", {})
    character_best = character_metrics.get("best_checkpoint", {})
    if mixed_calibration is not None:
        mixed_best = mixed_calibration
    if mixed_pair_rules is not None:
        mixed_best = mixed_pair_rules
    if mixed_calibration is not None and _mixedcase_calibration_includes_current_pair_rules(project_dir):
        mixed_best = mixed_calibration
    if mixed_hybrid is not None:
        mixed_best = mixed_hybrid
    if mixed_family_reranker is not None:
        mixed_best = mixed_family_reranker
    if character_calibration is not None:
        character_best = character_calibration
    if character_pair_rules is not None:
        character_best = character_pair_rules
    if character_calibration is not None and _character_calibration_includes_current_pair_rules(project_dir):
        character_best = character_calibration

    return [
        _gate("digit_specialist_exact", _float_or_none(digit_best.get("test_accuracy")), target),
        _gate("folded_alnum_exact", _float_or_none(folded_best.get("test_accuracy")), target),
        _gate("mixedcase_exact", _float_or_none(mixed_best.get("test_accuracy")), target),
        _gate("mixedcase_case_or_visual", _float_or_none(mixed_best.get("case_or_ambiguity_aware_test_accuracy")), target),
        _gate("mixedcase_digit_exact", _float_or_none(mixed_best.get("digit_test_accuracy")), target),
        _gate("mixedcase_upper_exact", _float_or_none(mixed_best.get("upper_test_accuracy")), target),
        _gate("mixedcase_lower_exact", _float_or_none(mixed_best.get("lower_test_accuracy")), target),
        _gate("character_exact", _float_or_none(character_best.get("validation_accuracy")), target),
        _gate("character_ambiguity", _float_or_none(character_best.get("ambiguity_aware_validation_accuracy")), target),
        _gate("character_digit_exact", _float_or_none(character_best.get("digit_validation_accuracy")), target),
        _gate("character_letter_exact", _float_or_none(character_best.get("letter_validation_accuracy")), target),
        _gate("punctuation_exact", _float_or_none(character_best.get("punctuation_validation_accuracy")), target),
        _gate(
            "punctuation_ambiguity",
            _float_or_none(character_best.get("punctuation_ambiguity_aware_validation_accuracy")),
            target,
        ),
    ]


def _read_mixedcase_calibration(project_dir: Path) -> dict[str, object] | None:
    """Return calibrated mixed-case metrics when the optional artifact matches."""

    import torch

    bias_path = project_dir / "mixedcase_logit_bias.pt"
    if not bias_path.exists():
        return None
    try:
        from alnum_model import MIXEDCASE_LABELS

        calibration = torch.load(bias_path, map_location="cpu", weights_only=True)
    except (ImportError, OSError, RuntimeError, ValueError, pickle.UnpicklingError):
        return None
    if list(calibration.get("labels", [])) != list(MIXEDCASE_LABELS):
        return None
    if not _artifact_matches_checkpoint(calibration, project_dir / "mixedcase_cnn.pt"):
        return None
    best = calibration.get("best_checkpoint")
    if not isinstance(best, dict):
        return None
    if calibration.get("includes_pair_rules") and not _mixedcase_pair_rule_dependency_current(calibration, project_dir):
        return None
    return best


def _read_mixedcase_pair_rules(project_dir: Path) -> dict[str, object] | None:
    """Return mixed-case pair-rule metrics when the optional artifact matches."""

    rules_path = project_dir / "mixedcase_pair_rules.json"
    if not rules_path.exists():
        return None
    try:
        from alnum_model import MIXEDCASE_LABELS
    except ImportError:
        return None
    artifact = _read_json(rules_path)
    if not isinstance(artifact, dict) or list(artifact.get("labels", [])) != list(MIXEDCASE_LABELS):
        return None
    if not _artifact_matches_checkpoint(artifact, project_dir / "mixedcase_cnn.pt"):
        return None
    best = artifact.get("best_checkpoint")
    return best if isinstance(best, dict) else None


def _read_mixedcase_hybrid(project_dir: Path) -> dict[str, object] | None:
    """Return mixed-case hybrid metrics when both checkpoint hashes match."""

    hybrid_path = project_dir / "mixedcase_hybrid.json"
    if not hybrid_path.exists():
        return None
    try:
        from alnum_model import MIXEDCASE_LABELS
    except ImportError:
        return None
    artifact = _read_json(hybrid_path)
    if not isinstance(artifact, dict) or not artifact.get("enabled", True):
        return None
    if list(artifact.get("labels", [])) != list(MIXEDCASE_LABELS):
        return None
    if not _artifact_hash_matches(artifact, "mixedcase_checkpoint_sha256", project_dir / "mixedcase_cnn.pt"):
        return None
    if not _artifact_hash_matches(artifact, "folded_checkpoint_sha256", project_dir / "alnum_cnn.pt"):
        return None
    if not _artifact_dependency_hash_matches(
        artifact,
        "mixedcase_logit_bias_sha256",
        project_dir / "mixedcase_logit_bias.pt",
    ):
        return None
    if not _artifact_dependency_hash_matches(
        artifact,
        "mixedcase_pair_rules_sha256",
        project_dir / "mixedcase_pair_rules.json",
    ):
        return None
    best = artifact.get("best_checkpoint")
    return best if isinstance(best, dict) else None


def _read_mixedcase_family_reranker(project_dir: Path) -> dict[str, object] | None:
    """Return mixed-case family-reranker metrics when dependency hashes match."""

    import torch

    artifact_path = project_dir / "mixedcase_family_reranker.pt"
    if not artifact_path.exists():
        return None
    try:
        from alnum_model import MIXEDCASE_LABELS

        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError):
        return None
    if not isinstance(artifact, dict) or not artifact.get("enabled", True):
        return None
    if list(artifact.get("labels", [])) != list(MIXEDCASE_LABELS):
        return None
    if not _artifact_hash_matches(artifact, "mixedcase_checkpoint_sha256", project_dir / "mixedcase_cnn.pt"):
        return None
    if not _artifact_hash_matches(artifact, "folded_checkpoint_sha256", project_dir / "alnum_cnn.pt"):
        return None
    if not _artifact_dependency_hash_matches(
        artifact,
        "mixedcase_logit_bias_sha256",
        project_dir / "mixedcase_logit_bias.pt",
    ):
        return None
    if not _artifact_dependency_hash_matches(
        artifact,
        "mixedcase_pair_rules_sha256",
        project_dir / "mixedcase_pair_rules.json",
    ):
        return None
    if not _artifact_dependency_hash_matches(
        artifact,
        "mixedcase_hybrid_sha256",
        project_dir / "mixedcase_hybrid.json",
    ):
        return None
    best = artifact.get("best_checkpoint")
    return best if isinstance(best, dict) else None


def _mixedcase_calibration_includes_current_pair_rules(project_dir: Path) -> bool:
    """Return whether mixed-case bias metrics already include current pair rules."""

    bias_path = project_dir / "mixedcase_logit_bias.pt"
    if not bias_path.exists():
        return False
    try:
        import torch

        calibration = torch.load(bias_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError):
        return False
    if not isinstance(calibration, dict) or not calibration.get("includes_pair_rules"):
        return False
    return _mixedcase_pair_rule_dependency_current(calibration, project_dir)


def _mixedcase_pair_rule_dependency_current(calibration: dict[str, object], project_dir: Path) -> bool:
    """Return whether a mixed-case calibration is tied to valid current pair rules."""

    if _read_mixedcase_pair_rules(project_dir) is None:
        return False
    return _artifact_dependency_hash_matches(calibration, "pair_rules_sha256", project_dir / "mixedcase_pair_rules.json")


def _read_character_calibration(project_dir: Path) -> dict[str, object] | None:
    """Return calibrated character metrics when the optional artifact matches."""

    import torch

    bias_path = project_dir / "character_logit_bias.pt"
    labels_path = project_dir / "character_labels.json"
    if not bias_path.exists() or not labels_path.exists():
        return None
    labels = _read_json(labels_path)
    if not isinstance(labels, list):
        return None
    try:
        calibration = torch.load(bias_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError):
        return None
    if list(calibration.get("labels", [])) != list(labels):
        return None
    if not _artifact_matches_checkpoint(calibration, project_dir / "character_cnn.pt"):
        return None
    best = calibration.get("best_checkpoint")
    if not isinstance(best, dict):
        return None
    if calibration.get("includes_pair_rules") and not _character_pair_rule_dependency_current(calibration, project_dir):
        return None
    return best


def _read_character_pair_rules(project_dir: Path) -> dict[str, object] | None:
    """Return character pair-rule metrics when the optional artifact matches."""

    rules_path = project_dir / "character_pair_rules.json"
    labels_path = project_dir / "character_labels.json"
    if not rules_path.exists() or not labels_path.exists():
        return None
    labels = _read_json(labels_path)
    if not isinstance(labels, list):
        return None
    artifact = _read_json(rules_path)
    if not isinstance(artifact, dict) or list(artifact.get("labels", [])) != list(labels):
        return None
    if not _artifact_matches_checkpoint(artifact, project_dir / "character_cnn.pt"):
        return None
    best = artifact.get("best_checkpoint")
    return best if isinstance(best, dict) else None


def _character_calibration_includes_current_pair_rules(project_dir: Path) -> bool:
    """Return whether character bias metrics already include current pair rules."""

    bias_path = project_dir / "character_logit_bias.pt"
    if not bias_path.exists():
        return False
    try:
        import torch

        calibration = torch.load(bias_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError):
        return False
    if not isinstance(calibration, dict) or not calibration.get("includes_pair_rules"):
        return False
    return _character_pair_rule_dependency_current(calibration, project_dir)


def _character_pair_rule_dependency_current(calibration: dict[str, object], project_dir: Path) -> bool:
    """Return whether a character calibration is tied to valid current pair rules."""

    if _read_character_pair_rules(project_dir) is None:
        return False
    return _artifact_dependency_hash_matches(calibration, "pair_rules_sha256", project_dir / "character_pair_rules.json")


def _artifact_matches_checkpoint(artifact: object, weights_path: Path) -> bool:
    """Return whether a fingerprinted calibration artifact matches weights."""

    return _artifact_hash_matches(artifact, "checkpoint_sha256", weights_path)


def _artifact_hash_matches(artifact: object, key: str, weights_path: Path) -> bool:
    """Return whether one named artifact digest matches a weights file."""

    if not isinstance(artifact, dict):
        return False
    expected = artifact.get(key)
    if not expected:
        return False
    return expected == _file_sha256(weights_path)


def _artifact_dependency_hash_matches(artifact: object, key: str, dependency_path: Path) -> bool:
    """Require matching dependency hashes when optional artifacts exist."""

    if not dependency_path.exists():
        return not isinstance(artifact, dict) or not artifact.get(key)
    if not isinstance(artifact, dict) or not artifact.get(key):
        return False
    return artifact.get(key) == _file_sha256(dependency_path)


def _file_sha256(path: Path) -> str | None:
    """Return a stable digest for checkpoint freshness checks."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def summarize_app_hardcases(
    target: float = 95.0,
    all_fonts: bool = True,
    script_cases: bool = False,
) -> list[dict[str, object]]:
    """Return app-level generated hard-case gates for the live recognizer stack."""

    from scripts.evaluate_hardcases import evaluate_cases

    report = evaluate_cases(all_fonts=all_fonts, script_cases=script_cases)
    prefix = "app_script_hardcase" if script_cases else "app_hardcase"
    return [
        _counted_gate(
            f"{prefix}_exact",
            _float_or_none(report.get("exact_accuracy")),
            target,
            report.get("exact_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            f"{prefix}_ambiguity",
            _float_or_none(report.get("ambiguity_aware_accuracy")),
            target,
            report.get("ambiguity_aware_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            f"{prefix}_raw_exact",
            _float_or_none(report.get("raw_exact_accuracy")),
            target,
            report.get("raw_exact_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            f"{prefix}_raw_ambiguity",
            _float_or_none(report.get("raw_ambiguity_aware_accuracy")),
            target,
            report.get("raw_ambiguity_aware_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            f"{prefix}_raw_label_compact_exact",
            _float_or_none(report.get("raw_label_compact_exact_accuracy")),
            target,
            report.get("raw_label_compact_exact_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            f"{prefix}_raw_label_compact_ambiguity",
            _float_or_none(report.get("raw_label_compact_ambiguity_aware_accuracy")),
            target,
            report.get("raw_label_compact_ambiguity_aware_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            f"{prefix}_non_replayed_exact",
            _float_or_none(report.get("non_replayed_exact_accuracy")),
            target,
            report.get("non_replayed_exact_correct", 0),
            report.get("non_replayed_total", 0),
        ),
    ]


def summarize_uploaded_hardcases(target: float = 95.0) -> list[dict[str, object]]:
    """Return app-level gates for saved real-upload fixtures."""

    from scripts.evaluate_hardcases import evaluate_uploaded_fixtures

    report = evaluate_uploaded_fixtures()
    return [
        _counted_gate(
            "uploaded_hardcase_exact",
            _float_or_none(report.get("exact_accuracy")),
            target,
            report.get("exact_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            "uploaded_hardcase_ambiguity",
            _float_or_none(report.get("ambiguity_aware_accuracy")),
            target,
            report.get("ambiguity_aware_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            "uploaded_hardcase_raw_exact",
            _float_or_none(report.get("raw_exact_accuracy")),
            target,
            report.get("raw_exact_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            "uploaded_hardcase_raw_ambiguity",
            _float_or_none(report.get("raw_ambiguity_aware_accuracy")),
            target,
            report.get("raw_ambiguity_aware_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            "uploaded_hardcase_raw_label_compact_exact",
            _float_or_none(report.get("raw_label_compact_exact_accuracy")),
            target,
            report.get("raw_label_compact_exact_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            "uploaded_hardcase_raw_label_compact_ambiguity",
            _float_or_none(report.get("raw_label_compact_ambiguity_aware_accuracy")),
            target,
            report.get("raw_label_compact_ambiguity_aware_correct", 0),
            report.get("total", 0),
        ),
        _counted_gate(
            "uploaded_hardcase_non_replayed_exact",
            _float_or_none(report.get("non_replayed_exact_accuracy")),
            target,
            report.get("non_replayed_exact_correct", 0),
            report.get("non_replayed_total", 0),
        ),
    ]


def summarize_correction_memory(target: float = 95.0, project_dir: Path = PROJECT_DIR) -> list[dict[str, object]]:
    """Return deployed character correction-memory coverage for priority labels."""

    from character_model import load_correction_memory_exemplars
    from main import CHARACTER_PRACTICE_PRIORITY_LABELS, PRACTICE_TARGET_PER_LABEL

    labels = _read_json(project_dir / "character_labels.json")
    if not isinstance(labels, list):
        return [
            _counted_gate("character_correction_memory_samples", None, target, 0, 0),
            _counted_gate("character_correction_memory_ready_labels", None, target, 0, 0),
        ]
    loaded = load_correction_memory_exemplars([str(label) for label in labels])
    counts: dict[str, int] = {}
    if loaded is not None:
        _images, targets = loaded
        for target_index in targets.tolist():
            label = str(labels[int(target_index)])
            counts[label] = counts.get(label, 0) + 1

    priority_labels = list(dict.fromkeys(CHARACTER_PRACTICE_PRIORITY_LABELS))
    target_samples = len(priority_labels) * PRACTICE_TARGET_PER_LABEL
    samples = sum(counts.get(label, 0) for label in priority_labels)
    ready_labels = sum(1 for label in priority_labels if counts.get(label, 0) >= PRACTICE_TARGET_PER_LABEL)
    not_ready_labels = [label for label in priority_labels if counts.get(label, 0) < PRACTICE_TARGET_PER_LABEL]
    sample_percent = 100.0 * samples / target_samples if target_samples else 100.0
    ready_percent = 100.0 * ready_labels / len(priority_labels) if priority_labels else 100.0
    metadata = {
        "by_label": {label: counts[label] for label in priority_labels if label in counts},
        "priority_labels": priority_labels,
        "not_ready_label_list": not_ready_labels,
        "not_ready_label_count": len(not_ready_labels),
        "samples_per_label_target": PRACTICE_TARGET_PER_LABEL,
    }
    rows = [
        _counted_gate("character_correction_memory_samples", sample_percent, target, samples, target_samples),
        _counted_gate("character_correction_memory_ready_labels", ready_percent, target, ready_labels, len(priority_labels)),
    ]
    return [{**row, **metadata} for row in rows]


def _correction_training_rows(
    prefix: str,
    counts: dict[str, int],
    priority_labels: str,
    target: float,
    target_per_label: int,
) -> list[dict[str, object]]:
    """Return flat benchmark rows for queued correction-training coverage."""

    from scripts.train_from_corrections import correction_readiness_summary, not_ready_label_list

    readiness = correction_readiness_summary(Counter(counts), priority_labels, target_per_label=target_per_label)
    label_list = list(dict.fromkeys(priority_labels))
    not_ready = not_ready_label_list(Counter(counts), priority_labels, target_per_label=target_per_label)
    metadata = {
        "by_label": {label: int(counts[label]) for label in label_list if int(counts.get(label, 0)) > 0},
        "priority_labels": label_list,
        "not_ready_label_list": not_ready,
        "not_ready_label_count": len(not_ready),
        "samples_per_label_target": target_per_label,
    }
    rows = [
        _counted_gate(
            f"{prefix}_correction_training_samples",
            _float_or_none(readiness.get("coverage_percent")),
            target,
            readiness.get("samples", 0),
            readiness.get("target_samples", 0),
        ),
        _counted_gate(
            f"{prefix}_correction_training_ready_labels",
            100.0 * int(readiness.get("ready_labels", 0)) / max(int(readiness.get("total_labels", 0)), 1),
            target,
            readiness.get("ready_labels", 0),
            readiness.get("total_labels", 0),
        ),
    ]
    return [{**row, **metadata} for row in rows]


def summarize_correction_training(target: float = 95.0) -> list[dict[str, object]]:
    """Return queued correction-training coverage for folded and mixed-case models."""

    from scripts.train_from_corrections import (
        DEFAULT_MIXEDCASE_PRIORITY_LABELS,
        DEFAULT_PRIORITY_LABELS,
        LABELS,
        MIXEDCASE_LABELS,
        correction_item_label_counts,
        filter_priority_labels,
        load_correction_cache,
    )
    from main import PRACTICE_TARGET_PER_LABEL

    folded_corrections = load_correction_cache(LABELS)
    mixed_corrections = load_correction_cache(list(MIXEDCASE_LABELS))
    folded_counts = correction_item_label_counts(LABELS, folded_corrections)
    mixed_counts = correction_item_label_counts(list(MIXEDCASE_LABELS), mixed_corrections)
    folded_priority_labels = filter_priority_labels(DEFAULT_PRIORITY_LABELS.upper(), LABELS)
    mixed_priority_labels = filter_priority_labels(DEFAULT_MIXEDCASE_PRIORITY_LABELS, list(MIXEDCASE_LABELS))
    rows = _correction_training_rows(
        "folded_alnum",
        dict(folded_counts),
        folded_priority_labels,
        target,
        PRACTICE_TARGET_PER_LABEL,
    )
    rows.extend(
        _correction_training_rows(
            "mixedcase",
            dict(mixed_counts),
            mixed_priority_labels,
            target,
            PRACTICE_TARGET_PER_LABEL,
        )
    )
    return rows


def _float_or_none(value: object) -> float | None:
    """Convert metric values to float when possible."""

    if value is None:
        return None
    return float(value)


def _best_checkpoint(metrics: Any) -> dict[str, Any]:
    """Return best-checkpoint data from either modern or legacy metric files."""

    if isinstance(metrics, dict):
        checkpoint = metrics.get("best_checkpoint", {})
        return checkpoint if isinstance(checkpoint, dict) else {}
    if isinstance(metrics, list):
        epoch_rows = [row for row in metrics if isinstance(row, dict) and "test_accuracy" in row]
        if not epoch_rows:
            return {}
        return max(epoch_rows, key=lambda row: float(row.get("test_accuracy", 0)))
    return {}


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Summarize saved recognizer benchmark gates.")
    parser.add_argument("--target", type=float, default=95.0)
    parser.add_argument(
        "--include-app-hardcases",
        action="store_true",
        help="Also run generated app-level hard cases through the live recognizer.",
    )
    parser.add_argument(
        "--single-font-hardcases",
        action="store_true",
        help="Use one font instead of all fonts when --include-app-hardcases is set.",
    )
    parser.add_argument(
        "--include-script-hardcases",
        action="store_true",
        help="Also run rough line-drawn hard cases through the live recognizer.",
    )
    parser.add_argument(
        "--include-uploaded-hardcases",
        action="store_true",
        help="Also run saved real-upload fixture images through the live recognizer.",
    )
    parser.add_argument(
        "--include-correction-memory",
        action="store_true",
        help="Also report usable saved-correction memory coverage for priority labels.",
    )
    parser.add_argument(
        "--include-correction-training",
        action="store_true",
        help="Also report queued correction-training coverage for folded and mixed-case priority labels.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = summarize_saved_metrics(target=args.target)
    if args.include_app_hardcases:
        report.extend(summarize_app_hardcases(target=args.target, all_fonts=not args.single_font_hardcases))
    if args.include_script_hardcases:
        report.extend(summarize_app_hardcases(target=args.target, all_fonts=False, script_cases=True))
    if args.include_uploaded_hardcases:
        report.extend(summarize_uploaded_hardcases(target=args.target))
    if args.include_correction_memory:
        report.extend(summarize_correction_memory(target=args.target))
    if args.include_correction_training:
        report.extend(summarize_correction_training(target=args.target))
    if args.json:
        print(json.dumps(report, indent=2))
        return
    for item in report:
        value = item["value"]
        value_text = "missing" if value is None else f"{float(value):.2f}%"
        if "correct" in item and "total" in item:
            value_text = f"{value_text} ({int(item['correct'])}/{int(item['total'])})"
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {item['name']}: {value_text} (target {float(item['target']):.2f}%)")


if __name__ == "__main__":
    main()
