"""Probe character checkpoint logit ensembles without deploying them."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from character_model import (  # noqa: E402
    CHARACTER_MODEL_TYPES,
    DATASET_ROOT,
    LOGIT_BIAS_PATH,
    PAIR_RULES_PATH,
    WEIGHTS_PATH,
    build_or_load_combined_cache,
    labels_match_with_ambiguity,
    stratified_split_indices,
)
from mnist_model import get_device  # noqa: E402
from scripts.analyze_character_confusions import _metric_extra_roots  # noqa: E402
from scripts.calibrate_character_logits import (  # noqa: E402
    _apply_pair_rules_to_predictions,
    _load_existing_bias,
    _load_existing_pair_rules,
)


DEFAULT_SEARCH_ROOTS = (
    PROJECT_DIR / ".automation_backups",
    PROJECT_DIR / ".codex_backups",
    PROJECT_DIR / ".training_backups",
    PROJECT_DIR / "backups",
    PROJECT_DIR / "tmp" / "daily_training_backups",
)
PROTECTED_METRICS = (
    "ambiguity_aware_validation_accuracy",
    "digit_validation_accuracy",
    "letter_validation_accuracy",
    "punctuation_validation_accuracy",
)


class AverageLogitModel(nn.Module):
    """Average logits from compatible character checkpoints."""

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        if not models:
            raise ValueError("At least one model is required.")
        self.models = nn.ModuleList(models)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = [model(images) for model in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)


def file_sha256(path: Path) -> str | None:
    """Return a checkpoint digest, or None when the file cannot be read."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def discover_checkpoint_paths_for(deployed_path: Path, search_roots: tuple[Path, ...]) -> tuple[list[Path], int]:
    """Return unique checkpoint paths starting from an explicit deployed path."""

    paths = [deployed_path]
    for root in search_roots:
        if not root.exists():
            continue
        paths.extend(sorted(path for path in root.rglob("character_cnn.pt") if path.is_file()))
    unique: list[Path] = []
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    duplicate_hashes = 0
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        digest = file_sha256(path)
        if digest is not None and digest in seen_hashes:
            duplicate_hashes += 1
            continue
        if digest is not None:
            seen_hashes.add(digest)
        unique.append(path)
    return unique, duplicate_hashes


def discover_checkpoint_paths(search_roots: tuple[Path, ...] = DEFAULT_SEARCH_ROOTS) -> tuple[list[Path], int]:
    """Return unique backup character checkpoints plus duplicate count."""

    return discover_checkpoint_paths_for(WEIGHTS_PATH, search_roots)


def load_raw_checkpoint(path: Path, labels: list[str] | None, device: torch.device) -> tuple[nn.Module, list[str]] | None:
    """Load one raw character checkpoint without calibration artifacts."""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError):
        return None
    checkpoint_labels = list(checkpoint.get("labels", []))
    if labels is not None and checkpoint_labels != list(labels):
        return None
    model_type = str(checkpoint.get("model_type", "mlp"))
    model_class = CHARACTER_MODEL_TYPES.get(model_type)
    if model_class is None:
        return None
    model = model_class(num_classes=len(checkpoint_labels)).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError):
        return None
    model.eval()
    return model, checkpoint_labels


def validation_tensors() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return the same character validation split used by benchmark summaries."""

    images, targets, labels = build_or_load_combined_cache(DATASET_ROOT, _metric_extra_roots())
    _, validation_indices = stratified_split_indices(
        list(range(len(targets))),
        test_size=0.15,
        random_state=42,
        stratify=targets.numpy(),
    )
    index_tensor = torch.tensor(validation_indices, dtype=torch.long)
    return images[index_tensor], targets[index_tensor], list(labels)


def _metrics(predictions: torch.Tensor, targets: torch.Tensor, labels: list[str]) -> dict[str, float]:
    """Return character exact and protected split metrics."""

    exact = predictions == targets
    ambiguity = []
    group_total = {"digit": 0, "letter": 0, "punctuation": 0}
    group_correct = {"digit": 0, "letter": 0, "punctuation": 0}
    for expected_index, predicted_index in zip(targets.tolist(), predictions.tolist()):
        expected = labels[int(expected_index)]
        predicted = labels[int(predicted_index)]
        ambiguity.append(labels_match_with_ambiguity(expected, predicted))
        group = "digit" if expected.isdigit() else "letter" if expected.isalpha() else "punctuation"
        group_total[group] += 1
        group_correct[group] += int(expected == predicted)
    return {
        "validation_accuracy": 100.0 * float(exact.float().mean().item()),
        "ambiguity_aware_validation_accuracy": 100.0 * sum(ambiguity) / max(len(ambiguity), 1),
        "digit_validation_accuracy": 100.0 * group_correct["digit"] / max(group_total["digit"], 1),
        "letter_validation_accuracy": 100.0 * group_correct["letter"] / max(group_total["letter"], 1),
        "punctuation_validation_accuracy": 100.0
        * group_correct["punctuation"]
        / max(group_total["punctuation"], 1),
    }


def calibrated_predictions(
    model: nn.Module,
    images: torch.Tensor,
    labels: list[str],
    device: torch.device,
    batch_size: int,
    apply_calibration: bool = True,
) -> torch.Tensor:
    """Run logits through current character bias and pair-rule calibration."""

    bias = _load_existing_bias(LOGIT_BIAS_PATH, labels) if apply_calibration else torch.zeros(len(labels))
    pair_rules = _load_existing_pair_rules(PAIR_RULES_PATH, labels) if apply_calibration else []
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for (batch_images,) in loader:
            scores = model(batch_images.to(device)).cpu() + bias.reshape(1, -1)
            starting = scores.argmax(dim=1)
            predictions.append(_apply_pair_rules_to_predictions(scores, starting, labels, pair_rules))
    return torch.cat(predictions)


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    """Return candidate-minus-baseline deltas for shared metrics."""

    return {key: float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in baseline}


def rejection_reason(
    baseline: dict[str, float],
    candidate: dict[str, float],
    min_delta: float,
) -> str | None:
    """Return why a candidate is rejected, or None when it is safe."""

    if candidate["validation_accuracy"] < baseline["validation_accuracy"] + min_delta:
        return "validation_delta_below_floor"
    for metric_name in PROTECTED_METRICS:
        if candidate[metric_name] < baseline[metric_name]:
            return f"{metric_name}_regressed"
    return None


def run_probe(candidate_limit: int, batch_size: int, min_delta: float) -> dict[str, object]:
    """Evaluate single-checkpoint and two-checkpoint character ensembles."""

    device = get_device()
    images, targets, labels = validation_tensors()
    loaded_current = load_raw_checkpoint(WEIGHTS_PATH, labels, device)
    if loaded_current is None:
        raise RuntimeError("Could not load current character checkpoint.")
    current_raw, labels = loaded_current
    baseline_predictions = calibrated_predictions(current_raw, images, labels, device, batch_size)
    baseline = _metrics(baseline_predictions, targets, labels)
    unique_paths, duplicate_hashes = discover_checkpoint_paths()
    reports: list[dict[str, object]] = []
    best: dict[str, object] = {"path": str(WEIGHTS_PATH), "metrics": baseline, "delta": metric_delta(baseline, baseline)}
    for path in unique_paths[1 : 1 + max(0, candidate_limit)]:
        loaded_candidate = load_raw_checkpoint(path, labels, device)
        if loaded_candidate is None:
            continue
        candidate_raw, _candidate_labels = loaded_candidate
        single_predictions = calibrated_predictions(candidate_raw, images, labels, device, batch_size)
        single_metrics = _metrics(single_predictions, targets, labels)
        ensemble_model = AverageLogitModel([current_raw, candidate_raw]).to(device).eval()
        ensemble_predictions = calibrated_predictions(ensemble_model, images, labels, device, batch_size)
        ensemble_metrics = _metrics(ensemble_predictions, targets, labels)
        reason = rejection_reason(baseline, ensemble_metrics, min_delta)
        report = {
            "path": str(path),
            "single_metrics": single_metrics,
            "single_delta": metric_delta(single_metrics, baseline),
            "ensemble_metrics": ensemble_metrics,
            "ensemble_delta": metric_delta(ensemble_metrics, baseline),
            "accepted": reason is None,
            "rejection_reason": reason,
        }
        reports.append(report)
        if reason is None and ensemble_metrics["validation_accuracy"] > dict(best["metrics"])["validation_accuracy"]:
            best = {
                "path": str(path),
                "metrics": ensemble_metrics,
                "delta": metric_delta(ensemble_metrics, baseline),
            }
    return {
        "baseline": baseline,
        "best": best,
        "reports": reports,
        "candidate_count": len(reports),
        "unique_checkpoint_count": len(unique_paths),
        "duplicate_checkpoint_count": duplicate_hashes,
        "min_delta": min_delta,
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Probe character checkpoint logit ensembles.")
    parser.add_argument("--candidate-limit", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-delta", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.candidate_limit, args.batch_size, args.min_delta), indent=2))


if __name__ == "__main__":
    main()
