"""Probe mixed-case checkpoint logit ensembles without deploying them."""

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

from alnum_model import (  # noqa: E402
    MIXEDCASE_HYBRID_PATH,
    MIXEDCASE_LABELS,
    MIXEDCASE_LOGIT_BIAS_PATH,
    MIXEDCASE_PAIR_RULES_PATH,
    MIXEDCASE_WEIGHTS_PATH,
    MODEL_CLASSES,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    load_alnum_model,
)
from mnist_model import get_device  # noqa: E402
from scripts.calibrate_mixedcase_logits import (  # noqa: E402
    _load_existing_pair_rules,
    _metrics,
)
from scripts.calibrate_mixedcase_hybrid import _load_hybrid_artifact, hybrid_predictions  # noqa: E402


DEFAULT_SEARCH_ROOTS = (
    PROJECT_DIR / ".automation_backups",
    PROJECT_DIR / ".training_backups",
    PROJECT_DIR / "backups",
    PROJECT_DIR / "tmp" / "daily_training_backups",
)


class AverageLogitModel(nn.Module):
    """Average logits from several compatible mixed-case checkpoints."""

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


def discover_checkpoint_paths(search_roots: tuple[Path, ...] = DEFAULT_SEARCH_ROOTS) -> tuple[list[Path], int]:
    """Return unique backup mixed-case checkpoints plus duplicate count."""

    return discover_checkpoint_paths_for(MIXEDCASE_WEIGHTS_PATH, search_roots)


def discover_checkpoint_paths_for(deployed_path: Path, search_roots: tuple[Path, ...]) -> tuple[list[Path], int]:
    """Return unique checkpoint paths starting from an explicit deployed path."""

    paths = [deployed_path]
    for root in search_roots:
        if not root.exists():
            continue
        paths.extend(sorted(path for path in root.rglob("mixedcase_cnn.pt") if path.is_file()))
    unique: list[Path] = []
    seen: set[Path] = set()
    seen_hashes: set[str] = set()
    duplicate_hashes = 0
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        digest = file_sha256(path)
        if digest is not None and digest in seen_hashes:
            duplicate_hashes += 1
            continue
        if digest is not None:
            seen_hashes.add(digest)
        unique.append(path)
    return unique, duplicate_hashes


def load_raw_checkpoint(path: Path, device: torch.device) -> nn.Module | None:
    """Load one raw mixed-case checkpoint without calibration artifacts."""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if list(checkpoint.get("labels", [])) != list(MIXEDCASE_LABELS):
        return None
    model_type = str(checkpoint.get("model_type", "cnn"))
    model_class = MODEL_CLASSES.get(model_type)
    if model_class is None:
        return None
    model = model_class(num_classes=len(MIXEDCASE_LABELS)).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError):
        return None
    model.eval()
    return model


def test_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the same mixed-case test tensors used by the saved benchmark."""

    mnist_images, mnist_targets = build_or_load_mnist_cache(train=False)
    byclass_images, byclass_targets = build_or_load_emnist_byclass_mixedcase_cache(train=False)
    return torch.cat([mnist_images, byclass_images]), torch.cat([mnist_targets, byclass_targets])


def hybrid_stack_metrics(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
    apply_calibration: bool = True,
    hybrid_artifact_path: Path = MIXEDCASE_HYBRID_PATH,
) -> dict[str, float]:
    """Evaluate mixed logits after the deployed folded-hybrid decision layer."""

    folded_model, folded_labels = load_alnum_model(device=device)
    folded_expected = [str(index) for index in range(10)] + [chr(ord("A") + index) for index in range(26)]
    if folded_model is None or list(folded_labels or []) != folded_expected:
        raise RuntimeError("Folded alnum model is required for hybrid ensemble probing.")
    artifact = _load_hybrid_artifact(hybrid_artifact_path)
    bias = load_current_logit_bias(device) if apply_calibration else None
    pair_rules = _load_existing_pair_rules(MIXEDCASE_PAIR_RULES_PATH, list(MIXEDCASE_LABELS)) if apply_calibration else []
    loader = DataLoader(TensorDataset(images, targets), batch_size=batch_size, shuffle=False)
    predictions: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_images, batch_targets in loader:
            inputs = batch_images.to(device)
            mixed_outputs = model(inputs).cpu()
            if bias is not None:
                mixed_outputs = mixed_outputs + bias.cpu()
            mixed_outputs = apply_pair_rules_to_scores(mixed_outputs, list(MIXEDCASE_LABELS), pair_rules)
            folded_outputs = folded_model(inputs).cpu()
            batch_predictions = hybrid_predictions(mixed_outputs, folded_outputs, artifact)
            predictions.append(batch_predictions)
            target_parts.append(batch_targets.cpu())
    final_predictions = torch.cat(predictions)
    final_targets = torch.cat(target_parts)
    metrics = _metrics(final_predictions, final_targets, list(MIXEDCASE_LABELS))
    metrics["balanced_group_accuracy"] = min(
        metrics["digit_test_accuracy"],
        metrics["upper_test_accuracy"],
        metrics["lower_test_accuracy"],
    )
    return metrics


def apply_pair_rules_to_scores(
    scores: torch.Tensor,
    labels: list[str],
    rules: list[dict[str, object]],
) -> torch.Tensor:
    """Apply serving-style pair-rule logit nudges before hybrid selection."""

    if not rules:
        return scores
    label_to_index = {label: index for index, label in enumerate(labels)}
    outputs = scores.clone()
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
        current = outputs.argmax(dim=1)
        margin = outputs[:, to_index] - outputs[:, from_index]
        flip_mask = (current == from_index) & (margin >= threshold)
        if bool(flip_mask.any()):
            outputs[flip_mask, to_index] = outputs[flip_mask, from_index] + 1e-4
    return outputs


def load_current_logit_bias(device: torch.device) -> torch.Tensor | None:
    """Return the current deployed mixed-case logit bias when it matches labels."""

    if not MIXEDCASE_LOGIT_BIAS_PATH.exists():
        return None
    try:
        artifact = torch.load(MIXEDCASE_LOGIT_BIAS_PATH, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not isinstance(artifact, dict) or list(artifact.get("labels", [])) != list(MIXEDCASE_LABELS):
        return None
    bias = artifact.get("bias")
    if not isinstance(bias, torch.Tensor) or bias.numel() != len(MIXEDCASE_LABELS):
        return None
    return bias.reshape(1, -1).to(device)


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    """Return candidate-minus-baseline deltas for shared metrics."""

    return {key: float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in baseline}


def meets_floors(metrics: dict[str, float], floors: dict[str, float]) -> bool:
    """Return whether all configured non-regression floors are preserved."""

    return all(float(metrics.get(key, 0.0)) >= floor for key, floor in floors.items())


def run_probe(candidate_limit: int, batch_size: int, min_delta: float) -> dict[str, object]:
    """Evaluate single-checkpoint and two-checkpoint hybrid ensembles."""

    device = get_device()
    images, targets = test_tensors()
    current_raw = load_raw_checkpoint(MIXEDCASE_WEIGHTS_PATH, device)
    if current_raw is None:
        raise RuntimeError("Could not load current raw mixed-case checkpoint.")
    baseline = hybrid_stack_metrics(current_raw, images, targets, device, batch_size)
    floors = {
        "case_or_ambiguity_aware_test_accuracy": baseline["case_or_ambiguity_aware_test_accuracy"],
        "digit_test_accuracy": baseline["digit_test_accuracy"],
        "upper_test_accuracy": baseline["upper_test_accuracy"],
        "lower_test_accuracy": baseline["lower_test_accuracy"],
    }
    reports: list[dict[str, object]] = []
    best: dict[str, object] = {"path": str(MIXEDCASE_WEIGHTS_PATH), "metrics": baseline, "delta": metric_delta(baseline, baseline)}
    unique_paths, duplicate_hashes = discover_checkpoint_paths()
    paths = unique_paths[1 : 1 + max(0, candidate_limit)]
    for path in paths:
        candidate_model = load_raw_checkpoint(path, device)
        if candidate_model is None:
            continue
        single_metrics = hybrid_stack_metrics(candidate_model, images, targets, device, batch_size)
        ensemble_model = AverageLogitModel([current_raw, candidate_model]).to(device).eval()
        ensemble_metrics = hybrid_stack_metrics(ensemble_model, images, targets, device, batch_size)
        report = {
            "path": str(path),
            "single_metrics": single_metrics,
            "single_delta": metric_delta(single_metrics, baseline),
            "ensemble_metrics": ensemble_metrics,
            "ensemble_delta": metric_delta(ensemble_metrics, baseline),
            "accepted": (
                ensemble_metrics["test_accuracy"] >= baseline["test_accuracy"] + min_delta
                and meets_floors(ensemble_metrics, floors)
            ),
        }
        reports.append(report)
        if report["accepted"] and ensemble_metrics["test_accuracy"] > dict(best["metrics"])["test_accuracy"]:
            best = {
                "path": str(path),
                "metrics": ensemble_metrics,
                "delta": metric_delta(ensemble_metrics, baseline),
            }
    return {
        "baseline": baseline,
        "floors": floors,
        "best": best,
        "candidate_count": len(reports),
        "unique_checkpoint_count": len(unique_paths),
        "duplicate_checkpoint_count": duplicate_hashes,
        "candidates": reports,
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Probe mixed-case checkpoint logit ensembles.")
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-delta", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.candidate_limit, args.batch_size, args.min_delta), indent=2))


if __name__ == "__main__":
    main()
