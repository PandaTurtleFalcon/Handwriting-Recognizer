"""Evaluate a candidate mixed-case checkpoint without deploying it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import (  # noqa: E402
    MIXEDCASE_LABELS,
    MIXEDCASE_WEIGHTS_PATH,
    MODEL_CLASSES,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    evaluate_mixedcase_breakdown,
)
from mnist_model import get_device  # noqa: E402
from scripts.probe_mixedcase_checkpoint_ensemble import hybrid_stack_metrics  # noqa: E402


GATE_KEYS = (
    "test_accuracy",
    "case_or_ambiguity_aware_test_accuracy",
    "digit_test_accuracy",
    "upper_test_accuracy",
    "lower_test_accuracy",
)


def candidate_test_tensors(sample_limit: int | None = None, seed: int = 2026) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MNIST plus EMNIST ByClass mixed-case test tensors."""

    mnist_images, mnist_targets = build_or_load_mnist_cache(train=False)
    byclass_images, byclass_targets = build_or_load_emnist_byclass_mixedcase_cache(train=False)
    images = torch.cat([mnist_images, byclass_images])
    targets = torch.cat([mnist_targets, byclass_targets])
    if sample_limit is None or sample_limit >= int(targets.numel()):
        return images, targets
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(int(targets.numel()), generator=generator)[:sample_limit]
    return images[selected], targets[selected]


def load_candidate_checkpoint(path: Path, device: torch.device) -> nn.Module:
    """Load a candidate mixed-case checkpoint file."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if list(checkpoint.get("labels", [])) != list(MIXEDCASE_LABELS):
        raise RuntimeError(f"{path} does not use the expected mixed-case label order.")
    model_type = str(checkpoint.get("model_type", "cnn"))
    model_class = MODEL_CLASSES.get(model_type)
    if model_class is None:
        raise RuntimeError(f"{path} uses unknown mixed-case model type: {model_type}")
    model = model_class(num_classes=len(MIXEDCASE_LABELS)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_candidate(
    checkpoint_path: Path,
    batch_size: int = 4096,
    device_name: str = "auto",
    mode: str = "raw",
    sample_limit: int | None = None,
) -> dict[str, object]:
    """Evaluate one candidate checkpoint and return benchmark-style metrics."""

    if device_name == "cpu":
        device = torch.device("cpu")
    elif device_name == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        device = torch.device("mps")
    else:
        device = get_device()
    model = load_candidate_checkpoint(checkpoint_path, device)
    images, targets = candidate_test_tensors(sample_limit=sample_limit)
    if mode == "hybrid":
        metrics = hybrid_stack_metrics(model, images, targets, device, batch_size, apply_calibration=True)
    elif mode == "raw":
        loader = DataLoader(TensorDataset(images, targets), batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        metrics = evaluate_mixedcase_breakdown(model, loader, criterion, list(MIXEDCASE_LABELS), device)
        metrics["balanced_group_accuracy"] = min(
            metrics["digit_test_accuracy"],
            metrics["upper_test_accuracy"],
            metrics["lower_test_accuracy"],
        )
    else:
        raise ValueError(f"Unknown candidate evaluation mode: {mode}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "mode": mode,
        "sample_limit": sample_limit,
        "total_examples": int(targets.numel()),
        "metrics": metrics,
    }


def gate_rows(metrics: dict[str, float], target: float) -> list[dict[str, object]]:
    """Return pass/fail rows for the main mixed-case candidate gates."""

    return [
        {
            "name": key,
            "value": float(metrics.get(key, 0.0)),
            "target": target,
            "passed": float(metrics.get(key, 0.0)) >= target,
        }
        for key in GATE_KEYS
    ]


def read_baseline_metrics(path: Path | None) -> dict[str, float]:
    """Read baseline metrics from a report or raw metrics JSON file."""

    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload) if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        raise RuntimeError(f"{path} does not contain a metrics object.")
    return {
        str(key): float(value)
        for key, value in metrics.items()
        if key in GATE_KEYS and isinstance(value, (int, float))
    }


def baseline_rows(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    tolerance: float = 0.0,
) -> list[dict[str, object]]:
    """Return pass/fail rows comparing candidate metrics to a baseline."""

    rows = []
    for key in GATE_KEYS:
        if key not in baseline_metrics:
            continue
        value = float(metrics.get(key, 0.0))
        baseline = float(baseline_metrics[key])
        rows.append(
            {
                "name": key,
                "value": value,
                "baseline": baseline,
                "tolerance": tolerance,
                "passed": value + tolerance >= baseline,
            }
        )
    return rows


def failed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return rows that did not pass their gate."""

    return [row for row in rows if not bool(row.get("passed"))]


def _print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    """Print a compact table of gate rows."""

    if not rows:
        return
    print(title)
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        if "baseline" in row:
            print(
                f"{row['name']}: {row['value']:.4f}% "
                f"baseline={row['baseline']:.4f}% tolerance={row['tolerance']:.4f}% {status}"
            )
        else:
            print(f"{row['name']}: {row['value']:.4f}% target={row['target']:.2f}% {status}")


def main() -> None:
    """Run the command-line candidate evaluator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=MIXEDCASE_WEIGHTS_PATH)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--mode", choices=("raw", "hybrid"), default="raw")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--target", type=float, default=95.0)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--baseline-tolerance", type=float, default=0.0)
    parser.add_argument("--require-target", action="store_true")
    parser.add_argument("--require-baseline", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = evaluate_candidate(
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device_name=args.device,
        mode=args.mode,
        sample_limit=args.sample_limit,
    )
    target_rows = gate_rows(report["metrics"], args.target)
    baseline_metrics = read_baseline_metrics(args.baseline_json)
    comparison_rows = baseline_rows(report["metrics"], baseline_metrics, args.baseline_tolerance)
    report["gates"] = target_rows
    report["baseline_gates"] = comparison_rows
    target_failures = failed_rows(target_rows) if args.require_target else []
    baseline_failures = failed_rows(comparison_rows) if args.require_baseline else []
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"checkpoint: {report['checkpoint_path']}")
        print(f"mode: {report['mode']} examples: {report['total_examples']}")
        _print_rows("target gates:", target_rows)
        _print_rows("baseline gates:", comparison_rows)
    if target_failures or baseline_failures:
        failure_names = [str(row["name"]) for row in [*target_failures, *baseline_failures]]
        raise SystemExit(f"Candidate failed required gate(s): {', '.join(failure_names)}")


if __name__ == "__main__":
    main()
