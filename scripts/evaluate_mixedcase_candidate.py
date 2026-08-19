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
    MIXEDCASE_HYBRID_PATH,
    MIXEDCASE_LABELS,
    MIXEDCASE_WEIGHTS_PATH,
    MODEL_CLASSES,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    evaluate_mixedcase_breakdown,
    load_mixedcase_model,
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


def load_tensor_pack(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an external mixed-case tensor pack with `images` and `targets`."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), torch.Tensor):
        raise RuntimeError(f"{path} does not contain an images tensor.")
    if not isinstance(payload.get("targets"), torch.Tensor):
        raise RuntimeError(f"{path} does not contain a targets tensor.")
    images = payload["images"].float()
    targets = payload["targets"].long()
    if int(images.shape[0]) != int(targets.numel()):
        raise RuntimeError(f"{path} has {images.shape[0]} images but {targets.numel()} targets.")
    return images, targets


def candidate_test_tensors(
    sample_limit: int | None = None,
    seed: int = 2026,
    tensor_path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the default test tensors or a supplied external tensor pack."""

    if tensor_path is None:
        mnist_images, mnist_targets = build_or_load_mnist_cache(train=False)
        byclass_images, byclass_targets = build_or_load_emnist_byclass_mixedcase_cache(train=False)
        images = torch.cat([mnist_images, byclass_images])
        targets = torch.cat([mnist_targets, byclass_targets])
    else:
        images, targets = load_tensor_pack(tensor_path)
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


def _selected_device(device_name: str) -> torch.device:
    """Return the requested torch device for candidate evaluation."""

    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    return get_device()


def evaluate_deployed_stack(
    batch_size: int = 4096,
    device_name: str = "auto",
    sample_limit: int | None = None,
    tensor_path: Path | None = None,
) -> dict[str, object]:
    """Evaluate the fully deployed mixed-case stack on the candidate tensor split."""

    device = _selected_device(device_name)
    model, labels = load_mixedcase_model(device=device)
    if model is None or labels is None:
        raise RuntimeError("The deployed mixed-case model could not be loaded.")
    if list(labels) != list(MIXEDCASE_LABELS):
        raise RuntimeError("The deployed mixed-case labels do not match the expected label order.")
    images, targets = candidate_test_tensors(sample_limit=sample_limit, tensor_path=tensor_path)
    loader = DataLoader(TensorDataset(images, targets), batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_mixedcase_breakdown(model, loader, criterion, list(MIXEDCASE_LABELS), device)
    metrics["balanced_group_accuracy"] = min(
        metrics["digit_test_accuracy"],
        metrics["upper_test_accuracy"],
        metrics["lower_test_accuracy"],
    )
    return {
        "checkpoint_path": str(MIXEDCASE_WEIGHTS_PATH),
        "mode": "deployed",
        "sample_limit": sample_limit,
        "tensor_path": str(tensor_path) if tensor_path is not None else None,
        "total_examples": int(targets.numel()),
        "metrics": metrics,
    }


def evaluate_candidate(
    checkpoint_path: Path,
    batch_size: int = 4096,
    device_name: str = "auto",
    mode: str = "raw",
    sample_limit: int | None = None,
    allow_deployed_calibration: bool = False,
    hybrid_artifact_path: Path | None = None,
    include_deployed_baseline: bool = False,
    tensor_path: Path | None = None,
) -> dict[str, object]:
    """Evaluate one candidate checkpoint and return benchmark-style metrics."""

    device = _selected_device(device_name)
    model = load_candidate_checkpoint(checkpoint_path, device)
    images, targets = candidate_test_tensors(sample_limit=sample_limit, tensor_path=tensor_path)
    if mode == "hybrid":
        artifact_path = hybrid_artifact_path
        if artifact_path is None and checkpoint_path.resolve() == MIXEDCASE_WEIGHTS_PATH.resolve():
            artifact_path = MIXEDCASE_HYBRID_PATH
        if artifact_path is None and not allow_deployed_calibration:
            raise RuntimeError(
                "Hybrid candidate evaluation needs a candidate-specific hybrid artifact. "
                "Use --mode raw, pass --hybrid-artifact-path, or pass --allow-deployed-calibration "
                "only for an explicit diagnostic using deployed calibration."
            )
        if artifact_path is None:
            artifact_path = MIXEDCASE_HYBRID_PATH
        uses_deployed_checkpoint = checkpoint_path.resolve() == MIXEDCASE_WEIGHTS_PATH.resolve()
        metrics = hybrid_stack_metrics(
            model,
            images,
            targets,
            device,
            batch_size,
            apply_calibration=uses_deployed_checkpoint or allow_deployed_calibration,
            hybrid_artifact_path=artifact_path,
        )
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
    report: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "mode": mode,
        "hybrid_artifact_path": str(hybrid_artifact_path) if hybrid_artifact_path is not None else None,
        "sample_limit": sample_limit,
        "tensor_path": str(tensor_path) if tensor_path is not None else None,
        "total_examples": int(targets.numel()),
        "metrics": metrics,
    }
    if include_deployed_baseline:
        baseline_kwargs: dict[str, Path] = {}
        if tensor_path is not None:
            baseline_kwargs["tensor_path"] = tensor_path
        report["deployed_baseline"] = evaluate_deployed_stack(
            batch_size=batch_size,
            device_name=device_name,
            sample_limit=sample_limit,
            **baseline_kwargs,
        )
    return report


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


def read_baseline_mode(path: Path | None) -> str | None:
    """Read the evaluation mode stored in a baseline report, when present."""

    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("mode"), str):
        return str(payload["mode"])
    return None


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


def improvement_row(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    objective: str,
    min_delta: float,
) -> dict[str, object] | None:
    """Return one pass/fail row for the required objective improvement."""

    if objective not in GATE_KEYS or objective not in baseline_metrics:
        return None
    value = float(metrics.get(objective, 0.0))
    baseline = float(baseline_metrics[objective])
    delta = value - baseline
    return {
        "name": objective,
        "value": value,
        "baseline": baseline,
        "delta": delta,
        "min_delta": min_delta,
        "passed": delta >= min_delta,
    }


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
    parser.add_argument(
        "--allow-deployed-calibration",
        action="store_true",
        help="Allow hybrid diagnostics for non-deployed checkpoints using deployed calibration artifacts.",
    )
    parser.add_argument(
        "--hybrid-artifact-path",
        type=Path,
        default=None,
        help="Candidate-specific hybrid artifact to use for --mode hybrid.",
    )
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--tensor-path", type=Path, default=None, help="Optional external tensor pack to evaluate.")
    parser.add_argument("--target", type=float, default=95.0)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument(
        "--include-deployed-baseline",
        action="store_true",
        help="Evaluate the current fully deployed mixed-case stack on the same examples.",
    )
    parser.add_argument("--baseline-tolerance", type=float, default=0.0)
    parser.add_argument("--baseline-objective", choices=GATE_KEYS, default="test_accuracy")
    parser.add_argument("--baseline-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--allow-baseline-mode-mismatch",
        action="store_true",
        help="Allow comparing reports produced by different evaluation modes.",
    )
    parser.add_argument("--require-target", action="store_true")
    parser.add_argument("--require-baseline", action="store_true")
    parser.add_argument("--require-improvement", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    baseline_mode = read_baseline_mode(args.baseline_json)
    if baseline_mode is not None and baseline_mode != args.mode and not args.allow_baseline_mode_mismatch:
        raise RuntimeError(
            f"Baseline report mode {baseline_mode!r} does not match requested mode {args.mode!r}. "
            "Use a matching baseline JSON or pass --allow-baseline-mode-mismatch for diagnostics."
        )

    report = evaluate_candidate(
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device_name=args.device,
        mode=args.mode,
        sample_limit=args.sample_limit,
        tensor_path=args.tensor_path,
        allow_deployed_calibration=args.allow_deployed_calibration,
        hybrid_artifact_path=args.hybrid_artifact_path,
        include_deployed_baseline=args.include_deployed_baseline,
    )
    if (
        args.include_deployed_baseline
        and (args.require_baseline or args.require_improvement)
        and not args.allow_baseline_mode_mismatch
        and report.get("mode") != "deployed"
    ):
        raise RuntimeError(
            "Required deployed-baseline gates would compare different evaluation modes. "
            "Use --allow-baseline-mode-mismatch only for an explicit diagnostic."
        )
    target_rows = gate_rows(report["metrics"], args.target)
    baseline_metrics = read_baseline_metrics(args.baseline_json)
    if not baseline_metrics and args.include_deployed_baseline:
        deployed_baseline = report.get("deployed_baseline")
        if isinstance(deployed_baseline, dict):
            deployed_metrics = deployed_baseline.get("metrics")
            if isinstance(deployed_metrics, dict):
                baseline_metrics = {
                    key: float(value)
                    for key, value in deployed_metrics.items()
                    if key in GATE_KEYS and isinstance(value, (int, float))
                }
    comparison_rows = baseline_rows(report["metrics"], baseline_metrics, args.baseline_tolerance)
    objective_row = improvement_row(
        report["metrics"],
        baseline_metrics,
        args.baseline_objective,
        args.baseline_min_delta,
    )
    report["gates"] = target_rows
    report["baseline_gates"] = comparison_rows
    report["improvement_gate"] = objective_row
    target_failures = failed_rows(target_rows) if args.require_target else []
    baseline_failures = failed_rows(comparison_rows) if args.require_baseline else []
    improvement_failures = (
        [objective_row]
        if args.require_improvement and (objective_row is None or not bool(objective_row.get("passed")))
        else []
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"checkpoint: {report['checkpoint_path']}")
        print(f"mode: {report['mode']} examples: {report['total_examples']}")
        _print_rows("target gates:", target_rows)
        _print_rows("baseline gates:", comparison_rows)
        if objective_row is not None:
            print(
                "improvement gate: "
                f"{objective_row['name']} delta={objective_row['delta']:.4f}% "
                f"min_delta={objective_row['min_delta']:.4f}% "
                f"{'PASS' if objective_row['passed'] else 'FAIL'}"
            )
    if target_failures or baseline_failures or improvement_failures:
        failure_names = [
            str(row["name"] if row is not None else args.baseline_objective)
            for row in [*target_failures, *baseline_failures, *improvement_failures]
        ]
        raise SystemExit(f"Candidate failed required gate(s): {', '.join(failure_names)}")


if __name__ == "__main__":
    main()
