import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from alnum_model import MIXEDCASE_LABELS
import scripts.evaluate_mixedcase_candidate as mixedcase_candidate
from scripts.evaluate_mixedcase_candidate import (
    baseline_rows,
    candidate_test_tensors,
    evaluate_candidate,
    failed_rows,
    gate_rows,
    improvement_row,
    load_candidate_checkpoint,
    read_baseline_metrics,
    read_baseline_mode,
)


class MixedcaseCandidateEvaluatorTests(unittest.TestCase):
    """Focused tests for candidate-only mixed-case checkpoint evaluation."""

    def test_gate_rows_reports_main_mixedcase_gates(self) -> None:
        rows = gate_rows(
            {
                "test_accuracy": 96.0,
                "case_or_ambiguity_aware_test_accuracy": 99.0,
                "digit_test_accuracy": 95.0,
                "upper_test_accuracy": 94.9,
                "lower_test_accuracy": 95.1,
            },
            target=95.0,
        )

        by_name = {row["name"]: row for row in rows}
        self.assertTrue(by_name["test_accuracy"]["passed"])
        self.assertTrue(by_name["digit_test_accuracy"]["passed"])
        self.assertFalse(by_name["upper_test_accuracy"]["passed"])
        self.assertEqual(by_name["upper_test_accuracy"]["value"], 94.9)

    def test_baseline_rows_allow_tolerance_for_noisy_samples(self) -> None:
        rows = baseline_rows(
            {
                "test_accuracy": 87.78,
                "case_or_ambiguity_aware_test_accuracy": 98.04,
            },
            {
                "test_accuracy": 87.79,
                "case_or_ambiguity_aware_test_accuracy": 98.04,
            },
            tolerance=0.02,
        )

        self.assertTrue(all(row["passed"] for row in rows))

    def test_failed_rows_returns_only_required_failures(self) -> None:
        rows = [
            {"name": "test_accuracy", "passed": True},
            {"name": "upper_test_accuracy", "passed": False},
        ]

        self.assertEqual(failed_rows(rows), [{"name": "upper_test_accuracy", "passed": False}])

    def test_improvement_row_requires_minimum_delta(self) -> None:
        row = improvement_row(
            {"test_accuracy": 87.80},
            {"test_accuracy": 87.78},
            "test_accuracy",
            min_delta=0.05,
        )

        self.assertIsNotNone(row)
        self.assertFalse(row["passed"])
        self.assertAlmostEqual(row["delta"], 0.02)

    def test_read_baseline_metrics_accepts_nested_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(
                '{"metrics":{"test_accuracy":87.7,"upper_test_accuracy":84.6},"ignored":true}',
                encoding="utf-8",
            )

            metrics = read_baseline_metrics(path)

        self.assertEqual(metrics, {"test_accuracy": 87.7, "upper_test_accuracy": 84.6})

    def test_read_baseline_mode_accepts_nested_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text('{"mode":"hybrid","metrics":{"test_accuracy":87.7}}', encoding="utf-8")

            mode = read_baseline_mode(path)

        self.assertEqual(mode, "hybrid")

    def test_main_rejects_baseline_mode_mismatch_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text('{"mode":"hybrid","metrics":{"test_accuracy":87.7}}', encoding="utf-8")
            argv = [
                "evaluate_mixedcase_candidate.py",
                "--mode",
                "raw",
                "--baseline-json",
                str(path),
            ]

            with (
                patch("sys.argv", argv),
                patch("scripts.evaluate_mixedcase_candidate.evaluate_candidate") as evaluator,
            ):
                with self.assertRaisesRegex(RuntimeError, "does not match requested mode"):
                    mixedcase_candidate.main()

            evaluator.assert_not_called()

    def test_candidate_test_tensors_can_sample_deterministically(self) -> None:
        mnist_images = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1)
        mnist_targets = torch.arange(6)
        byclass_images = torch.arange(6, 12, dtype=torch.float32).view(6, 1, 1, 1)
        byclass_targets = torch.arange(6, 12)

        with (
            patch(
                "scripts.evaluate_mixedcase_candidate.build_or_load_mnist_cache",
                return_value=(mnist_images, mnist_targets),
            ),
            patch(
                "scripts.evaluate_mixedcase_candidate.build_or_load_emnist_byclass_mixedcase_cache",
                return_value=(byclass_images, byclass_targets),
            ),
        ):
            first_images, first_targets = candidate_test_tensors(sample_limit=5, seed=7)
            second_images, second_targets = candidate_test_tensors(sample_limit=5, seed=7)

        self.assertEqual(first_targets.tolist(), second_targets.tolist())
        self.assertEqual(first_images.flatten().tolist(), second_images.flatten().tolist())
        self.assertEqual(int(first_targets.numel()), 5)

    def test_load_candidate_checkpoint_rejects_wrong_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "bad.pt"
            torch.save(
                {
                    "labels": list(MIXEDCASE_LABELS[:-1]),
                    "model_type": "cnn",
                    "model_state_dict": {},
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(RuntimeError, "label order"):
                load_candidate_checkpoint(checkpoint_path, torch.device("cpu"))

    def test_hybrid_mode_rejects_non_deployed_candidate_by_default(self) -> None:
        class FixedModel(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return torch.zeros((inputs.size(0), len(MIXEDCASE_LABELS)))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "candidate.pt"
            with (
                patch(
                    "scripts.evaluate_mixedcase_candidate.candidate_test_tensors",
                    return_value=(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long)),
                ),
                patch("scripts.evaluate_mixedcase_candidate.load_candidate_checkpoint", return_value=FixedModel()),
            ):
                with self.assertRaisesRegex(RuntimeError, "candidate-specific hybrid artifact"):
                    evaluate_candidate(checkpoint_path, device_name="cpu", mode="hybrid")

    def test_hybrid_mode_uses_explicit_candidate_artifact_path(self) -> None:
        class FixedModel(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return torch.zeros((inputs.size(0), len(MIXEDCASE_LABELS)))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "candidate.pt"
            artifact_path = Path(directory) / "candidate_hybrid.json"
            with (
                patch(
                    "scripts.evaluate_mixedcase_candidate.candidate_test_tensors",
                    return_value=(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long)),
                ),
                patch("scripts.evaluate_mixedcase_candidate.load_candidate_checkpoint", return_value=FixedModel()),
                patch(
                    "scripts.evaluate_mixedcase_candidate.hybrid_stack_metrics",
                    return_value={
                        "test_accuracy": 1.0,
                        "case_or_ambiguity_aware_test_accuracy": 1.0,
                        "digit_test_accuracy": 1.0,
                        "upper_test_accuracy": 1.0,
                        "lower_test_accuracy": 1.0,
                    },
                ) as stack_metrics,
            ):
                report = evaluate_candidate(
                    checkpoint_path,
                    device_name="cpu",
                    mode="hybrid",
                    hybrid_artifact_path=artifact_path,
                )

        self.assertEqual(report["hybrid_artifact_path"], str(artifact_path))
        self.assertEqual(stack_metrics.call_args.kwargs["hybrid_artifact_path"], artifact_path)
        self.assertFalse(stack_metrics.call_args.kwargs["apply_calibration"])


if __name__ == "__main__":
    unittest.main()
