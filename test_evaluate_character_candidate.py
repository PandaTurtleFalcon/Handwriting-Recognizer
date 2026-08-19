import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import scripts.evaluate_character_candidate as character_candidate
from scripts.evaluate_character_candidate import (
    baseline_rows,
    candidate_validation_tensors,
    evaluate_candidate,
    evaluate_deployed_stack,
    failed_rows,
    gate_rows,
    improvement_row,
    load_candidate_checkpoint,
    read_baseline_metrics,
    read_baseline_mode,
)


class CharacterCandidateEvaluatorTests(unittest.TestCase):
    """Focused tests for candidate-only character checkpoint evaluation."""

    def test_gate_rows_reports_main_character_gates(self) -> None:
        rows = gate_rows(
            {
                "validation_accuracy": 95.1,
                "ambiguity_aware_validation_accuracy": 99.0,
                "digit_validation_accuracy": 95.0,
                "letter_validation_accuracy": 94.9,
                "punctuation_validation_accuracy": 96.0,
            },
            target=95.0,
        )

        by_name = {row["name"]: row for row in rows}
        self.assertTrue(by_name["validation_accuracy"]["passed"])
        self.assertTrue(by_name["digit_validation_accuracy"]["passed"])
        self.assertFalse(by_name["letter_validation_accuracy"]["passed"])
        self.assertEqual(by_name["letter_validation_accuracy"]["value"], 94.9)

    def test_baseline_rows_allow_tolerance_for_noisy_samples(self) -> None:
        rows = baseline_rows(
            {
                "validation_accuracy": 94.16,
                "letter_validation_accuracy": 93.59,
            },
            {
                "validation_accuracy": 94.17,
                "letter_validation_accuracy": 93.60,
            },
            tolerance=0.02,
        )

        self.assertTrue(all(row["passed"] for row in rows))

    def test_failed_rows_returns_only_required_failures(self) -> None:
        rows = [
            {"name": "validation_accuracy", "passed": True},
            {"name": "letter_validation_accuracy", "passed": False},
        ]

        self.assertEqual(failed_rows(rows), [{"name": "letter_validation_accuracy", "passed": False}])

    def test_improvement_row_requires_minimum_delta(self) -> None:
        row = improvement_row(
            {"letter_validation_accuracy": 93.62},
            {"letter_validation_accuracy": 93.59},
            "letter_validation_accuracy",
            min_delta=0.05,
        )

        self.assertIsNotNone(row)
        self.assertFalse(row["passed"])
        self.assertAlmostEqual(row["delta"], 0.03)

    def test_read_baseline_metrics_accepts_nested_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(
                '{"metrics":{"validation_accuracy":94.1,"letter_validation_accuracy":93.5},"ignored":true}',
                encoding="utf-8",
            )

            metrics = read_baseline_metrics(path)

        self.assertEqual(metrics, {"validation_accuracy": 94.1, "letter_validation_accuracy": 93.5})

    def test_read_baseline_mode_accepts_nested_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text('{"mode":"raw","metrics":{"validation_accuracy":94.1}}', encoding="utf-8")

            mode = read_baseline_mode(path)

        self.assertEqual(mode, "raw")

    def test_main_rejects_baseline_mode_mismatch_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text('{"mode":"raw","metrics":{"validation_accuracy":94.1}}', encoding="utf-8")
            argv = [
                "evaluate_character_candidate.py",
                "--mode",
                "calibrated",
                "--baseline-json",
                str(path),
            ]

            with (
                patch("sys.argv", argv),
                patch("scripts.evaluate_character_candidate.evaluate_candidate") as evaluator,
            ):
                with self.assertRaisesRegex(RuntimeError, "does not match requested mode"):
                    character_candidate.main()

            evaluator.assert_not_called()

    def test_candidate_validation_tensors_can_sample_deterministically(self) -> None:
        images = torch.arange(12, dtype=torch.float32).view(12, 1, 1, 1)
        targets = torch.arange(12)
        labels = [str(index) for index in range(12)]

        with patch(
            "scripts.evaluate_character_candidate.validation_tensors",
            return_value=(images, targets, labels),
        ):
            first_images, first_targets, first_labels = candidate_validation_tensors(sample_limit=5, seed=7)
            second_images, second_targets, second_labels = candidate_validation_tensors(sample_limit=5, seed=7)

        self.assertEqual(first_targets.tolist(), second_targets.tolist())
        self.assertEqual(first_images.flatten().tolist(), second_images.flatten().tolist())
        self.assertEqual(first_labels, second_labels)
        self.assertEqual(int(first_targets.numel()), 5)

    def test_load_candidate_checkpoint_rejects_wrong_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "bad.pt"
            torch.save(
                {
                    "labels": ["A"],
                    "model_type": "cnn",
                    "model_state_dict": {},
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(RuntimeError, "label order"):
                load_candidate_checkpoint(checkpoint_path, ["A", "B"], torch.device("cpu"))

    def test_calibrated_mode_rejects_non_deployed_candidate_by_default(self) -> None:
        class FixedModel(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return torch.zeros((inputs.size(0), 2))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "candidate.pt"
            with (
                patch(
                    "scripts.evaluate_character_candidate.candidate_validation_tensors",
                    return_value=(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long), ["A", "B"]),
                ),
                patch("scripts.evaluate_character_candidate.load_candidate_checkpoint", return_value=FixedModel()),
            ):
                with self.assertRaisesRegex(RuntimeError, "deployed character calibration"):
                    evaluate_candidate(checkpoint_path, device_name="cpu", mode="calibrated")

    def test_calibrated_mode_uses_candidate_artifacts_without_deployed_calibration(self) -> None:
        class FixedModel(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "candidate.pt"
            bias_path = Path(directory) / "candidate_bias.pt"
            rules_path = Path(directory) / "candidate_rules.json"
            images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
            targets = torch.tensor([0, 1], dtype=torch.long)
            labels = ["A", "B"]
            with (
                patch(
                    "scripts.evaluate_character_candidate.candidate_validation_tensors",
                    return_value=(images, targets, labels),
                ),
                patch("scripts.evaluate_character_candidate.load_candidate_checkpoint", return_value=FixedModel()),
                patch("scripts.evaluate_character_candidate.attach_character_logit_bias") as attach_bias,
                patch("scripts.evaluate_character_candidate.attach_character_pair_rules") as attach_rules,
                patch("scripts.evaluate_character_candidate.calibrated_predictions") as predictions,
            ):
                predictions.return_value = targets
                report = evaluate_candidate(
                    checkpoint_path,
                    device_name="cpu",
                    mode="calibrated",
                    logit_bias_path=bias_path,
                    pair_rules_path=rules_path,
                )

            self.assertEqual(report["metrics"]["validation_accuracy"], 100.0)
            attach_bias.assert_called_once()
            attach_rules.assert_called_once()
            self.assertFalse(predictions.call_args.kwargs["apply_calibration"])

    def test_evaluate_deployed_stack_uses_wrapped_model(self) -> None:
        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        targets = torch.tensor([1, 1], dtype=torch.long)
        labels = ["A", "B"]

        class FixedModel(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return torch.zeros((inputs.size(0), len(labels)), dtype=torch.float32)

        with (
            patch(
                "scripts.evaluate_character_candidate.candidate_validation_tensors",
                return_value=(images, targets, labels),
            ),
            patch("scripts.evaluate_character_candidate.load_character_model", return_value=(FixedModel(), labels)),
            patch("scripts.evaluate_character_candidate.calibrated_predictions", return_value=targets) as predictions,
        ):
            report = evaluate_deployed_stack(batch_size=2, device_name="cpu", sample_limit=2)

        self.assertEqual(report["mode"], "deployed")
        self.assertEqual(report["total_examples"], 2)
        self.assertEqual(report["metrics"]["validation_accuracy"], 100.0)
        self.assertTrue(predictions.call_args.kwargs["apply_calibration"])

    def test_candidate_can_include_deployed_baseline_report(self) -> None:
        class FixedModel(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return torch.zeros((inputs.size(0), 2), dtype=torch.float32)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "candidate.pt"
            with (
                patch(
                    "scripts.evaluate_character_candidate.candidate_validation_tensors",
                    return_value=(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long), ["A", "B"]),
                ),
                patch("scripts.evaluate_character_candidate.load_candidate_checkpoint", return_value=FixedModel()),
                patch(
                    "scripts.evaluate_character_candidate.evaluate_deployed_stack",
                    return_value={
                        "mode": "deployed",
                        "metrics": {
                            "validation_accuracy": 99.0,
                            "ambiguity_aware_validation_accuracy": 99.0,
                            "digit_validation_accuracy": 99.0,
                            "letter_validation_accuracy": 99.0,
                            "punctuation_validation_accuracy": 99.0,
                        },
                    },
                ) as deployed,
            ):
                report = evaluate_candidate(
                    checkpoint_path,
                    batch_size=4,
                    device_name="cpu",
                    mode="raw",
                    include_deployed_baseline=True,
                    sample_limit=1,
                )

        self.assertEqual(report["deployed_baseline"]["mode"], "deployed")
        deployed.assert_called_once_with(batch_size=4, device_name="cpu", sample_limit=1)

    def test_main_can_compare_against_deployed_baseline(self) -> None:
        candidate_report = {
            "checkpoint_path": "candidate.pt",
            "mode": "raw",
            "sample_limit": None,
            "total_examples": 1,
            "metrics": {
                "validation_accuracy": 98.0,
                "ambiguity_aware_validation_accuracy": 98.0,
                "digit_validation_accuracy": 98.0,
                "letter_validation_accuracy": 98.0,
                "punctuation_validation_accuracy": 98.0,
            },
            "deployed_baseline": {
                "mode": "deployed",
                "metrics": {
                    "validation_accuracy": 97.0,
                    "ambiguity_aware_validation_accuracy": 97.0,
                    "digit_validation_accuracy": 97.0,
                    "letter_validation_accuracy": 97.0,
                    "punctuation_validation_accuracy": 97.0,
                },
            },
        }
        argv = [
            "evaluate_character_candidate.py",
            "--include-deployed-baseline",
            "--require-baseline",
            "--allow-baseline-mode-mismatch",
            "--json",
        ]

        with (
            patch("sys.argv", argv),
            patch("scripts.evaluate_character_candidate.evaluate_candidate", return_value=candidate_report),
            patch("builtins.print") as printer,
        ):
            character_candidate.main()

        payload = printer.call_args.args[0]
        report = character_candidate.json.loads(payload)
        self.assertEqual(len(report["baseline_gates"]), len(character_candidate.GATE_KEYS))
        self.assertTrue(all(row["passed"] for row in report["baseline_gates"]))

    def test_main_rejects_required_deployed_baseline_mode_mismatch_by_default(self) -> None:
        candidate_report = {
            "checkpoint_path": "candidate.pt",
            "mode": "raw",
            "sample_limit": None,
            "total_examples": 1,
            "metrics": {"validation_accuracy": 98.0},
            "deployed_baseline": {"mode": "deployed", "metrics": {"validation_accuracy": 97.0}},
        }
        argv = ["evaluate_character_candidate.py", "--include-deployed-baseline", "--require-baseline"]

        with (
            patch("sys.argv", argv),
            patch("scripts.evaluate_character_candidate.evaluate_candidate", return_value=candidate_report),
        ):
            with self.assertRaisesRegex(RuntimeError, "different evaluation modes"):
                character_candidate.main()


if __name__ == "__main__":
    unittest.main()
