import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.evaluate_character_candidate import (
    baseline_rows,
    candidate_validation_tensors,
    failed_rows,
    gate_rows,
    load_candidate_checkpoint,
    read_baseline_metrics,
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

    def test_read_baseline_metrics_accepts_nested_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(
                '{"metrics":{"validation_accuracy":94.1,"letter_validation_accuracy":93.5},"ignored":true}',
                encoding="utf-8",
            )

            metrics = read_baseline_metrics(path)

        self.assertEqual(metrics, {"validation_accuracy": 94.1, "letter_validation_accuracy": 93.5})

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


if __name__ == "__main__":
    unittest.main()
