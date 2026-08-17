import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.summarize_benchmarks import summarize_app_hardcases, summarize_correction_memory, summarize_saved_metrics


class BenchmarkSummaryTests(unittest.TestCase):
    """Regression tests for saved benchmark gate summaries."""

    def test_summarizes_pass_fail_saved_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}),
                encoding="utf-8",
            )
            (root / "alnum_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}),
                encoding="utf-8",
            )
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "test_accuracy": 80.0,
                            "case_or_ambiguity_aware_test_accuracy": 97.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 92.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "punctuation_validation_accuracy": 95.2,
                            "punctuation_ambiguity_aware_validation_accuracy": 98.6,
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertTrue(by_name["digit_specialist_exact"]["passed"])
        self.assertFalse(by_name["mixedcase_exact"]["passed"])
        self.assertTrue(by_name["mixedcase_case_or_visual"]["passed"])
        self.assertFalse(by_name["character_exact"]["passed"])
        self.assertTrue(by_name["punctuation_exact"]["passed"])

    def test_summarizes_app_hardcase_gates_on_demand(self) -> None:
        with patch(
            "scripts.evaluate_hardcases.evaluate_cases",
            return_value={
                "exact_accuracy": 100.0,
                "exact_correct": 176,
                "ambiguity_aware_accuracy": 100.0,
                "ambiguity_aware_correct": 176,
                "total": 176,
            },
        ) as evaluate:
            report = summarize_app_hardcases(target=95.0, all_fonts=False)

        evaluate.assert_called_once_with(all_fonts=False)
        by_name = {str(item["name"]): item for item in report}
        self.assertTrue(by_name["app_hardcase_exact"]["passed"])
        self.assertTrue(by_name["app_hardcase_ambiguity"]["passed"])
        self.assertEqual(by_name["app_hardcase_exact"]["correct"], 176)
        self.assertEqual(by_name["app_hardcase_exact"]["total"], 176)

    def test_summarizes_correction_memory_priority_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "character_labels.json").write_text(json.dumps(["s", "O", "V"]), encoding="utf-8")

            with patch("main.PRACTICE_PRIORITY_LABELS", ["s", "O"]):
                with patch("main.PRACTICE_TARGET_PER_LABEL", 2):
                    with patch(
                        "character_model.load_correction_memory_exemplars",
                        return_value=(torch.zeros((3, 1, 32, 32)), torch.tensor([0, 0, 1])),
                    ):
                        report = summarize_correction_memory(target=95.0, project_dir=root)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["character_correction_memory_samples"]["correct"], 3)
        self.assertEqual(by_name["character_correction_memory_samples"]["total"], 4)
        self.assertAlmostEqual(by_name["character_correction_memory_samples"]["value"], 75.0)
        self.assertEqual(by_name["character_correction_memory_samples"]["by_label"], {"s": 2, "O": 1})
        self.assertEqual(by_name["character_correction_memory_samples"]["not_ready_label_list"], ["O"])
        self.assertEqual(by_name["character_correction_memory_ready_labels"]["correct"], 1)
        self.assertEqual(by_name["character_correction_memory_ready_labels"]["total"], 2)
        self.assertFalse(by_name["character_correction_memory_ready_labels"]["passed"])


if __name__ == "__main__":
    unittest.main()
