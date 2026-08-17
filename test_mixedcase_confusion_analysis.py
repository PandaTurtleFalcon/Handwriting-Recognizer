import unittest
from unittest.mock import patch

import torch
from torch import nn

from scripts.analyze_mixedcase_confusions import analyze_confusions, mixedcase_error_budget


class MixedcaseConfusionAnalysisTests(unittest.TestCase):
    """Regression tests for mixed-case confusion reporting."""

    def test_analyze_confusions_reports_top_pairs_and_groups(self) -> None:
        class FixedModel(nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros((images.size(0), 4), dtype=torch.float32)
                predictions = [1, 1, 2, 3]
                for row, label_index in enumerate(predictions[: images.size(0)]):
                    logits[row, label_index] = 10.0
                return logits

        images = torch.zeros((4, 1, 28, 28), dtype=torch.float32)
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        labels = ["0", "O", "A", "a"]

        with (
            patch("scripts.analyze_mixedcase_confusions.get_device", return_value=torch.device("cpu")),
            patch("scripts.analyze_mixedcase_confusions.load_mixedcase_model", return_value=(FixedModel(), labels)),
            patch("scripts.analyze_mixedcase_confusions.build_or_load_mnist_cache", return_value=(images[:2], targets[:2])),
            patch(
                "scripts.analyze_mixedcase_confusions.build_or_load_emnist_byclass_mixedcase_cache",
                return_value=(images[2:], targets[2:]),
            ),
        ):
            report = analyze_confusions(batch_size=4, top=3)

        self.assertEqual(report["total"], 4)
        self.assertAlmostEqual(report["exact_accuracy"], 75.0)
        self.assertEqual(report["top_confusions"], [{"expected": "0", "predicted": "O", "count": 1}])
        self.assertEqual(report["visual_twin_error_budget"][0]["family"], "0Oo")
        self.assertAlmostEqual(report["visual_twin_error_budget"][0]["error_percent"], 100.0)
        self.assertAlmostEqual(report["group_accuracy"]["digit"], 0.0)
        self.assertAlmostEqual(report["group_accuracy"]["upper"], 100.0)
        self.assertAlmostEqual(report["group_accuracy"]["lower"], 100.0)

    def test_mixedcase_error_budget_ranks_visual_twin_families(self) -> None:
        """The budget should show which visual families consume exact errors."""

        budget = mixedcase_error_budget(
            {
                ("1", "l"): 5,
                ("I", "l"): 3,
                ("A", "B"): 10,
                ("O", "0"): 2,
            },
            total_errors=20,
            top=2,
        )

        self.assertEqual(budget[0]["family"], "!/1Iil|")
        self.assertEqual(budget[0]["count"], 8)
        self.assertAlmostEqual(budget[0]["error_percent"], 40.0)
        self.assertEqual(budget[1]["family"], "0Oo")


if __name__ == "__main__":
    unittest.main()
