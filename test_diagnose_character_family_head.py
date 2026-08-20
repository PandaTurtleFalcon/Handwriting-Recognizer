import unittest

import torch

from scripts.diagnose_character_family_head import family_head_accuracy, family_mask, split_head_report


class DiagnoseCharacterFamilyHeadTests(unittest.TestCase):
    """Focused tests for character-family head diagnostics."""

    def test_family_mask_selects_true_family_targets(self) -> None:
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        self.assertEqual(family_mask(targets, (1, 3)).tolist(), [False, True, False, True])

    def test_family_head_accuracy_uses_local_family_targets(self) -> None:
        class FixedHead(torch.nn.Module):
            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return torch.tensor(
                    [
                        [4.0, 0.0],
                        [0.0, 4.0],
                        [4.0, 0.0],
                    ],
                    dtype=torch.float32,
                )[: features.shape[0]]

        features = torch.zeros((4, 2), dtype=torch.float32)
        targets = torch.tensor([5, 9, 5, 0], dtype=torch.long)

        report = family_head_accuracy(FixedHead(), features, targets, (5, 9))

        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["correct"], 3)
        self.assertEqual(report["accuracy"], 100.0)

    def test_split_head_report_compares_base_and_family_head(self) -> None:
        class AlwaysSecond(torch.nn.Module):
            def forward(self, features: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros((features.shape[0], 2), dtype=torch.float32)
                logits[:, 1] = 1.0
                return logits

        labels = ["A", "B", "0"]
        predictions = torch.tensor([0, 0, 2], dtype=torch.long)
        targets = torch.tensor([0, 1, 2], dtype=torch.long)
        features = torch.zeros((3, 2), dtype=torch.float32)

        report = split_head_report("mock", predictions, targets, labels, features, AlwaysSecond(), (0, 1))

        self.assertEqual(report["split"], "mock")
        self.assertEqual(report["family_samples"], 2)
        self.assertEqual(report["base_family_accuracy"], 50.0)
        self.assertEqual(report["head_family_accuracy"], 50.0)
        self.assertAlmostEqual(report["base_metrics"]["validation_accuracy"], 100.0 * 2 / 3, places=4)


if __name__ == "__main__":
    unittest.main()
