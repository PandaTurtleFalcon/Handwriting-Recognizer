import unittest

import torch

from scripts.probe_mixedcase_family_specialists import (
    Specialist,
    apply_specialists,
    family_indices,
    parse_families,
)


class MixedcaseFamilySpecialistTests(unittest.TestCase):
    def test_family_indices_keep_requested_order(self) -> None:
        self.assertEqual(family_indices("0Oo"), (0, 24, 50))
        self.assertEqual(family_indices("1Ili"), (1, 18, 47, 44))

    def test_parse_families_uses_default_when_blank(self) -> None:
        self.assertIn("0Oo", parse_families(""))
        self.assertEqual(parse_families("0Oo, 5Ss"), ("0Oo", "5Ss"))

    def test_apply_specialists_only_replaces_matching_family_predictions(self) -> None:
        class FixedSpecialist(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.0, 2.0], [0.0, 2.0]], dtype=torch.float32)

        images = torch.zeros((3, 1, 28, 28), dtype=torch.float32)
        base_predictions = torch.tensor([0, 24, 2], dtype=torch.long)
        specialist = Specialist("0O", (0, 24), FixedSpecialist())

        predictions, reports = apply_specialists(
            base_predictions,
            images,
            [specialist],
            batch_size=8,
            device=torch.device("cpu"),
            confidence_threshold=0.0,
            margin_threshold=0.0,
        )

        self.assertEqual(predictions.tolist(), [24, 24, 2])
        self.assertEqual(reports, [{"family": "0O", "eligible": 2, "changed": 1}])

    def test_apply_specialists_respects_confidence_and_margin_gates(self) -> None:
        class UncertainSpecialist(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.0, 0.1]], dtype=torch.float32)

        images = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        base_predictions = torch.tensor([0], dtype=torch.long)
        specialist = Specialist("0O", (0, 24), UncertainSpecialist())

        predictions, reports = apply_specialists(
            base_predictions,
            images,
            [specialist],
            batch_size=8,
            device=torch.device("cpu"),
            confidence_threshold=0.8,
            margin_threshold=0.2,
        )

        self.assertEqual(predictions.tolist(), [0])
        self.assertEqual(reports, [{"family": "0O", "eligible": 1, "changed": 0}])


if __name__ == "__main__":
    unittest.main()
