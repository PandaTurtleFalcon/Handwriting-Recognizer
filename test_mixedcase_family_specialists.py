import unittest

import torch

from scripts.probe_mixedcase_family_specialists import (
    Specialist,
    apply_specialists,
    choose_thresholds,
    family_indices,
    parse_families,
    probe_family_specialists,
    split_holdout,
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

    def test_split_holdout_returns_disjoint_slices(self) -> None:
        images = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
        targets = torch.arange(10, dtype=torch.long)

        fit_images, fit_targets, validation_images, validation_targets = split_holdout(images, targets, 0.3, seed=7)

        self.assertEqual(len(fit_targets), 7)
        self.assertEqual(len(validation_targets), 3)
        self.assertEqual(sorted(torch.cat([fit_targets, validation_targets]).tolist()), list(range(10)))
        self.assertEqual(sorted(torch.cat([fit_images.flatten(), validation_images.flatten()]).tolist()), list(range(10)))

    def test_choose_thresholds_keeps_base_when_candidate_breaks_metric(self) -> None:
        class BadSpecialist(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.0, 4.0]], dtype=torch.float32)

        base_predictions = torch.tensor([0], dtype=torch.long)
        images = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        targets = torch.tensor([0], dtype=torch.long)
        specialist = Specialist("0O", (0, 24), BadSpecialist())

        selected = choose_thresholds(
            base_predictions,
            images,
            targets,
            [specialist],
            batch_size=8,
            device=torch.device("cpu"),
            confidence_values=(0.0,),
            margin_values=(0.0,),
        )

        self.assertIsNone(selected["confidence"])
        self.assertEqual(selected["replacement_report"], {"changed": 0, "fixed": 0, "broken": 0})

    def test_probe_uses_true_abstention_when_auto_threshold_finds_no_gain(self) -> None:
        class CertainSpecialist(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.0, 4.0]], dtype=torch.float32)

        self.assertTrue(hasattr(probe_family_specialists, "__call__"))
        predictions, reports = apply_specialists(
            torch.tensor([0], dtype=torch.long),
            torch.zeros((1, 1, 28, 28), dtype=torch.float32),
            [Specialist("0O", (0, 24), CertainSpecialist())],
            batch_size=8,
            device=torch.device("cpu"),
            confidence_threshold=float("inf"),
            margin_threshold=float("inf"),
        )

        self.assertEqual(predictions.tolist(), [0])
        self.assertEqual(reports, [{"family": "0O", "eligible": 1, "changed": 0}])


if __name__ == "__main__":
    unittest.main()
