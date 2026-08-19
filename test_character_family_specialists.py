import unittest
from unittest.mock import patch

import torch

from scripts.probe_character_family_specialists import (
    Specialist,
    apply_specialists,
    choose_thresholds,
    deployed_predictions,
    family_indices,
    parse_families,
    parse_source_groups,
    source_group_mask,
    split_holdout,
)


class CharacterFamilySpecialistTests(unittest.TestCase):
    def test_family_indices_keep_requested_order(self) -> None:
        labels = ["!", "/", "1", "I", "i", "l", "|"]

        self.assertEqual(family_indices("!/1Iil|", labels), (0, 1, 2, 3, 4, 5, 6))

    def test_parse_families_uses_defaults_for_blank(self) -> None:
        self.assertIn("!/1Iil|", parse_families(""))
        self.assertEqual(parse_families("0Oo, 5Ss"), ("0Oo", "5Ss"))

    def test_source_group_mask_filters_current_prediction_groups(self) -> None:
        labels = ["!", "1", "A", "a"]
        predictions = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        self.assertEqual(parse_source_groups("letter,digit"), ("letter", "digit"))
        self.assertEqual(source_group_mask(predictions, labels, ("letter",)).tolist(), [False, False, True, True])
        with self.assertRaisesRegex(ValueError, "Unknown source group"):
            parse_source_groups("symbol")

    def test_split_holdout_returns_disjoint_slices(self) -> None:
        images = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
        targets = torch.arange(10, dtype=torch.long)

        fit_images, fit_targets, holdout_images, holdout_targets = split_holdout(images, targets, 0.3, seed=7)

        self.assertEqual(len(fit_targets), 7)
        self.assertEqual(len(holdout_targets), 3)
        self.assertEqual(sorted(torch.cat([fit_targets, holdout_targets]).tolist()), list(range(10)))
        self.assertEqual(sorted(torch.cat([fit_images.flatten(), holdout_images.flatten()]).tolist()), list(range(10)))

    def test_apply_specialists_respects_confidence_margin_and_groups(self) -> None:
        class FixedSpecialist(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.0, 3.0]], dtype=torch.float32).repeat(images.size(0), 1)

        labels = ["0", "O", "A"]
        images = torch.zeros((3, 1, 32, 32), dtype=torch.float32)
        base_predictions = torch.tensor([0, 1, 2], dtype=torch.long)
        specialist = Specialist("0O", (0, 1), FixedSpecialist())

        predictions, reports = apply_specialists(
            base_predictions,
            images,
            [specialist],
            labels,
            batch_size=8,
            device=torch.device("cpu"),
            confidence_threshold=0.0,
            margin_threshold=0.0,
            source_groups=("digit",),
        )

        self.assertEqual(predictions.tolist(), [1, 1, 2])
        self.assertEqual(reports, [{"family": "0O", "eligible": 1, "changed": 1}])

    def test_choose_thresholds_keeps_base_when_candidate_breaks_metric(self) -> None:
        class BadSpecialist(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.0, 3.0]], dtype=torch.float32)

        labels = ["0", "O"]
        base_predictions = torch.tensor([0], dtype=torch.long)
        images = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        targets = torch.tensor([0], dtype=torch.long)
        specialist = Specialist("0O", (0, 1), BadSpecialist())

        selected = choose_thresholds(
            base_predictions,
            images,
            targets,
            labels,
            [specialist],
            batch_size=8,
            device=torch.device("cpu"),
            confidence_grid=(0.0,),
            margin_grid=(0.0,),
            source_groups=("digit", "letter", "punctuation"),
        )

        self.assertIsNone(selected["confidence"])
        self.assertEqual(selected["replacement_report"], {"changed": 0, "fixed": 0, "broken": 0})
        self.assertEqual(dict(selected["best_rejected"])["gain"], -100.0)

    def test_choose_thresholds_reports_best_rejected_gain(self) -> None:
        class RiskySpecialist(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.0, 3.0], [0.0, 3.0]], dtype=torch.float32)

        labels = ["0", "O"]
        base_predictions = torch.tensor([0, 0], dtype=torch.long)
        images = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
        targets = torch.tensor([1, 0], dtype=torch.long)
        specialist = Specialist("0O", (0, 1), RiskySpecialist())

        selected = choose_thresholds(
            base_predictions,
            images,
            targets,
            labels,
            [specialist],
            batch_size=8,
            device=torch.device("cpu"),
            confidence_grid=(0.0,),
            margin_grid=(0.0,),
            source_groups=("digit", "letter", "punctuation"),
        )

        self.assertIsNone(selected["confidence"])
        self.assertEqual(dict(selected["best_rejected"])["gain"], 0.0)
        self.assertEqual(dict(selected["best_rejected"])["replacement_report"], {"changed": 2, "fixed": 1, "broken": 1})

    def test_deployed_predictions_handles_empty_batches(self) -> None:
        with patch(
            "scripts.probe_character_family_specialists.load_character_model",
            return_value=(torch.nn.Linear(1, 1), ["A"]),
        ):
            predictions, labels = deployed_predictions(
                torch.zeros((0, 1, 32, 32)),
                batch_size=8,
                device=torch.device("cpu"),
            )

        self.assertEqual(predictions.tolist(), [])
        self.assertEqual(labels, ["A"])


if __name__ == "__main__":
    unittest.main()
