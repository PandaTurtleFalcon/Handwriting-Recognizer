import unittest
from unittest.mock import patch

import torch

from scripts.probe_character_family_reranker import (
    CharacterFamilyProbe,
    _gate_metrics,
    _split_calibration,
    apply_family_probe,
    family_indices,
    geometry_features,
    parse_label_groups,
    parse_families,
    run_probe,
)


class CharacterFamilyRerankerTests(unittest.TestCase):
    """Regression tests for the character-family reranker probe."""

    def test_geometry_features_are_finite_for_blank_and_ink(self) -> None:
        images = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
        images[1, :, 8:24, 12:20] = 1.0

        features = geometry_features(images)

        self.assertEqual(tuple(features.shape), (2, 22))
        self.assertTrue(bool(torch.isfinite(features).all()))

    def test_family_indices_keep_requested_order(self) -> None:
        labels = ["!", "1", "I", "l", "i", "|", "/", "O"]

        self.assertEqual(family_indices("1Ili|!/", labels), (1, 2, 3, 4, 5, 0, 6))

    def test_parse_families_uses_defaults_for_blank(self) -> None:
        self.assertIn("1Ili|!/", parse_families(""))
        self.assertEqual(parse_families("1Ili,0Oo"), ("1Ili", "0Oo"))

    def test_parse_label_groups_rejects_unknown_groups(self) -> None:
        self.assertEqual(parse_label_groups("letter,punctuation"), ("letter", "punctuation"))
        with self.assertRaisesRegex(ValueError, "Unknown label"):
            parse_label_groups("word")

    def test_apply_family_probe_can_skip_digit_sources(self) -> None:
        labels = ["1", "I", "l"]
        predictions = torch.tensor([0, 1], dtype=torch.long)
        images = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
        outputs = torch.zeros((2, 3), dtype=torch.float32)

        class AlwaysL(torch.nn.Module):
            def forward(self, features: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros((features.shape[0], 3), dtype=torch.float32)
                logits[:, 2] = 1.0
                return logits

        reranked = apply_family_probe(
            predictions,
            images,
            outputs,
            CharacterFamilyProbe("1Il", (0, 1, 2), AlwaysL()),
            labels,
            source_groups=("letter",),
        )

        self.assertEqual(reranked.tolist(), [0, 2])

    def test_split_calibration_reserves_confirmation_holdout(self) -> None:
        fit, selection, confirmation = _split_calibration(
            torch.arange(10),
            calibration_ratio=0.4,
            confirmation_ratio=0.5,
            seed=7,
        )

        self.assertEqual(int(fit.numel()), 6)
        self.assertEqual(int(selection.numel()), 2)
        self.assertEqual(int(confirmation.numel()), 2)

    def test_gate_metrics_rejects_letter_regression(self) -> None:
        before = {
            "validation_accuracy": 94.0,
            "ambiguity_aware_validation_accuracy": 99.0,
            "digit_validation_accuracy": 95.0,
            "letter_validation_accuracy": 93.0,
            "punctuation_validation_accuracy": 96.0,
        }
        after = {**before, "validation_accuracy": 94.1, "letter_validation_accuracy": 92.9}

        passed, reason, delta = _gate_metrics(before, after, min_delta=0.0)

        self.assertFalse(passed)
        self.assertEqual(reason, "letter_validation_accuracy_regressed")
        self.assertGreater(delta, 0)

    def test_run_probe_rejects_family_without_confirmation_gain(self) -> None:
        labels = ["!", "1"]
        images = torch.zeros((12, 1, 32, 32), dtype=torch.float32)
        targets = torch.tensor([0, 1] * 6, dtype=torch.long)
        outputs = torch.zeros((12, 2), dtype=torch.float32)

        class FixedModel(torch.nn.Module):
            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                return torch.zeros((batch.shape[0], 2), dtype=torch.float32)

        def fake_apply(predictions, *_args, **_kwargs):
            fake_apply.calls += 1
            candidate = predictions.clone()
            if fake_apply.calls == 1:
                candidate[:] = 1
            else:
                candidate[:] = 0
            return candidate

        fake_apply.calls = 0

        with (
            patch("scripts.probe_character_family_reranker.get_device", return_value=torch.device("cpu")),
            patch("scripts.probe_character_family_reranker.load_character_model", return_value=(FixedModel(), labels)),
            patch("scripts.probe_character_family_reranker._character_tensors", return_value=(images, targets, labels)),
            patch(
                "scripts.probe_character_family_reranker.stratified_split_indices",
                return_value=([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11]),
            ),
            patch("scripts.probe_character_family_reranker._model_outputs", return_value=outputs[:4]),
            patch("scripts.probe_character_family_reranker.family_features", return_value=torch.zeros((6, 1))),
            patch(
                "scripts.probe_character_family_reranker.train_family_probe",
                return_value=CharacterFamilyProbe("!1", (0, 1), torch.nn.Linear(1, 2)),
            ),
            patch("scripts.probe_character_family_reranker.apply_family_probe", side_effect=fake_apply),
            patch(
                "scripts.probe_character_family_reranker._metrics",
                side_effect=[
                    {
                        "validation_accuracy": 50.0,
                        "ambiguity_aware_validation_accuracy": 99.0,
                        "digit_validation_accuracy": 95.0,
                        "letter_validation_accuracy": 93.0,
                        "punctuation_validation_accuracy": 96.0,
                    },
                    {
                        "validation_accuracy": 75.0,
                        "ambiguity_aware_validation_accuracy": 99.0,
                        "digit_validation_accuracy": 95.0,
                        "letter_validation_accuracy": 93.0,
                        "punctuation_validation_accuracy": 96.0,
                    },
                    {
                        "validation_accuracy": 50.0,
                        "ambiguity_aware_validation_accuracy": 99.0,
                        "digit_validation_accuracy": 95.0,
                        "letter_validation_accuracy": 93.0,
                        "punctuation_validation_accuracy": 96.0,
                    },
                    {
                        "validation_accuracy": 50.0,
                        "ambiguity_aware_validation_accuracy": 99.0,
                        "digit_validation_accuracy": 95.0,
                        "letter_validation_accuracy": 93.0,
                        "punctuation_validation_accuracy": 96.0,
                    },
                    {
                        "validation_accuracy": 94.0,
                        "ambiguity_aware_validation_accuracy": 99.0,
                        "digit_validation_accuracy": 95.0,
                        "letter_validation_accuracy": 93.0,
                        "punctuation_validation_accuracy": 96.0,
                    },
                    {
                        "validation_accuracy": 94.0,
                        "ambiguity_aware_validation_accuracy": 99.0,
                        "digit_validation_accuracy": 95.0,
                        "letter_validation_accuracy": 93.0,
                        "punctuation_validation_accuracy": 96.0,
                    },
                ],
            ),
        ):
            report = run_probe(
                batch_size=4,
                epochs=1,
                learning_rate=0.01,
                families=("!1",),
                calibration_ratio=0.5,
                confirmation_ratio=0.5,
                min_family_delta=0.01,
                seed=3,
                hidden_units=4,
                source_groups=None,
            )

        self.assertEqual(report["families"][0]["rejection_reason"], "confirmation_validation_delta_below_floor")
        self.assertEqual(report["validation_delta"], 0.0)
        self.assertFalse(report["promotable"])


if __name__ == "__main__":
    unittest.main()
