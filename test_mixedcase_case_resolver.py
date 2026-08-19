import unittest

import torch

from scripts.probe_mixedcase_case_resolver import (
    _letter_identity_index,
    apply_case_resolver,
    case_resolver_features,
    oracle_case_predictions,
    parse_threshold_values,
    sweep_case_resolver_thresholds,
    train_case_resolver,
)


class MixedcaseCaseResolverTests(unittest.TestCase):
    """Regression tests for the mixed-case case-resolver probe."""

    def test_letter_identity_index_marks_non_letters(self) -> None:
        """Mixed-case labels should collapse to A-Z identities only for letters."""

        targets = torch.tensor([0, 10, 35, 36, 61], dtype=torch.long)

        identities = _letter_identity_index(targets)

        self.assertEqual(identities.tolist(), [-1, 0, 25, 0, 25])

    def test_case_resolver_features_include_identity_logits_and_geometry(self) -> None:
        """Feature rows should be finite and include one predicted-letter one-hot."""

        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        mixed_outputs = torch.zeros((2, 62), dtype=torch.float32)
        folded_outputs = torch.zeros((2, 36), dtype=torch.float32)
        folded_outputs[:, 10] = 3.0

        features, folded_predictions = case_resolver_features(images, mixed_outputs, folded_outputs)

        self.assertEqual(folded_predictions.tolist(), [10, 10])
        self.assertEqual(tuple(features.shape), (2, 56))
        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertEqual(features[:, :26].sum(dim=1).tolist(), [1.0, 1.0])

    def test_oracle_case_predictions_keeps_folded_identity_with_true_case(self) -> None:
        """Oracle predictions should expose the best case-only resolver ceiling."""

        base_predictions = torch.tensor([10, 10, 11, 5], dtype=torch.long)
        folded_outputs = torch.zeros((4, 36), dtype=torch.float32)
        folded_outputs[0, 10] = 1.0
        folded_outputs[1, 10] = 1.0
        folded_outputs[2, 11] = 1.0
        folded_outputs[3, 5] = 1.0
        targets = torch.tensor([10, 36, 37, 5], dtype=torch.long)

        predictions = oracle_case_predictions(base_predictions, folded_outputs, targets)

        self.assertEqual(predictions.tolist(), [10, 36, 37, 5])

    def test_apply_case_resolver_requires_letter_and_thresholds(self) -> None:
        """The resolver should only replace alphabetic predictions that pass gates."""

        model = torch.nn.Linear(3, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias[:] = torch.tensor([0.0, 5.0])
        base_predictions = torch.tensor([10, 0, 11], dtype=torch.long)
        folded_predictions = torch.tensor([10, 10, 11], dtype=torch.long)
        features = torch.zeros((3, 3), dtype=torch.float32)

        predictions = apply_case_resolver(
            base_predictions,
            features,
            folded_predictions,
            model,
            confidence_threshold=0.9,
            margin_threshold=0.5,
        )

        self.assertEqual(predictions.tolist(), [36, 0, 37])

    def test_parse_threshold_values_rejects_empty_lists(self) -> None:
        """CLI sweeps should require at least one numeric threshold."""

        self.assertEqual(parse_threshold_values("0, 0.5,1"), [0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "At least one"):
            parse_threshold_values(" , ")

    def test_sweep_case_resolver_thresholds_selects_safe_improvement(self) -> None:
        """Threshold sweeps should return the best non-regressing candidate."""

        model = torch.nn.Linear(3, 2)
        with torch.no_grad():
            model.weight[:] = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
            model.bias.zero_()
        base_predictions = torch.tensor([10, 10, 36, 0], dtype=torch.long)
        targets = torch.tensor([10, 36, 36, 0], dtype=torch.long)
        folded_predictions = torch.tensor([10, 10, 10, 10], dtype=torch.long)
        features = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        predictions, metrics, rows = sweep_case_resolver_thresholds(
            base_predictions,
            targets,
            features,
            folded_predictions,
            model,
            confidence_thresholds=[0.0, 0.999],
            margin_thresholds=[0.0],
        )

        self.assertEqual(predictions.tolist(), [10, 36, 36, 0])
        self.assertGreater(metrics["test_accuracy"], rows[1]["metrics"]["test_accuracy"])
        self.assertTrue(rows[0]["safe"])
        self.assertFalse(rows[1]["safe"])

    def test_train_case_resolver_returns_none_without_matching_identity_samples(self) -> None:
        """Training should abstain when folded identity never matches true letters."""

        features = torch.zeros((4, 3), dtype=torch.float32)
        targets = torch.tensor([10, 36, 11, 37], dtype=torch.long)
        folded_predictions = torch.tensor([12, 12, 12, 12], dtype=torch.long)

        self.assertIsNone(
            train_case_resolver(
                features,
                targets,
                folded_predictions,
                hidden_units=0,
                epochs=1,
                learning_rate=0.01,
            )
        )


if __name__ == "__main__":
    unittest.main()
