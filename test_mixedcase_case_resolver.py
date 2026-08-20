import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.probe_mixedcase_case_resolver import (
    _case_target_counts,
    _letter_identity_index,
    _prediction_change_summary,
    _resolver_floor_failures,
    _resolver_metric_deltas,
    _resolver_objective,
    apply_case_resolver,
    case_resolver_features,
    oracle_case_predictions,
    parse_threshold_values,
    prepare_case_resolver_data,
    select_confirm_case_resolver_thresholds,
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

    def test_case_resolver_features_can_append_normalized_embeddings(self) -> None:
        """Optional CNN activations should become normalized case-resolver evidence."""

        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        mixed_outputs = torch.zeros((2, 62), dtype=torch.float32)
        folded_outputs = torch.zeros((2, 36), dtype=torch.float32)
        folded_outputs[:, 10] = 3.0
        embeddings = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32)

        features, _folded_predictions = case_resolver_features(
            images,
            mixed_outputs,
            folded_outputs,
            embeddings,
        )

        self.assertEqual(tuple(features.shape), (2, 58))
        self.assertTrue(torch.allclose(features[:, -2:].norm(dim=1), torch.ones(2)))

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
        self.assertEqual(rows[0]["changes"], {"changed": 1, "fixed": 1, "broken": 0, "still_wrong_changed": 0})

    def test_resolver_objective_can_score_balanced_group_floor(self) -> None:
        """Balanced selection should prefer the weakest protected metric."""

        metrics = {
            "test_accuracy": 90.0,
            "case_or_ambiguity_aware_test_accuracy": 99.0,
            "digit_test_accuracy": 96.0,
            "upper_test_accuracy": 88.0,
            "lower_test_accuracy": 75.0,
        }

        self.assertEqual(_resolver_objective(metrics, "exact"), 90.0)
        self.assertEqual(_resolver_objective(metrics, "balanced"), 75.0)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _resolver_objective(metrics, "mystery")

    def test_resolver_diagnostics_report_metric_deltas_and_floor_failures(self) -> None:
        """Resolver sweeps should explain which protected metric blocked promotion."""

        base = {
            "test_accuracy": 80.0,
            "case_or_ambiguity_aware_test_accuracy": 98.0,
            "digit_test_accuracy": 95.0,
            "upper_test_accuracy": 84.0,
            "lower_test_accuracy": 73.0,
        }
        candidate = {
            "test_accuracy": 81.0,
            "case_or_ambiguity_aware_test_accuracy": 98.1,
            "digit_test_accuracy": 94.9,
            "upper_test_accuracy": 84.0,
            "lower_test_accuracy": 72.5,
        }

        deltas = _resolver_metric_deltas(base, candidate)

        self.assertEqual(deltas["test_accuracy"], 1.0)
        self.assertAlmostEqual(deltas["digit_test_accuracy"], -0.1)
        self.assertEqual(
            _resolver_floor_failures(base, candidate),
            ["digit_test_accuracy", "lower_test_accuracy"],
        )

    def test_prediction_change_summary_counts_fix_break_balance(self) -> None:
        """Resolver diagnostics should expose whether changes help or hurt."""

        before = torch.tensor([10, 10, 36, 37, 11], dtype=torch.long)
        after = torch.tensor([10, 36, 10, 36, 12], dtype=torch.long)
        targets = torch.tensor([10, 36, 36, 37, 13], dtype=torch.long)

        self.assertEqual(
            _prediction_change_summary(before, after, targets),
            {
                "changed": 4,
                "fixed": 1,
                "broken": 2,
                "still_wrong_changed": 1,
            },
        )

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

    def test_train_case_resolver_rejects_unknown_class_weighting(self) -> None:
        """Class weighting should fail fast for unsupported modes."""

        features = torch.zeros((20, 3), dtype=torch.float32)
        targets = torch.tensor([10, 36] * 10, dtype=torch.long)
        folded_predictions = torch.tensor([10, 10] * 10, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "Unsupported"):
            train_case_resolver(
                features,
                targets,
                folded_predictions,
                hidden_units=0,
                epochs=1,
                learning_rate=0.01,
                class_weighting="weird",
            )

    def test_case_target_counts_reports_eligible_upper_lower_samples(self) -> None:
        """Diagnostics should count only samples whose folded identity is right."""

        targets = torch.tensor([10, 36, 11, 37, 5], dtype=torch.long)
        folded_predictions = torch.tensor([10, 10, 12, 11, 5], dtype=torch.long)

        self.assertEqual(_case_target_counts(targets, folded_predictions), {"upper": 1, "lower": 2})

    def test_prepare_data_can_use_external_test_tensor_pack(self) -> None:
        """Resolver probes should allow an external held-out tensor pack."""

        train_images = torch.zeros((20, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10, 36] * 10, dtype=torch.long)
        test_images = torch.ones((3, 1, 28, 28), dtype=torch.float32)
        test_targets = torch.tensor([10, 36, 0], dtype=torch.long)
        mixed_outputs = torch.zeros((20, 62), dtype=torch.float32)
        folded_outputs = torch.zeros((20, 36), dtype=torch.float32)
        mixed_outputs[:, 10] = 1.0
        folded_outputs[:, 10] = 1.0
        test_mixed_outputs = torch.zeros((3, 62), dtype=torch.float32)
        test_folded_outputs = torch.zeros((3, 36), dtype=torch.float32)
        test_mixed_outputs[:, 10] = 1.0
        test_folded_outputs[:, 10] = 1.0

        def fake_split(train: bool, sample_limit: int | None):
            return (train_images, train_targets) if train else self.fail("default test split should not load")

        def fake_outputs(images: torch.Tensor, batch_size: int):
            if int(images.shape[0]) == 3:
                return test_mixed_outputs, test_folded_outputs
            return mixed_outputs[: int(images.shape[0])], folded_outputs[: int(images.shape[0])]

        with (
            patch("scripts.probe_mixedcase_case_resolver._split_tensors", side_effect=fake_split),
            patch("scripts.probe_mixedcase_case_resolver.load_tensor_pack", return_value=(test_images, test_targets)),
            patch("scripts.probe_mixedcase_case_resolver._model_outputs", side_effect=fake_outputs),
            patch("scripts.probe_mixedcase_case_resolver._load_hybrid_artifact", return_value={}),
            patch(
                "scripts.probe_mixedcase_case_resolver.hybrid_predictions",
                side_effect=lambda mixed, folded, artifact: mixed.argmax(dim=1),
            ),
        ):
            data = prepare_case_resolver_data(
                batch_size=4,
                train_sample_limit=None,
                seed=1,
                test_tensor_path=Path("rough-validation.pt"),
            )

        self.assertEqual(data.test_samples, 3)
        self.assertEqual(data.test_tensor_path, Path("rough-validation.pt"))
        self.assertEqual(data.test_targets.tolist(), [10, 36, 0])

    def test_select_confirm_case_resolver_rejects_confirmation_regression(self) -> None:
        """Selection wins should not be used on test when confirmation disagrees."""

        model = torch.nn.Linear(3, 2)
        with torch.no_grad():
            model.weight[:] = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
            model.bias.zero_()
        selection_predictions = torch.tensor([10, 10, 36, 0], dtype=torch.long)
        selection_targets = torch.tensor([10, 36, 36, 0], dtype=torch.long)
        confirmation_predictions = torch.tensor([10, 10, 36, 0], dtype=torch.long)
        confirmation_targets = torch.tensor([10, 10, 36, 0], dtype=torch.long)
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

        selected, confirmation, rows = select_confirm_case_resolver_thresholds(
            selection_predictions,
            selection_targets,
            features,
            folded_predictions,
            confirmation_predictions,
            confirmation_targets,
            features,
            folded_predictions,
            model,
            confidence_thresholds=[0.0],
            margin_thresholds=[0.0],
        )

        self.assertIsNone(selected)
        self.assertIsNotNone(confirmation)
        self.assertFalse(confirmation["safe"])
        self.assertEqual(len(rows), 1)

    def test_select_confirm_case_resolver_returns_confirmed_threshold(self) -> None:
        """Confirmed selection rows should preserve the selected threshold values."""

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

        selected, confirmation, _rows = select_confirm_case_resolver_thresholds(
            base_predictions,
            targets,
            features,
            folded_predictions,
            base_predictions,
            targets,
            features,
            folded_predictions,
            model,
            confidence_thresholds=[0.0],
            margin_thresholds=[0.0],
        )

        self.assertIsNotNone(selected)
        self.assertIsNotNone(confirmation)
        self.assertTrue(confirmation["safe"])
        self.assertEqual(selected["confidence_threshold"], 0.0)

    def test_select_confirm_case_resolver_checks_later_safe_rows(self) -> None:
        """A rejected first choice should not hide another confirmed threshold."""

        model = torch.nn.Linear(3, 2)
        with torch.no_grad():
            model.weight[:] = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
            model.bias.zero_()
        selection_predictions = torch.tensor([10, 10, 36, 36], dtype=torch.long)
        selection_targets = torch.tensor([10, 36, 36, 36], dtype=torch.long)
        confirmation_predictions = torch.tensor([10, 10, 10, 36], dtype=torch.long)
        confirmation_targets = torch.tensor([10, 10, 36, 36], dtype=torch.long)
        folded_predictions = torch.tensor([10, 10, 10, 10], dtype=torch.long)
        selection_features = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.7, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        confirmation_features = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.7, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        selected, confirmation, rows = select_confirm_case_resolver_thresholds(
            selection_predictions,
            selection_targets,
            selection_features,
            folded_predictions,
            confirmation_predictions,
            confirmation_targets,
            confirmation_features,
            folded_predictions,
            model,
            confidence_thresholds=[0.0, 0.95],
            margin_thresholds=[0.0],
        )

        self.assertEqual(len(rows), 2)
        self.assertIsNotNone(selected)
        self.assertIsNotNone(confirmation)
        self.assertTrue(confirmation["safe"])
        self.assertEqual(selected["confidence_threshold"], 0.95)


if __name__ == "__main__":
    unittest.main()
