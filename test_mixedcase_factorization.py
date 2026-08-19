import unittest

import torch

from scripts.analyze_mixedcase_factorization import (
    evaluate_learned_replacement_gate,
    factorized_predictions,
    folded_identity_targets,
    protected_promotable,
    replacement_gate_features,
    split_indices,
    sweep_factorized_gates,
    type_targets,
)


class MixedcaseFactorizationTests(unittest.TestCase):
    """Focused tests for identity/type factorization diagnostics."""

    def test_type_and_folded_identity_targets_match_mixedcase_layout(self) -> None:
        targets = torch.tensor([0, 9, 10, 35, 36, 61])

        self.assertEqual(type_targets(targets).tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(folded_identity_targets(targets).tolist(), [0, 9, 10, 35, 10, 35])

    def test_factorized_predictions_use_folded_identity_and_type_logits(self) -> None:
        mixed_outputs = torch.zeros((3, 62))
        folded_outputs = torch.zeros((3, 36))
        base_predictions = torch.tensor([10, 10, 36])
        folded_outputs[0, 10] = 4.0
        folded_outputs[1, 11] = 4.0
        folded_outputs[2, 12] = 4.0
        mixed_outputs[0, :10] = -2.0
        mixed_outputs[0, 10:36] = 2.0
        mixed_outputs[0, 36:] = 1.0
        mixed_outputs[1, 10:36] = 4.0
        mixed_outputs[1, 36:] = 1.0
        mixed_outputs[2, 10:36] = 1.0
        mixed_outputs[2, 36:] = 4.0

        predictions, features = factorized_predictions(mixed_outputs, folded_outputs, base_predictions)

        self.assertEqual(predictions.tolist(), [10, 11, 38])
        self.assertEqual(features["type_prediction"].tolist(), [1, 1, 2])

    def test_protected_promotable_requires_all_split_metrics_to_hold(self) -> None:
        baseline = {
            "test_accuracy": 80.0,
            "case_or_ambiguity_aware_test_accuracy": 98.0,
            "digit_test_accuracy": 95.0,
            "upper_test_accuracy": 85.0,
            "lower_test_accuracy": 75.0,
        }
        candidate = {**baseline, "test_accuracy": 80.1}

        self.assertTrue(protected_promotable(candidate, baseline))
        self.assertFalse(protected_promotable({**candidate, "lower_test_accuracy": 74.9}, baseline))

    def test_sweep_factorized_gates_reports_promotable_rows(self) -> None:
        labels = ["0", "A", "a"]
        targets = torch.tensor([1, 2])
        base_predictions = torch.tensor([0, 2])
        mixed_outputs = torch.tensor(
            [
                [0.0, 4.0, 1.0],
                [0.0, 1.0, 4.0],
            ]
        )
        folded_outputs = torch.tensor(
            [
                [0.0, 4.0],
                [0.0, 4.0],
            ]
        )

        report = sweep_factorized_gates(
            mixed_outputs=mixed_outputs,
            folded_outputs=folded_outputs,
            targets=targets,
            labels=labels,
            base_predictions=base_predictions,
            folded_confidences=[0.0],
            folded_margins=[0.0],
            type_confidences=[0.0],
            type_margins=[0.0],
        )

        self.assertEqual(report["factorized_changed"], 1)
        self.assertEqual(report["factorized_fixed"], 1)
        self.assertEqual(report["factorized_broken"], 0)
        self.assertEqual(report["promotable_count"], 1)
        self.assertTrue(report["best"]["promotable"])

    def test_split_indices_partitions_all_samples(self) -> None:
        fit, selection, confirmation = split_indices(20, calibration_ratio=0.4, confirmation_ratio=0.5, seed=123)

        combined = torch.cat([fit, selection, confirmation]).tolist()

        self.assertEqual(sorted(combined), list(range(20)))
        self.assertEqual(len(combined), len(set(combined)))
        self.assertGreater(len(fit), 0)
        self.assertGreater(len(selection), 0)
        self.assertGreater(len(confirmation), 0)

    def test_replacement_gate_features_include_confidences_and_type_one_hot(self) -> None:
        mixed_outputs = torch.zeros((2, 3))
        folded_outputs = torch.zeros((2, 2))
        base_predictions = torch.tensor([0, 1])
        factorized = torch.tensor([1, 2])
        mixed_outputs[0, 0] = 3.0
        mixed_outputs[0, 1] = 2.0
        mixed_outputs[1, 1] = 1.0
        mixed_outputs[1, 2] = 4.0
        features = {
            "folded_confidence": torch.tensor([0.9, 0.8]),
            "folded_margin": torch.tensor([0.5, 0.4]),
            "type_confidence": torch.tensor([0.7, 0.6]),
            "type_margin": torch.tensor([0.3, 0.2]),
            "type_prediction": torch.tensor([1, 2]),
        }

        rows = replacement_gate_features(mixed_outputs, folded_outputs, base_predictions, factorized, features)

        self.assertEqual(tuple(rows.shape), (2, 10))
        self.assertEqual(rows[:, 7:].tolist(), [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_learned_replacement_gate_reports_unavailable_when_no_changes_exist(self) -> None:
        labels = ["0", "A", "a"]
        targets = torch.tensor([1, 1, 1])
        base_predictions = torch.tensor([1, 1, 1])
        mixed_outputs = torch.zeros((3, 3))
        folded_outputs = torch.zeros((3, 2))
        mixed_outputs[:, 1] = 4.0
        folded_outputs[:, 1] = 4.0

        report = evaluate_learned_replacement_gate(
            mixed_outputs=mixed_outputs,
            folded_outputs=folded_outputs,
            targets=targets,
            labels=labels,
            base_predictions=base_predictions,
            calibration_ratio=0.5,
            confirmation_ratio=0.5,
            seed=123,
            epochs=2,
            learning_rate=0.01,
            thresholds=[0.5],
        )

        self.assertFalse(report["available"])
        self.assertEqual(report["reason"], "no_factorized_replacements")


if __name__ == "__main__":
    unittest.main()
