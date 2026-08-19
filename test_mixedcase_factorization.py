import unittest

import torch

from scripts.analyze_mixedcase_factorization import (
    factorized_predictions,
    folded_identity_targets,
    protected_promotable,
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


if __name__ == "__main__":
    unittest.main()
