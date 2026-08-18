import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.probe_mixedcase_feature_reranker import FamilyProbe
from scripts.probe_mixedcase_residual_clusters import (
    DEFAULT_CLUSTERS,
    _gate_metrics,
    _split_calibration,
    cluster_indices,
    parse_clusters,
    run_probe,
)


class MixedcaseResidualClusterTests(unittest.TestCase):
    """Regression tests for non-family mixed-case residual probes."""

    def test_cluster_indices_preserve_requested_order(self) -> None:
        self.assertEqual(cluster_indices("6bG"), (6, 37, 16))
        self.assertEqual(cluster_indices("2a"), (2, 36))

    def test_parse_clusters_uses_defaults_for_blank_input(self) -> None:
        self.assertEqual(parse_clusters(""), DEFAULT_CLUSTERS)
        self.assertEqual(parse_clusters("6bG, 2a"), ("6bG", "2a"))

    def test_split_calibration_reserves_confirmation_holdout(self) -> None:
        targets = torch.arange(10)

        fit, selection, confirmation = _split_calibration(
            targets,
            calibration_ratio=0.4,
            confirmation_ratio=0.5,
            seed=7,
        )

        self.assertEqual(int(fit.numel()), 6)
        self.assertEqual(int(selection.numel()), 2)
        self.assertEqual(int(confirmation.numel()), 2)

    def test_gate_metrics_rejects_protected_split_regression(self) -> None:
        before = {
            "test_accuracy": 80.0,
            "case_or_ambiguity_aware_test_accuracy": 98.0,
            "digit_test_accuracy": 96.0,
            "upper_test_accuracy": 88.0,
            "lower_test_accuracy": 75.0,
        }
        after = {**before, "test_accuracy": 80.1, "upper_test_accuracy": 87.9}

        passed, reason, delta = _gate_metrics(before, after, min_delta=0.0)

        self.assertFalse(passed)
        self.assertEqual(reason, "upper_test_accuracy_regressed")
        self.assertGreater(delta, 0)

    def test_run_probe_rejects_cluster_without_confirmation_gain(self) -> None:
        train_images = torch.zeros((8, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([6, 37, 6, 37, 6, 37, 6, 37], dtype=torch.long)
        test_images = torch.zeros((4, 1, 28, 28), dtype=torch.float32)
        test_targets = torch.tensor([6, 37, 6, 37], dtype=torch.long)
        probe_model = torch.nn.Linear(1, 2)

        def fake_train_family_probe(*_args, **_kwargs):
            return FamilyProbe("6b", (6, 37), probe_model)

        call_count = {"count": 0}

        def fake_apply(predictions, *_args, **_kwargs):
            call_count["count"] += 1
            candidate = predictions.clone()
            if call_count["count"] == 1:
                candidate[:] = 37
            else:
                candidate[:] = 6
            return candidate

        with (
            patch(
                "scripts.probe_mixedcase_residual_clusters._split_tensors",
                side_effect=[(train_images, train_targets), (test_images, test_targets)],
            ),
            patch(
                "scripts.probe_mixedcase_residual_clusters._model_outputs",
                side_effect=lambda images, _batch_size: (
                    torch.zeros((images.shape[0], 62), dtype=torch.float32),
                    torch.zeros((images.shape[0], 36), dtype=torch.float32),
                ),
            ),
            patch(
                "scripts.probe_mixedcase_residual_clusters._fit_tensors",
                side_effect=lambda images, targets, *_args: (images, targets),
            ),
            patch("scripts.probe_mixedcase_residual_clusters._load_hybrid_artifact", return_value={"enabled": False}),
            patch(
                "scripts.probe_mixedcase_residual_clusters.hybrid_predictions",
                side_effect=lambda mixed, _folded, _artifact: torch.full((mixed.shape[0],), 6, dtype=torch.long),
            ),
            patch("scripts.probe_mixedcase_residual_clusters.family_features", return_value=torch.zeros((6, 1))),
            patch(
                "scripts.probe_mixedcase_residual_clusters.train_family_probe",
                side_effect=fake_train_family_probe,
            ),
            patch("scripts.probe_mixedcase_residual_clusters.apply_family_probe", side_effect=fake_apply),
            patch(
                "scripts.probe_mixedcase_residual_clusters._metrics",
                side_effect=[
                    {
                        "test_accuracy": 50.0,
                        "case_or_ambiguity_aware_test_accuracy": 98.0,
                        "digit_test_accuracy": 96.0,
                        "upper_test_accuracy": 88.0,
                        "lower_test_accuracy": 75.0,
                    },
                    {
                        "test_accuracy": 75.0,
                        "case_or_ambiguity_aware_test_accuracy": 98.0,
                        "digit_test_accuracy": 96.0,
                        "upper_test_accuracy": 88.0,
                        "lower_test_accuracy": 75.0,
                    },
                    {
                        "test_accuracy": 50.0,
                        "case_or_ambiguity_aware_test_accuracy": 98.0,
                        "digit_test_accuracy": 96.0,
                        "upper_test_accuracy": 88.0,
                        "lower_test_accuracy": 75.0,
                    },
                    {
                        "test_accuracy": 50.0,
                        "case_or_ambiguity_aware_test_accuracy": 98.0,
                        "digit_test_accuracy": 96.0,
                        "upper_test_accuracy": 88.0,
                        "lower_test_accuracy": 75.0,
                    },
                    {
                        "test_accuracy": 80.0,
                        "case_or_ambiguity_aware_test_accuracy": 98.0,
                        "digit_test_accuracy": 96.0,
                        "upper_test_accuracy": 88.0,
                        "lower_test_accuracy": 75.0,
                    },
                    {
                        "test_accuracy": 80.0,
                        "case_or_ambiguity_aware_test_accuracy": 98.0,
                        "digit_test_accuracy": 96.0,
                        "upper_test_accuracy": 88.0,
                        "lower_test_accuracy": 75.0,
                    },
                ],
            ),
        ):
            report = run_probe(
                batch_size=8,
                epochs=1,
                learning_rate=0.01,
                train_sample_limit=None,
                clusters=("6b",),
                calibration_ratio=0.5,
                confirmation_ratio=0.5,
                min_cluster_delta=0.01,
                seed=3,
                extra_roots=[Path("unused")],
                hidden_units=4,
            )

        self.assertEqual(report["clusters"][0]["rejection_reason"], "confirmation_test_delta_below_floor")
        self.assertGreater(report["clusters"][0]["selection_delta"], 0)
        self.assertLessEqual(report["clusters"][0]["confirmation_delta"], 0)
        self.assertEqual(report["test_delta"], 0.0)
        self.assertEqual(report["hidden_units"], 4)


if __name__ == "__main__":
    unittest.main()
