import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.probe_mixedcase_feature_reranker import FamilyProbe
from scripts.probe_mixedcase_feature_reranker import _final_gate_rejection
from scripts.probe_mixedcase_feature_reranker import _fit_tensors
from scripts.probe_mixedcase_feature_reranker import _is_promotable
from scripts.probe_mixedcase_feature_reranker import geometry_features
from scripts.probe_mixedcase_feature_reranker import parse_family_names
from scripts.probe_mixedcase_feature_reranker import parse_source_groups
from scripts.probe_mixedcase_feature_reranker import selected_families
from scripts.probe_mixedcase_feature_reranker import source_group_mask
from scripts.probe_mixedcase_feature_reranker import train_family_probe
from scripts.probe_mixedcase_feature_reranker import run_probe


class MixedcaseFeatureRerankerTests(unittest.TestCase):
    """Regression tests for the mixed-case feature-reranker probe."""

    def test_geometry_features_are_finite(self) -> None:
        """Blank and inked tensors should produce stable finite feature rows."""

        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        images[1, :, 8:20, 10:18] = 1.0

        features = geometry_features(images)

        self.assertEqual(tuple(features.shape), (2, 22))
        self.assertTrue(bool(torch.isfinite(features).all()))

    def test_selected_families_returns_model_label_indices(self) -> None:
        """Family probes should only include labels that exist in the 62-class model."""

        families = selected_families(limit=3)

        self.assertEqual(len(families), 3)
        self.assertTrue(all(len(family) > 1 for family in families))
        self.assertTrue(all(0 <= index < 62 for family in families for index in family))

    def test_selected_families_can_use_explicit_family_names(self) -> None:
        """Roadmap probes should be able to target non-prefix ambiguity families."""

        families = selected_families(family_names=("MNmn", "9qg"))

        self.assertEqual(len(families), 2)
        self.assertEqual(len(families[0]), 4)
        self.assertEqual(len(families[1]), 3)

    def test_parse_family_names_returns_none_for_blank(self) -> None:
        """The CLI should preserve default family behavior when the flag is blank."""

        self.assertIsNone(parse_family_names(""))
        self.assertEqual(parse_family_names("1Iil, 0Oo"), ("1Iil", "0Oo"))

    def test_source_group_mask_filters_current_prediction_groups(self) -> None:
        """Family probes should be able to avoid groups that regress."""

        predictions = torch.tensor([1, 10, 35, 36, 61], dtype=torch.long)

        self.assertEqual(parse_source_groups(" digit, lower "), ("digit", "lower"))
        self.assertEqual(source_group_mask(predictions, ("digit", "lower")).tolist(), [True, False, False, True, True])
        with self.assertRaisesRegex(ValueError, "Unknown source group"):
            parse_source_groups("symbol")

    def test_fit_tensors_appends_capped_extra_roots(self) -> None:
        """Optional adviser data should be capped before joining fit tensors."""

        train_images = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10], dtype=torch.long)
        extra_images = torch.ones((4, 1, 28, 28), dtype=torch.float32)
        extra_targets = torch.tensor([36, 36, 36, 37], dtype=torch.long)

        with patch(
            "scripts.probe_mixedcase_feature_reranker.load_mixedcase_extra_cache",
            return_value=(extra_images, extra_targets),
        ):
            images, targets = _fit_tensors(
                train_images,
                train_targets,
                [Path("extra.pt")],
                extra_samples_per_class=1,
                seed=11,
            )

        self.assertEqual(tuple(images.shape), (3, 1, 28, 28))
        self.assertEqual(torch.bincount(targets, minlength=62)[10].item(), 1)
        self.assertEqual(torch.bincount(targets, minlength=62)[36].item(), 1)
        self.assertEqual(torch.bincount(targets, minlength=62)[37].item(), 1)

    def test_family_probe_defaults_to_linear_model(self) -> None:
        """The original linear probe remains the default adapter."""

        features = torch.randn((20, 5), dtype=torch.float32)
        targets = torch.tensor([10, 11] * 10, dtype=torch.long)

        probe = train_family_probe(features, targets, (10, 11), epochs=1, learning_rate=0.01)

        self.assertIsInstance(probe.model, torch.nn.Linear)

    def test_family_probe_can_train_mlp_adapter(self) -> None:
        """A hidden layer enables a stronger nonlinear family adapter probe."""

        features = torch.randn((20, 5), dtype=torch.float32)
        targets = torch.tensor([10, 11] * 10, dtype=torch.long)

        probe = train_family_probe(features, targets, (10, 11), epochs=1, learning_rate=0.01, hidden_units=4)

        self.assertIsInstance(probe.model, torch.nn.Sequential)
        self.assertIsInstance(probe.model[0], torch.nn.Linear)
        self.assertEqual(probe.model[0].out_features, 4)

    def test_promotable_requires_exact_gain_without_split_regressions(self) -> None:
        """Adapter probes should not look deployable when a protected split falls."""

        base = {
            "test_accuracy": 90.0,
            "case_or_ambiguity_aware_test_accuracy": 98.0,
            "digit_test_accuracy": 96.0,
            "upper_test_accuracy": 88.0,
            "lower_test_accuracy": 75.0,
        }
        candidate = {**base, "test_accuracy": 90.1, "upper_test_accuracy": 87.9}
        safe_candidate = {**base, "test_accuracy": 90.1}

        self.assertFalse(_is_promotable(base, candidate))
        self.assertTrue(_is_promotable(base, safe_candidate))

    def test_final_gate_rejects_protected_test_regression(self) -> None:
        """Family adapters should be rejected when final test split accuracy regresses."""

        base = {
            "test_accuracy": 87.0,
            "case_or_ambiguity_aware_test_accuracy": 98.0,
            "digit_test_accuracy": 95.0,
            "upper_test_accuracy": 85.0,
            "lower_test_accuracy": 73.0,
        }
        upper_regression = {**base, "test_accuracy": 87.2, "upper_test_accuracy": 84.9}
        tiny_gain = {**base, "test_accuracy": 87.005}
        safe = {**base, "test_accuracy": 87.02}

        self.assertEqual(
            _final_gate_rejection(base, upper_regression, min_delta=0.01),
            "final_upper_test_accuracy_regressed",
        )
        self.assertEqual(_final_gate_rejection(base, tiny_gain, min_delta=0.01), "final_delta_below_floor")
        self.assertIsNone(_final_gate_rejection(base, safe, min_delta=0.01))

    def test_run_probe_reports_adapter_shape_and_promotion_state(self) -> None:
        """Probe JSON should expose enough fields for automation to reject regressions."""

        train_images = torch.zeros((6, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10, 11, 10, 11, 10, 11], dtype=torch.long)
        test_images = torch.zeros((4, 1, 28, 28), dtype=torch.float32)
        test_targets = torch.tensor([10, 11, 10, 11], dtype=torch.long)
        outputs = torch.zeros((4, 62), dtype=torch.float32)
        outputs[:, 10] = 1.0

        with (
            patch(
                "scripts.probe_mixedcase_feature_reranker._split_tensors",
                side_effect=[(train_images, train_targets), (test_images, test_targets)],
            ),
            patch(
                "scripts.probe_mixedcase_feature_reranker._model_outputs",
                return_value=(outputs, outputs),
            ),
            patch("scripts.probe_mixedcase_feature_reranker._load_hybrid_artifact", return_value={"enabled": False}),
            patch("scripts.probe_mixedcase_feature_reranker.selected_families", return_value=[]),
        ):
            report = run_probe(
                batch_size=8,
                epochs=1,
                learning_rate=0.01,
                train_sample_limit=None,
                family_limit=None,
                calibration_ratio=0.5,
                min_family_delta=0.0,
                seed=3,
                hidden_units=7,
                family_names=("AB",),
            )

        self.assertEqual(report["hidden_units"], 7)
        self.assertEqual(report["confirmation_ratio"], 0.5)
        self.assertEqual(report["family_names"], ["AB"])
        self.assertEqual(report["source_groups"], ["digit", "upper", "lower"])
        self.assertEqual(report["selection_samples"], 1)
        self.assertEqual(report["confirmation_samples"], 2)
        self.assertEqual(report["test_delta"], 0.0)
        self.assertFalse(report["promotable"])

    def test_run_probe_rejects_adapter_without_confirmation_gain(self) -> None:
        """One calibration win should not be enough to touch final test predictions."""

        train_images = torch.zeros((8, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10, 11, 10, 11, 10, 11, 10, 11], dtype=torch.long)
        test_images = torch.zeros((4, 1, 28, 28), dtype=torch.float32)
        test_targets = torch.tensor([10, 11, 10, 11], dtype=torch.long)
        probe_model = torch.nn.Linear(1, 2)

        def fake_train_family_probe(*_args, **_kwargs):
            return FamilyProbe("AB", (10, 11), probe_model)

        call_count = {"count": 0}

        def fake_apply(predictions, *_args, **_kwargs):
            call_count["count"] += 1
            candidate = predictions.clone()
            if call_count["count"] == 1:
                candidate[:] = 11
            else:
                candidate[:] = 10
            return candidate

        with (
            patch(
                "scripts.probe_mixedcase_feature_reranker._split_tensors",
                side_effect=[(train_images, train_targets), (test_images, test_targets)],
            ),
            patch(
                "scripts.probe_mixedcase_feature_reranker._model_outputs",
                side_effect=lambda images, _batch_size: (
                    torch.zeros((images.shape[0], 62), dtype=torch.float32),
                    torch.zeros((images.shape[0], 36), dtype=torch.float32),
                ),
            ),
            patch("scripts.probe_mixedcase_feature_reranker._load_hybrid_artifact", return_value={"enabled": False}),
            patch(
                "scripts.probe_mixedcase_feature_reranker.hybrid_predictions",
                side_effect=lambda mixed, _folded, _artifact: torch.full((mixed.shape[0],), 10, dtype=torch.long),
            ),
            patch("scripts.probe_mixedcase_feature_reranker.selected_families", return_value=[(10, 11)]),
            patch("scripts.probe_mixedcase_feature_reranker.family_features", return_value=torch.zeros((6, 1))),
            patch("scripts.probe_mixedcase_feature_reranker.train_family_probe", side_effect=fake_train_family_probe),
            patch("scripts.probe_mixedcase_feature_reranker.apply_family_probe", side_effect=fake_apply),
            patch(
                "scripts.probe_mixedcase_feature_reranker._metrics",
                side_effect=[
                    {"test_accuracy": 50.0},
                    {"test_accuracy": 75.0},
                    {"test_accuracy": 50.0},
                    {"test_accuracy": 50.0},
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
                family_limit=None,
                calibration_ratio=0.5,
                min_family_delta=0.01,
                seed=3,
                confirmation_ratio=0.5,
            )

        self.assertEqual(report["families"][0]["rejection_reason"], "confirmation_delta_below_floor")
        self.assertGreater(report["families"][0]["selection_delta"], 0)
        self.assertLessEqual(report["families"][0]["confirmation_delta"], 0)
        self.assertEqual(report["test_delta"], 0.0)

    def test_run_probe_final_rejection_reports_full_split_metrics(self) -> None:
        """Final-gate failures should include before/after split metrics."""

        train_images = torch.zeros((8, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10, 11, 10, 11, 10, 11, 10, 11], dtype=torch.long)
        test_images = torch.zeros((4, 1, 28, 28), dtype=torch.float32)
        test_targets = torch.tensor([10, 11, 10, 11], dtype=torch.long)
        probe_model = torch.nn.Linear(1, 2)
        protected_base = {
            "test_accuracy": 80.0,
            "case_or_ambiguity_aware_test_accuracy": 98.0,
            "digit_test_accuracy": 96.0,
            "upper_test_accuracy": 88.0,
            "lower_test_accuracy": 75.0,
        }
        upper_regressed = {**protected_base, "test_accuracy": 80.2, "upper_test_accuracy": 87.9}

        with (
            patch(
                "scripts.probe_mixedcase_feature_reranker._split_tensors",
                side_effect=[(train_images, train_targets), (test_images, test_targets)],
            ),
            patch(
                "scripts.probe_mixedcase_feature_reranker._model_outputs",
                side_effect=lambda images, _batch_size: (
                    torch.zeros((images.shape[0], 62), dtype=torch.float32),
                    torch.zeros((images.shape[0], 36), dtype=torch.float32),
                ),
            ),
            patch("scripts.probe_mixedcase_feature_reranker._load_hybrid_artifact", return_value={"enabled": False}),
            patch(
                "scripts.probe_mixedcase_feature_reranker.hybrid_predictions",
                side_effect=lambda mixed, _folded, _artifact: torch.full((mixed.shape[0],), 10, dtype=torch.long),
            ),
            patch("scripts.probe_mixedcase_feature_reranker.selected_families", return_value=[(10, 11)]),
            patch("scripts.probe_mixedcase_feature_reranker.family_features", return_value=torch.zeros((6, 1))),
            patch(
                "scripts.probe_mixedcase_feature_reranker.train_family_probe",
                return_value=FamilyProbe("AB", (10, 11), probe_model),
            ),
            patch(
                "scripts.probe_mixedcase_feature_reranker.apply_family_probe",
                side_effect=lambda predictions, *_args, **_kwargs: predictions,
            ),
            patch(
                "scripts.probe_mixedcase_feature_reranker._metrics",
                side_effect=[
                    {"test_accuracy": 50.0},
                    {"test_accuracy": 60.0},
                    {"test_accuracy": 50.0},
                    {"test_accuracy": 60.0},
                    protected_base,
                    upper_regressed,
                    protected_base,
                    protected_base,
                ],
            ),
        ):
            report = run_probe(
                batch_size=8,
                epochs=1,
                learning_rate=0.01,
                train_sample_limit=None,
                family_limit=None,
                calibration_ratio=0.5,
                min_family_delta=0.01,
                seed=3,
                confirmation_ratio=0.5,
            )

        self.assertEqual(report["families"][0]["rejection_reason"], "final_upper_test_accuracy_regressed")
        self.assertEqual(report["families"][0]["before_metrics"]["upper_test_accuracy"], 88.0)
        self.assertEqual(report["families"][0]["after_metrics"]["upper_test_accuracy"], 87.9)


if __name__ == "__main__":
    unittest.main()
