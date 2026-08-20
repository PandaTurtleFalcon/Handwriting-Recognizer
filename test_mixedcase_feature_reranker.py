import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.probe_mixedcase_feature_reranker import FamilyProbe
from scripts.probe_mixedcase_feature_reranker import base_prediction_uncertainty_mask
from scripts.probe_mixedcase_feature_reranker import _final_gate_rejection
from scripts.probe_mixedcase_feature_reranker import _fit_tensors
from scripts.probe_mixedcase_feature_reranker import _is_promotable
from scripts.probe_mixedcase_feature_reranker import _split_tensors
from scripts.probe_mixedcase_feature_reranker import geometry_features
from scripts.probe_mixedcase_feature_reranker import family_features
from scripts.probe_mixedcase_feature_reranker import parse_family_names
from scripts.probe_mixedcase_feature_reranker import parse_source_groups
from scripts.probe_mixedcase_feature_reranker import pixel_features
from scripts.probe_mixedcase_feature_reranker import prepare_feature_probe_data
from scripts.probe_mixedcase_feature_reranker import load_or_prepare_feature_probe_data
from scripts.probe_mixedcase_feature_reranker import protected_balanced_score
from scripts.probe_mixedcase_feature_reranker import selected_families
from scripts.probe_mixedcase_feature_reranker import source_group_mask
from scripts.probe_mixedcase_feature_reranker import train_family_probe
from scripts.probe_mixedcase_feature_reranker import run_probe
from scripts.probe_mixedcase_feature_reranker import apply_family_probe
from scripts.probe_mixedcase_feature_reranker import merge_family_probe_artifacts


class MixedcaseFeatureRerankerTests(unittest.TestCase):
    """Regression tests for the mixed-case feature-reranker probe."""

    def test_geometry_features_are_finite(self) -> None:
        """Blank and inked tensors should produce stable finite feature rows."""

        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        images[1, :, 8:20, 10:18] = 1.0

        features = geometry_features(images)

        self.assertEqual(tuple(features.shape), (2, 22))
        self.assertTrue(bool(torch.isfinite(features).all()))

    def test_split_tensors_cache_reuses_loaded_dataset_tensors(self) -> None:
        """Sweeps should not reload the same large train/test tensors per row."""

        mnist_images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        mnist_targets = torch.tensor([0, 1], dtype=torch.long)
        byclass_images = torch.ones((2, 1, 28, 28), dtype=torch.float32)
        byclass_targets = torch.tensor([10, 11], dtype=torch.long)
        _split_tensors.cache_clear()
        try:
            with (
                patch(
                    "scripts.probe_mixedcase_feature_reranker.build_or_load_mnist_cache",
                    return_value=(mnist_images, mnist_targets),
                ) as mnist_loader,
                patch(
                    "scripts.probe_mixedcase_feature_reranker.build_or_load_emnist_byclass_mixedcase_cache",
                    return_value=(byclass_images, byclass_targets),
                ) as byclass_loader,
            ):
                first_images, first_targets = _split_tensors(train=True, sample_limit=None)
                second_images, second_targets = _split_tensors(train=True, sample_limit=None)
        finally:
            _split_tensors.cache_clear()

        self.assertIs(first_images, second_images)
        self.assertIs(first_targets, second_targets)
        self.assertEqual(mnist_loader.call_count, 1)
        self.assertEqual(byclass_loader.call_count, 1)

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

    def test_family_features_can_include_digit_specialist_outputs(self) -> None:
        """Digit logits should be optional features for digit-sensitive families."""

        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        mixed = torch.zeros((2, 62), dtype=torch.float32)
        folded = torch.zeros((2, 36), dtype=torch.float32)
        digit = torch.zeros((2, 10), dtype=torch.float32)

        base = family_features(images, mixed, folded, (1, 18, 47))
        enriched = family_features(images, mixed, folded, (1, 18, 47), digit)

        self.assertEqual(enriched.shape[0], base.shape[0])
        self.assertEqual(enriched.shape[1], base.shape[1] + 22)

    def test_family_features_can_include_pixel_sketch(self) -> None:
        """Small pixel sketches should be optional shape evidence for rerankers."""

        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        mixed = torch.zeros((2, 62), dtype=torch.float32)
        folded = torch.zeros((2, 36), dtype=torch.float32)

        pixels = pixel_features(images, size=12)
        base = family_features(images, mixed, folded, (10, 36))
        enriched = family_features(images, mixed, folded, (10, 36), include_pixel_features=True)

        self.assertEqual(tuple(pixels.shape), (2, 144))
        self.assertEqual(enriched.shape[0], base.shape[0])
        self.assertEqual(enriched.shape[1], base.shape[1] + 144)

    def test_family_features_can_include_learned_embeddings(self) -> None:
        """CNN penultimate activations should be optional reranker evidence."""

        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        mixed = torch.zeros((2, 62), dtype=torch.float32)
        folded = torch.zeros((2, 36), dtype=torch.float32)
        embeddings = torch.ones((2, 5), dtype=torch.float32)

        base = family_features(images, mixed, folded, (10, 36))
        enriched = family_features(images, mixed, folded, (10, 36), embedding_outputs=embeddings)

        self.assertEqual(enriched.shape[0], base.shape[0])
        self.assertEqual(enriched.shape[1], base.shape[1] + 5)

    def test_apply_family_probe_respects_probe_confidence_gate(self) -> None:
        """Low-confidence family probe predictions should abstain."""

        model = torch.nn.Linear(28, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias[:] = torch.tensor([0.0, 0.1])
        probe = FamilyProbe("AB", (10, 11), model)
        predictions = torch.tensor([10], dtype=torch.long)
        images = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        mixed = torch.zeros((1, 62), dtype=torch.float32)
        folded = torch.zeros((1, 36), dtype=torch.float32)

        kept = apply_family_probe(predictions, images, mixed, folded, probe, probe_confidence=0.9)
        changed = apply_family_probe(predictions, images, mixed, folded, probe, probe_confidence=0.0)

        self.assertEqual(kept.tolist(), [10])
        self.assertEqual(changed.tolist(), [11])

    def test_base_prediction_uncertainty_mask_can_require_low_confidence(self) -> None:
        """Family probes should be able to touch only uncertain base predictions."""

        mixed = torch.zeros((2, 62), dtype=torch.float32)
        mixed[0, 10] = 8.0
        mixed[1, 10] = 0.2
        mixed[1, 11] = 0.1
        predictions = torch.tensor([10, 10], dtype=torch.long)

        mask = base_prediction_uncertainty_mask(mixed, predictions, confidence_max=0.2, margin_max=0.1)

        self.assertEqual(mask.tolist(), [False, True])

    def test_apply_family_probe_respects_base_confidence_gate(self) -> None:
        """High-confidence base predictions should be protected when requested."""

        model = torch.nn.Linear(28, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias[:] = torch.tensor([0.0, 5.0])
        probe = FamilyProbe("AB", (10, 11), model)
        predictions = torch.tensor([10, 10], dtype=torch.long)
        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        mixed = torch.zeros((2, 62), dtype=torch.float32)
        mixed[0, 10] = 8.0
        mixed[1, 10] = 0.2
        mixed[1, 11] = 0.1
        folded = torch.zeros((2, 36), dtype=torch.float32)

        gated = apply_family_probe(
            predictions,
            images,
            mixed,
            folded,
            probe,
            base_confidence_max=0.2,
            base_margin_max=0.1,
        )

        self.assertEqual(gated.tolist(), [10, 11])

    def test_apply_family_probe_can_protect_digit_specialist_agreement(self) -> None:
        """Digit-like predictions should survive when the digit specialist is confident."""

        model = torch.nn.Linear(50, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias[:] = torch.tensor([0.0, 5.0])
        probe = FamilyProbe("0O", (0, 24), model)
        predictions = torch.tensor([0, 0], dtype=torch.long)
        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        mixed = torch.zeros((2, 62), dtype=torch.float32)
        folded = torch.zeros((2, 36), dtype=torch.float32)
        digit_outputs = torch.zeros((2, 10), dtype=torch.float32)
        digit_outputs[0, 0] = 8.0
        digit_outputs[1, 0] = 0.1

        protected = apply_family_probe(
            predictions,
            images,
            mixed,
            folded,
            probe,
            digit_outputs=digit_outputs,
            digit_protect_confidence=0.9,
        )

        self.assertEqual(protected.tolist(), [0, 24])

    def test_apply_family_probe_can_protect_confident_uppercase_predictions(self) -> None:
        """Confident current uppercase predictions should be protected when requested."""

        model = torch.nn.Linear(28, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias[:] = torch.tensor([0.0, 5.0])
        probe = FamilyProbe("Oo", (24, 50), model)
        predictions = torch.tensor([24, 24], dtype=torch.long)
        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        mixed = torch.zeros((2, 62), dtype=torch.float32)
        mixed[0, 24] = 8.0
        mixed[1, 24] = 0.1
        folded = torch.zeros((2, 36), dtype=torch.float32)

        protected = apply_family_probe(
            predictions,
            images,
            mixed,
            folded,
            probe,
            upper_protect_confidence=0.9,
        )

        self.assertEqual(protected.tolist(), [24, 50])

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

    def test_family_probe_can_cap_and_minibatch_training_samples(self) -> None:
        """Large feature probes can run bounded minibatch training."""

        features = torch.randn((80, 5), dtype=torch.float32)
        targets = torch.tensor([10, 11] * 40, dtype=torch.long)

        probe = train_family_probe(
            features,
            targets,
            (10, 11),
            epochs=2,
            learning_rate=0.01,
            max_train_samples=16,
            mini_batch_size=4,
            seed=99,
        )

        self.assertIsNotNone(probe)

    def test_family_probe_rejects_caps_below_minimum_family_coverage(self) -> None:
        """Sample caps should not train undersized family adapters."""

        features = torch.randn((80, 5), dtype=torch.float32)
        targets = torch.tensor([10, 11] * 40, dtype=torch.long)

        probe = train_family_probe(
            features,
            targets,
            (10, 11),
            epochs=1,
            learning_rate=0.01,
            max_train_samples=15,
            mini_batch_size=4,
            seed=99,
        )

        self.assertIsNone(probe)

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
        target_safe_candidate = {**base, "test_accuracy": 90.1, "digit_test_accuracy": 95.5}

        self.assertFalse(_is_promotable(base, candidate))
        self.assertTrue(_is_promotable(base, safe_candidate))
        self.assertTrue(_is_promotable({**base, "digit_test_accuracy": 96.0}, target_safe_candidate, min_digit=95.0))

    def test_protected_balanced_score_returns_weakest_required_metric(self) -> None:
        """Balanced ranking should follow the weakest protected split."""

        metrics = {
            "test_accuracy": 90.0,
            "case_or_ambiguity_aware_test_accuracy": 98.0,
            "digit_test_accuracy": 96.0,
            "upper_test_accuracy": 88.0,
            "lower_test_accuracy": 75.0,
        }

        self.assertEqual(protected_balanced_score(metrics), 75.0)

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
        above_target_digit = {**base, "test_accuracy": 87.02, "digit_test_accuracy": 95.5}

        self.assertEqual(
            _final_gate_rejection(base, upper_regression, min_delta=0.01),
            "final_upper_test_accuracy_regressed",
        )
        self.assertEqual(_final_gate_rejection(base, tiny_gain, min_delta=0.01), "final_delta_below_floor")
        self.assertIsNone(_final_gate_rejection(base, safe, min_delta=0.01))
        self.assertIsNone(
            _final_gate_rejection(
                {**base, "digit_test_accuracy": 96.0},
                above_target_digit,
                min_delta=0.01,
                min_digit=95.0,
            )
        )

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
        self.assertFalse(report["include_digit_features"])
        self.assertIsNone(report["max_probe_train_samples"])
        self.assertIsNone(report["mini_batch_size"])
        self.assertEqual(
            report["minimum_gates"],
            {
                "case_or_ambiguity_aware_test_accuracy": None,
                "digit_test_accuracy": None,
                "upper_test_accuracy": None,
                "lower_test_accuracy": None,
            },
        )
        self.assertEqual(
            report["probe_thresholds"],
            {
                "confidence": 0.0,
                "margin": 0.0,
                "base_confidence_max": None,
                "base_margin_max": None,
                "digit_protect_confidence": None,
                "upper_protect_confidence": None,
            },
        )
        self.assertEqual(report["selection_samples"], 1)
        self.assertEqual(report["confirmation_samples"], 2)
        self.assertEqual(report["test_delta"], 0.0)
        self.assertEqual(report["balanced_score"], 0.0)
        self.assertEqual(report["balanced_delta"], 0.0)
        self.assertFalse(report["promotable"])

    def test_prepare_data_can_use_external_test_tensor_pack(self) -> None:
        """Feature probes should allow an external held-out tensor pack."""

        train_images = torch.zeros((20, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10, 11] * 10, dtype=torch.long)
        test_images = torch.ones((3, 1, 28, 28), dtype=torch.float32)
        test_targets = torch.tensor([10, 11, 12], dtype=torch.long)
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

        def fake_outputs(images: torch.Tensor, batch_size: int, include_embedding_features: bool):
            if int(images.shape[0]) == 3:
                return test_mixed_outputs, test_folded_outputs, None
            count = int(images.shape[0])
            return mixed_outputs[:count], folded_outputs[:count], None

        with (
            patch("scripts.probe_mixedcase_feature_reranker._split_tensors", side_effect=fake_split),
            patch("scripts.probe_mixedcase_feature_reranker.load_tensor_pack", return_value=(test_images, test_targets)),
            patch("scripts.probe_mixedcase_feature_reranker._model_outputs_with_embeddings", side_effect=fake_outputs),
            patch("scripts.probe_mixedcase_feature_reranker._load_hybrid_artifact", return_value={}),
            patch(
                "scripts.probe_mixedcase_feature_reranker.hybrid_predictions",
                side_effect=lambda mixed, folded, artifact: mixed.argmax(dim=1),
            ),
        ):
            data = prepare_feature_probe_data(
                batch_size=4,
                train_sample_limit=None,
                calibration_ratio=0.2,
                seed=1,
                test_tensor_path=Path("rough-validation.pt"),
            )

        self.assertEqual(data.test_samples, 3)
        self.assertEqual(data.test_tensor_path, Path("rough-validation.pt"))
        self.assertEqual(data.test_targets.tolist(), [10, 11, 12])

    def test_prepare_feature_probe_data_reuses_precomputed_cache(self) -> None:
        """Repeated probe runs should reuse matching prepared logits/features."""

        train_images = torch.zeros((12, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10, 11] * 6, dtype=torch.long)
        test_images = torch.ones((4, 1, 28, 28), dtype=torch.float32)
        test_targets = torch.tensor([10, 11, 10, 11], dtype=torch.long)

        def fake_split(train: bool, sample_limit: int | None):
            return (train_images, train_targets) if train else (test_images, test_targets)

        def fake_outputs(images: torch.Tensor, _batch_size: int, _include_embedding_features: bool):
            mixed = torch.zeros((images.shape[0], 62), dtype=torch.float32)
            folded = torch.zeros((images.shape[0], 36), dtype=torch.float32)
            mixed[:, 10] = 1.0
            folded[:, 10] = 1.0
            return mixed, folded, None

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "prepared.pt"
            with (
                patch("scripts.probe_mixedcase_feature_reranker._split_tensors", side_effect=fake_split),
                patch("scripts.probe_mixedcase_feature_reranker._model_outputs_with_embeddings", side_effect=fake_outputs) as outputs_mock,
                patch("scripts.probe_mixedcase_feature_reranker._load_hybrid_artifact", return_value={}),
                patch(
                    "scripts.probe_mixedcase_feature_reranker.hybrid_predictions",
                    side_effect=lambda mixed, _folded, _artifact: mixed.argmax(dim=1),
                ),
                patch("scripts.probe_mixedcase_feature_reranker._current_artifact_hashes", return_value={"mixed": "hash"}),
            ):
                first, first_hit = load_or_prepare_feature_probe_data(
                    cache_path,
                    batch_size=4,
                    train_sample_limit=None,
                    calibration_ratio=0.25,
                    seed=3,
                )
                second, second_hit = load_or_prepare_feature_probe_data(
                    cache_path,
                    batch_size=4,
                    train_sample_limit=None,
                    calibration_ratio=0.25,
                    seed=3,
                )

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(outputs_mock.call_count, 4)
        self.assertEqual(first.test_samples, second.test_samples)
        self.assertEqual(second.test_targets.tolist(), first.test_targets.tolist())

    def test_prepare_feature_probe_data_invalidates_cache_when_embedding_flag_changes(self) -> None:
        """Prepared data caches should not cross feature-flag boundaries."""

        train_images = torch.zeros((12, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10, 11] * 6, dtype=torch.long)
        test_images = torch.ones((4, 1, 28, 28), dtype=torch.float32)
        test_targets = torch.tensor([10, 11, 10, 11], dtype=torch.long)

        def fake_split(train: bool, sample_limit: int | None):
            return (train_images, train_targets) if train else (test_images, test_targets)

        def fake_outputs(images: torch.Tensor, _batch_size: int, include_embedding_features: bool):
            mixed = torch.zeros((images.shape[0], 62), dtype=torch.float32)
            folded = torch.zeros((images.shape[0], 36), dtype=torch.float32)
            mixed[:, 10] = 1.0
            folded[:, 10] = 1.0
            embedding = torch.ones((images.shape[0], 3), dtype=torch.float32) if include_embedding_features else None
            return mixed, folded, embedding

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "prepared.pt"
            with (
                patch("scripts.probe_mixedcase_feature_reranker._split_tensors", side_effect=fake_split),
                patch("scripts.probe_mixedcase_feature_reranker._model_outputs_with_embeddings", side_effect=fake_outputs) as outputs_mock,
                patch("scripts.probe_mixedcase_feature_reranker._load_hybrid_artifact", return_value={}),
                patch(
                    "scripts.probe_mixedcase_feature_reranker.hybrid_predictions",
                    side_effect=lambda mixed, _folded, _artifact: mixed.argmax(dim=1),
                ),
                patch("scripts.probe_mixedcase_feature_reranker._current_artifact_hashes", return_value={"mixed": "hash"}),
            ):
                load_or_prepare_feature_probe_data(
                    cache_path,
                    batch_size=4,
                    train_sample_limit=None,
                    calibration_ratio=0.25,
                    seed=3,
                    include_embedding_features=False,
                )
                _data, cache_hit = load_or_prepare_feature_probe_data(
                    cache_path,
                    batch_size=4,
                    train_sample_limit=None,
                    calibration_ratio=0.25,
                    seed=3,
                    include_embedding_features=True,
                )

        self.assertFalse(cache_hit)
        self.assertEqual(outputs_mock.call_count, 8)

    def test_run_probe_delegates_to_prepared_data_path(self) -> None:
        """Direct probes should use the same preparation path as sweeps."""

        fake_data = object()
        with (
            patch(
                "scripts.probe_mixedcase_feature_reranker.load_or_prepare_feature_probe_data",
                return_value=(fake_data, True),
            ) as prepare_mock,
            patch(
                "scripts.probe_mixedcase_feature_reranker.run_probe_from_data",
                return_value={"base": {}, "reranked": {}, "promotable": False},
            ) as run_mock,
        ):
            report = run_probe(
                batch_size=4,
                epochs=1,
                learning_rate=0.01,
                train_sample_limit=None,
                family_limit=None,
                calibration_ratio=0.2,
                min_family_delta=0.0,
                seed=5,
                prepare_cache_path=Path("prepared.pt"),
            )

        self.assertTrue(report["prepare_cache_hit"])
        self.assertEqual(report["prepare_cache_path"], "prepared.pt")
        self.assertIs(run_mock.call_args.kwargs["data"], fake_data)
        self.assertEqual(prepare_mock.call_args.args[0], Path("prepared.pt"))

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

    def test_run_probe_writes_promotable_family_reranker_artifact(self) -> None:
        """Promotable probes should be saveable as hash-checked artifacts."""

        train_images = torch.zeros((8, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10, 11, 10, 11, 10, 11, 10, 11], dtype=torch.long)
        test_images = torch.zeros((4, 1, 28, 28), dtype=torch.float32)
        test_targets = torch.tensor([10, 11, 10, 11], dtype=torch.long)
        probe_model = torch.nn.Linear(1, 2)
        protected_base = {
            "test_accuracy": 80.0,
            "case_or_ambiguity_aware_test_accuracy": 98.0,
            "digit_test_accuracy": 95.0,
            "upper_test_accuracy": 88.0,
            "lower_test_accuracy": 75.0,
        }
        improved = {**protected_base, "test_accuracy": 80.2}

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "mixedcase_family_reranker.pt"
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
                        improved,
                        protected_base,
                        improved,
                    ],
                ),
                patch("scripts.probe_mixedcase_feature_reranker._file_sha256", return_value="hash"),
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
                    output_path=output_path,
                    write=True,
                )

            artifact = torch.load(output_path, map_location="cpu", weights_only=True)

        self.assertTrue(report["wrote"])
        self.assertEqual(artifact["probes"][0]["family"], "AB")
        self.assertEqual(artifact["best_checkpoint"]["test_accuracy"], 80.2)

    def test_run_probe_deployed_guard_rejects_split_only_artifact(self) -> None:
        """Deployed validation should block artifacts that only win in the probe split."""

        train_images = torch.zeros((8, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10, 11, 10, 11, 10, 11, 10, 11], dtype=torch.long)
        test_images = torch.zeros((4, 1, 28, 28), dtype=torch.float32)
        test_targets = torch.tensor([10, 11, 10, 11], dtype=torch.long)
        probe_model = torch.nn.Linear(1, 2)
        protected_base = {
            "test_accuracy": 80.0,
            "case_or_ambiguity_aware_test_accuracy": 98.0,
            "digit_test_accuracy": 95.0,
            "upper_test_accuracy": 88.0,
            "lower_test_accuracy": 75.0,
        }
        improved = {**protected_base, "test_accuracy": 80.2}

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "mixedcase_family_reranker.pt"
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
                        improved,
                        protected_base,
                        improved,
                    ],
                ),
                patch("scripts.probe_mixedcase_feature_reranker._file_sha256", return_value="hash"),
                patch(
                    "scripts.probe_mixedcase_feature_reranker._deployed_validation_report",
                    return_value={
                        "base": protected_base,
                        "candidate": protected_base,
                        "delta": {"test_accuracy": 0.0},
                        "promotable": False,
                    },
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
                    output_path=output_path,
                    write=True,
                    require_deployed_validation=True,
                )

            self.assertFalse(output_path.exists())

        self.assertFalse(report["promotable"])
        self.assertFalse(report["wrote"])
        self.assertFalse(report["deployed_validation"]["promotable"])

    def test_merge_family_probe_artifacts_preserves_unrelated_existing_probe(self) -> None:
        """New artifact writes should accumulate safe families across iterations."""

        existing = [{"family": "1Iil", "state_dict": {"weight": torch.tensor([1.0])}}]
        accepted = [{"family": "0Oo", "state_dict": {"weight": torch.tensor([2.0])}}]

        merged = merge_family_probe_artifacts(existing, accepted)

        self.assertEqual([probe["family"] for probe in merged], ["1Iil", "0Oo"])

    def test_merge_family_probe_artifacts_replaces_same_family_probe(self) -> None:
        """A rerun for the same visual family should replace stale probe weights."""

        existing = [{"family": "0Oo", "version": "old"}, {"family": "5Ss", "version": "kept"}]
        accepted = [{"family": "0Oo", "version": "new"}]

        merged = merge_family_probe_artifacts(existing, accepted)

        self.assertEqual(merged, [{"family": "5Ss", "version": "kept"}, {"family": "0Oo", "version": "new"}])


if __name__ == "__main__":
    unittest.main()
