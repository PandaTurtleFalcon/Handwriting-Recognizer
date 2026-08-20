import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.probe_character_family_reranker import (
    CharacterFamilyProbe,
    _gate_metrics,
    _metric_deltas,
    _prediction_change_summary,
    _split_calibration,
    apply_family_probe,
    family_features,
    family_indices,
    geometry_features,
    parse_label_groups,
    parse_families,
    pixel_features,
    merge_family_probe_artifacts,
    run_probe,
    train_family_probe,
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

    def test_family_features_can_include_pixel_sketch(self) -> None:
        """Small pixel sketches should be optional shape evidence for rerankers."""

        images = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
        outputs = torch.zeros((2, 4), dtype=torch.float32)

        pixels = pixel_features(images, size=12)
        base = family_features(images, outputs, (0, 1))
        enriched = family_features(images, outputs, (0, 1), include_pixel_features=True)

        self.assertEqual(tuple(pixels.shape), (2, 144))
        self.assertEqual(enriched.shape[0], base.shape[0])
        self.assertEqual(enriched.shape[1], base.shape[1] + 144)

    def test_family_features_can_include_learned_embeddings(self) -> None:
        """Model embeddings should be optional reranker evidence."""

        images = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
        outputs = torch.zeros((2, 4), dtype=torch.float32)
        embeddings = torch.ones((2, 5), dtype=torch.float32)

        base = family_features(images, outputs, (0, 1))
        enriched = family_features(images, outputs, (0, 1), embedding_outputs=embeddings)

        self.assertEqual(enriched.shape[0], base.shape[0])
        self.assertEqual(enriched.shape[1], base.shape[1] + 5)

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

    def test_apply_family_probe_respects_confidence_gate(self) -> None:
        """A low-confidence family probe should keep the original prediction."""

        labels = ["1", "I"]
        predictions = torch.tensor([0], dtype=torch.long)
        images = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        outputs = torch.zeros((1, 2), dtype=torch.float32)
        model = torch.nn.Linear(28, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias[:] = torch.tensor([0.0, 0.1])

        kept = apply_family_probe(
            predictions,
            images,
            outputs,
            CharacterFamilyProbe("1I", (0, 1), model),
            labels,
            probe_confidence=0.9,
        )
        changed = apply_family_probe(
            predictions,
            images,
            outputs,
            CharacterFamilyProbe("1I", (0, 1), model),
            labels,
            probe_confidence=0.0,
        )

        self.assertEqual(kept.tolist(), [0])
        self.assertEqual(changed.tolist(), [1])

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

    def test_metric_deltas_reports_shared_after_minus_before(self) -> None:
        before = {"validation_accuracy": 94.0, "letter_validation_accuracy": 93.0}
        after = {"validation_accuracy": 94.2, "letter_validation_accuracy": 92.8}

        self.assertEqual(
            _metric_deltas(before, after),
            {"letter_validation_accuracy": -0.20000000000000284, "validation_accuracy": 0.20000000000000284},
        )

    def test_prediction_change_summary_counts_fixes_and_breaks(self) -> None:
        before = torch.tensor([0, 0, 1, 2, 2])
        after = torch.tensor([0, 1, 0, 1, 2])
        targets = torch.tensor([0, 1, 1, 2, 0])

        self.assertEqual(
            _prediction_change_summary(before, after, targets),
            {"changed": 3, "fixed": 1, "broken": 2, "still_wrong_changed": 0},
        )

    def test_family_probe_can_cap_and_minibatch_training_samples(self) -> None:
        features = torch.randn((90, 6), dtype=torch.float32)
        targets = torch.tensor([0, 1, 2] * 30, dtype=torch.long)
        labels = ["!", "1", "I"]

        probe = train_family_probe(
            features,
            targets,
            (0, 1, 2),
            labels,
            epochs=2,
            learning_rate=0.01,
            hidden_units=5,
            max_train_samples=30,
            mini_batch_size=6,
            seed=99,
        )

        self.assertIsNotNone(probe)

    def test_family_probe_rejects_caps_below_minimum_family_coverage(self) -> None:
        features = torch.randn((90, 6), dtype=torch.float32)
        targets = torch.tensor([0, 1, 2] * 30, dtype=torch.long)
        labels = ["!", "1", "I"]

        probe = train_family_probe(
            features,
            targets,
            (0, 1, 2),
            labels,
            epochs=1,
            learning_rate=0.01,
            hidden_units=0,
            max_train_samples=23,
            mini_batch_size=6,
            seed=99,
        )

        self.assertIsNone(probe)

    def test_merge_family_probe_artifacts_replaces_same_family(self) -> None:
        """New accepted probes should replace older weights for the same family."""

        existing = [{"family": "1I", "version": "old"}, {"family": "0O", "version": "kept"}]
        accepted = [{"family": "1I", "version": "new"}]

        self.assertEqual(
            merge_family_probe_artifacts(existing, accepted),
            [{"family": "0O", "version": "kept"}, {"family": "1I", "version": "new"}],
        )

    def test_run_probe_writes_promotable_family_reranker_artifact(self) -> None:
        """Promotable character probes should be saveable as hash-checked artifacts."""

        labels = ["1", "I"]
        images = torch.zeros((8, 1, 32, 32), dtype=torch.float32)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        probe_model = torch.nn.Linear(1, 2)
        base = {
            "validation_accuracy": 80.0,
            "ambiguity_aware_validation_accuracy": 98.0,
            "digit_validation_accuracy": 95.0,
            "letter_validation_accuracy": 93.0,
            "punctuation_validation_accuracy": 96.0,
        }
        improved = {**base, "validation_accuracy": 80.2}

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "character_family_reranker.pt"
            with (
                patch("scripts.probe_character_family_reranker.get_device", return_value=torch.device("cpu")),
                patch(
                    "scripts.probe_character_family_reranker.load_character_model",
                    return_value=(torch.nn.Linear(1, 2), labels),
                ) as load_model_mock,
                patch("scripts.probe_character_family_reranker._character_tensors", return_value=(images, targets, labels)),
                patch(
                    "scripts.probe_character_family_reranker.stratified_split_indices",
                    return_value=(list(range(6)), list(range(6, 8))),
                ),
                patch(
                    "scripts.probe_character_family_reranker._split_calibration",
                    return_value=(torch.arange(2), torch.arange(2, 4), torch.arange(4, 6)),
                ),
                patch("scripts.probe_character_family_reranker._model_outputs", return_value=torch.zeros((2, 2))),
                patch("scripts.probe_character_family_reranker.family_features", return_value=torch.zeros((2, 1))),
                patch(
                    "scripts.probe_character_family_reranker.train_family_probe",
                    return_value=CharacterFamilyProbe("1I", (0, 1), probe_model),
                ),
                patch(
                    "scripts.probe_character_family_reranker.apply_family_probe",
                    side_effect=lambda predictions, *_args, **_kwargs: predictions,
                ),
                patch(
                    "scripts.probe_character_family_reranker._metrics",
                    side_effect=[base, improved, base, improved, base, improved, base, improved],
                ),
                patch(
                    "scripts.probe_character_family_reranker._prediction_change_summary",
                    return_value={"changed": 1, "fixed": 1, "broken": 0, "still_wrong_changed": 0},
                ),
                patch("scripts.probe_character_family_reranker._file_sha256", return_value="hash"),
            ):
                report = run_probe(
                    batch_size=2,
                    epochs=1,
                    learning_rate=0.01,
                    families=("1I",),
                    calibration_ratio=0.25,
                    confirmation_ratio=0.5,
                    min_family_delta=0.01,
                    seed=7,
                    hidden_units=0,
                    output_path=output_path,
                    write=True,
                )

            artifact = torch.load(output_path, map_location="cpu", weights_only=True)

        self.assertTrue(report["wrote"])
        self.assertEqual(artifact["labels"], labels)
        self.assertEqual(artifact["probes"][0]["family"], "1I")
        self.assertEqual(artifact["checkpoint_sha256"], "hash")
        load_model_mock.assert_called_once_with(device=torch.device("cpu"), family_reranker_path=None)

    def test_run_probe_requires_deployed_validation_before_write(self) -> None:
        """Split-only wins should not write artifacts when deployed validation is flat."""

        labels = ["1", "I"]
        images = torch.zeros((8, 1, 32, 32), dtype=torch.float32)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        probe_model = torch.nn.Linear(1, 2)
        base = {
            "validation_accuracy": 80.0,
            "ambiguity_aware_validation_accuracy": 98.0,
            "digit_validation_accuracy": 95.0,
            "letter_validation_accuracy": 93.0,
            "punctuation_validation_accuracy": 96.0,
        }
        improved = {**base, "validation_accuracy": 80.2}

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "character_family_reranker.pt"
            deployed_path = Path(directory) / "deployed.pt"
            deployed_path.write_bytes(b"placeholder")
            with (
                patch("scripts.probe_character_family_reranker.get_device", return_value=torch.device("cpu")),
                patch(
                    "scripts.probe_character_family_reranker.load_character_model",
                    return_value=(torch.nn.Linear(1, 2), labels),
                ),
                patch("scripts.probe_character_family_reranker._character_tensors", return_value=(images, targets, labels)),
                patch(
                    "scripts.probe_character_family_reranker.stratified_split_indices",
                    return_value=(list(range(6)), list(range(6, 8))),
                ),
                patch(
                    "scripts.probe_character_family_reranker._split_calibration",
                    return_value=(torch.arange(2), torch.arange(2, 4), torch.arange(4, 6)),
                ),
                patch("scripts.probe_character_family_reranker._model_outputs", return_value=torch.zeros((2, 2))),
                patch("scripts.probe_character_family_reranker.family_features", return_value=torch.zeros((2, 1))),
                patch(
                    "scripts.probe_character_family_reranker.train_family_probe",
                    return_value=CharacterFamilyProbe("1I", (0, 1), probe_model),
                ),
                patch(
                    "scripts.probe_character_family_reranker.apply_family_probe",
                    side_effect=lambda predictions, *_args, **_kwargs: predictions,
                ),
                patch(
                    "scripts.probe_character_family_reranker._metrics",
                    side_effect=[base, improved, base, improved, base, improved, base, improved],
                ),
                patch(
                    "scripts.probe_character_family_reranker._prediction_change_summary",
                    return_value={"changed": 1, "fixed": 1, "broken": 0, "still_wrong_changed": 0},
                ),
                patch("scripts.probe_character_family_reranker._file_sha256", return_value="hash"),
                patch("scripts.probe_character_family_reranker.FAMILY_RERANKER_PATH", deployed_path),
                patch(
                    "scripts.probe_character_family_reranker.deployed_validation_report",
                    return_value={"safe": False, "delta": {"validation_accuracy": 0.0}},
                ) as deployed_validation,
            ):
                report = run_probe(
                    batch_size=2,
                    epochs=1,
                    learning_rate=0.01,
                    families=("1I",),
                    calibration_ratio=0.25,
                    confirmation_ratio=0.5,
                    min_family_delta=0.01,
                    seed=7,
                    hidden_units=0,
                    output_path=output_path,
                    write=True,
                    require_deployed_validation=True,
                )

            self.assertFalse(output_path.exists())

        self.assertEqual(deployed_validation.call_args.args[1], deployed_path)
        self.assertTrue(report["split_promotable"])
        self.assertFalse(report["promotable"])
        self.assertFalse(report["wrote"])
        self.assertEqual(report["deployed_validation"], {"safe": False, "delta": {"validation_accuracy": 0.0}})

    def test_run_probe_rejects_validation_noops(self) -> None:
        """A family probe that changes no validation samples should not be accepted."""

        labels = ["1", "I"]
        images = torch.zeros((8, 1, 32, 32), dtype=torch.float32)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        metrics = {
            "validation_accuracy": 80.0,
            "ambiguity_aware_validation_accuracy": 98.0,
            "digit_validation_accuracy": 95.0,
            "letter_validation_accuracy": 93.0,
            "punctuation_validation_accuracy": 96.0,
        }

        with (
            patch("scripts.probe_character_family_reranker.get_device", return_value=torch.device("cpu")),
            patch(
                "scripts.probe_character_family_reranker.load_character_model",
                return_value=(torch.nn.Linear(1, 2), labels),
            ),
            patch("scripts.probe_character_family_reranker._character_tensors", return_value=(images, targets, labels)),
            patch(
                "scripts.probe_character_family_reranker.stratified_split_indices",
                return_value=(list(range(6)), list(range(6, 8))),
            ),
            patch(
                "scripts.probe_character_family_reranker._split_calibration",
                return_value=(torch.arange(2), torch.arange(2, 4), torch.arange(4, 6)),
            ),
            patch("scripts.probe_character_family_reranker._model_outputs", return_value=torch.zeros((2, 2))),
            patch("scripts.probe_character_family_reranker.family_features", return_value=torch.zeros((2, 1))),
            patch(
                "scripts.probe_character_family_reranker.train_family_probe",
                return_value=CharacterFamilyProbe("1I", (0, 1), torch.nn.Linear(1, 2)),
            ),
            patch(
                "scripts.probe_character_family_reranker.apply_family_probe",
                side_effect=lambda predictions, *_args, **_kwargs: predictions,
            ),
            patch("scripts.probe_character_family_reranker._metrics", return_value=metrics),
        ):
            report = run_probe(
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                families=("1I",),
                calibration_ratio=0.25,
                confirmation_ratio=0.5,
                min_family_delta=0.0,
                seed=7,
                hidden_units=0,
            )

        self.assertEqual(report["families"][0]["rejection_reason"], "validation_no_changes")
        self.assertFalse(report["promotable"])

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
            patch(
                "scripts.probe_character_family_reranker._model_outputs",
                side_effect=lambda _model, batch_images, *_args: outputs[: len(batch_images)],
            ),
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

    def test_run_probe_adds_train_only_extras_to_fit_split(self) -> None:
        """External caches should train rerankers without touching holdout tensors."""

        labels = ["!", "1"]
        images = torch.zeros((12, 1, 32, 32), dtype=torch.float32)
        targets = torch.tensor([0, 1] * 6, dtype=torch.long)
        outputs = torch.zeros((12, 2), dtype=torch.float32)
        captured_fit = {}

        class FixedModel(torch.nn.Module):
            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                return torch.zeros((batch.shape[0], 2), dtype=torch.float32)

        def fake_train_family_probe(features, train_targets, *_args, **_kwargs):
            captured_fit["feature_rows"] = int(features.shape[0])
            captured_fit["target_rows"] = int(train_targets.numel())
            return None

        def fake_model_outputs(_model, batch_images, *_args, **_kwargs):
            return torch.zeros((batch_images.shape[0], 2), dtype=torch.float32)

        with (
            patch("scripts.probe_character_family_reranker.get_device", return_value=torch.device("cpu")),
            patch("scripts.probe_character_family_reranker.load_character_model", return_value=(FixedModel(), labels)),
            patch("scripts.probe_character_family_reranker._character_tensors", return_value=(images, targets, labels)),
            patch(
                "scripts.probe_character_family_reranker.stratified_split_indices",
                return_value=([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11]),
            ),
            patch("scripts.probe_character_family_reranker._model_outputs", side_effect=fake_model_outputs),
            patch("scripts.probe_character_family_reranker.family_features", return_value=torch.zeros((8, 1))),
            patch("scripts.probe_character_family_reranker.train_family_probe", side_effect=fake_train_family_probe),
            patch(
                "scripts.probe_character_family_reranker.load_extra_character_tensors",
                return_value=(torch.zeros((2, 1, 32, 32)), torch.tensor([0, 1], dtype=torch.long)),
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
                train_only_extra_roots=(Path("extra.pt"),),
                include_pixel_features=True,
                max_probe_train_samples=128,
                mini_batch_size=32,
            )

        self.assertEqual(report["train_only_extra_samples"], 2)
        self.assertTrue(report["include_pixel_features"])
        self.assertEqual(report["max_probe_train_samples"], 128)
        self.assertEqual(report["mini_batch_size"], 32)
        self.assertEqual(report["fit_samples"], 6)
        self.assertEqual(report["selection_samples"], 2)
        self.assertEqual(report["validation_samples"], 4)
        self.assertEqual(captured_fit["feature_rows"], 8)
        self.assertEqual(captured_fit["target_rows"], 6)


if __name__ == "__main__":
    unittest.main()
