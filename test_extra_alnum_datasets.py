import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw

from alnum_model import (
    AugmentedTensorDataset,
    FocalCrossEntropyLoss,
    HybridMixedcaseModel,
    LABELS,
    DEFAULT_MIXEDCASE_BENCHMARK_GATES,
    MIXEDCASE_LABELS,
    MODEL_CLASSES,
    _chars74k_sample_label,
    _mixedcase_train_dataset,
    _nist_sd19_label_from_hex,
    attach_mixedcase_hybrid,
    attach_mixedcase_pair_rules,
    build_or_load_mixedcase_ascii_folder_cache,
    evaluate_mixedcase_breakdown,
    freeze_feature_layers,
    initialize_mixedcase_from_folded_checkpoint,
    limit_mixedcase_extra_cache,
    load_correction_cache,
    load_mixedcase_extra_cache,
    mixedcase_auxiliary_loss,
    mixedcase_benchmark_gate_failures,
    mixedcase_checkpoint_floor_failures,
    mixedcase_checkpoint_meets_floors,
    mixedcase_checkpoint_score,
    mixedcase_distillation_loss,
    mixedcase_folded_logits,
    mixedcase_folded_targets,
    mixedcase_loss_weights,
    mixedcase_labels_match_with_ambiguity,
    mixedcase_labels_match_with_visual_ambiguity,
    mixedcase_type_logits,
    mixedcase_type_targets,
    parse_mixedcase_benchmark_gate_names,
    validate_mixedcase_warm_start_checkpoint,
)
import alnum_model
from extra_alnum_datasets import load_labeled_image_folder


def tiny_transform(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.resize((28, 28)), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class ExtraAlnumDatasetTests(unittest.TestCase):
    """Regression tests for optional local alphanumeric datasets."""

    def test_hybrid_mixedcase_keeps_mixedcase_digits(self) -> None:
        class FixedModel(nn.Module):
            def __init__(self, outputs: torch.Tensor) -> None:
                super().__init__()
                self.outputs = outputs

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.outputs[: inputs.size(0)].clone()

        mixed_outputs = torch.full((3, len(MIXEDCASE_LABELS)), -10.0)
        mixed_outputs[0, 5] = 8.0
        mixed_outputs[1, MIXEDCASE_LABELS.index("a")] = 2.0
        mixed_outputs[1, MIXEDCASE_LABELS.index("A")] = 4.0
        mixed_outputs[2, MIXEDCASE_LABELS.index("b")] = 4.0
        mixed_outputs[2, MIXEDCASE_LABELS.index("B")] = 2.0
        folded_outputs = torch.full((3, len(LABELS)), -10.0)
        folded_outputs[0, LABELS.index("S")] = 9.0
        folded_outputs[1, LABELS.index("A")] = 9.0
        folded_outputs[2, LABELS.index("B")] = 9.0

        model = HybridMixedcaseModel(FixedModel(mixed_outputs), FixedModel(folded_outputs))
        predictions = model(torch.zeros(3, 1, 28, 28)).argmax(dim=1).tolist()

        self.assertEqual(MIXEDCASE_LABELS[predictions[0]], "5")
        self.assertEqual(MIXEDCASE_LABELS[predictions[1]], "A")
        self.assertEqual(MIXEDCASE_LABELS[predictions[2]], "b")

    def test_hybrid_mixedcase_respects_folded_confidence_gate(self) -> None:
        class FixedModel(nn.Module):
            def __init__(self, outputs: torch.Tensor) -> None:
                super().__init__()
                self.outputs = outputs

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.outputs[: inputs.size(0)].clone()

        mixed_outputs = torch.full((1, len(MIXEDCASE_LABELS)), -10.0)
        mixed_outputs[0, MIXEDCASE_LABELS.index("Q")] = 9.0
        folded_outputs = torch.zeros((1, len(LABELS)))
        folded_outputs[0, LABELS.index("A")] = 0.1
        model = HybridMixedcaseModel(
            FixedModel(mixed_outputs),
            FixedModel(folded_outputs),
            folded_confidence_threshold=0.25,
            folded_margin_threshold=0.5,
        )

        prediction = model(torch.zeros(1, 1, 28, 28)).argmax(dim=1).item()

        self.assertEqual(MIXEDCASE_LABELS[prediction], "Q")

    def test_hybrid_mixedcase_uses_per_letter_case_thresholds(self) -> None:
        class FixedModel(nn.Module):
            def __init__(self, outputs: torch.Tensor) -> None:
                super().__init__()
                self.outputs = outputs

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.outputs[: inputs.size(0)].clone()

        mixed_outputs = torch.full((2, len(MIXEDCASE_LABELS)), -10.0)
        mixed_outputs[0, MIXEDCASE_LABELS.index("A")] = 5.0
        mixed_outputs[0, MIXEDCASE_LABELS.index("a")] = 5.2
        mixed_outputs[1, MIXEDCASE_LABELS.index("B")] = 5.0
        mixed_outputs[1, MIXEDCASE_LABELS.index("b")] = 5.2
        folded_outputs = torch.full((2, len(LABELS)), -10.0)
        folded_outputs[0, LABELS.index("A")] = 9.0
        folded_outputs[1, LABELS.index("B")] = 9.0
        model = HybridMixedcaseModel(
            FixedModel(mixed_outputs),
            FixedModel(folded_outputs),
            letter_case_threshold=0.0,
            letter_case_thresholds={"A": 1.0},
        )

        predictions = model(torch.zeros(2, 1, 28, 28)).argmax(dim=1).tolist()

        self.assertEqual(MIXEDCASE_LABELS[predictions[0]], "A")
        self.assertEqual(MIXEDCASE_LABELS[predictions[1]], "b")

    def test_hybrid_mixedcase_can_apply_factorized_gate(self) -> None:
        class FixedModel(nn.Module):
            def __init__(self, outputs: torch.Tensor) -> None:
                super().__init__()
                self.outputs = outputs

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.outputs[: inputs.size(0)].clone()

        mixed_outputs = torch.full((1, len(MIXEDCASE_LABELS)), -10.0)
        mixed_outputs[0, MIXEDCASE_LABELS.index("A")] = 8.0
        mixed_outputs[0, MIXEDCASE_LABELS.index("a")] = 7.5
        mixed_outputs[0, MIXEDCASE_LABELS.index("b")] = 7.5
        mixed_outputs[0, MIXEDCASE_LABELS.index("c")] = 7.5
        folded_outputs = torch.full((1, len(LABELS)), -10.0)
        folded_outputs[0, LABELS.index("A")] = 9.0
        model = HybridMixedcaseModel(
            FixedModel(mixed_outputs),
            FixedModel(folded_outputs),
            factorized_gate_enabled=True,
            factorized_folded_confidence_threshold=0.5,
            factorized_type_confidence_threshold=0.4,
        )

        prediction = model(torch.zeros(1, 1, 28, 28)).argmax(dim=1).item()

        self.assertEqual(MIXEDCASE_LABELS[prediction], "a")

    def test_hybrid_mixedcase_factorized_gate_respects_type_confidence(self) -> None:
        class FixedModel(nn.Module):
            def __init__(self, outputs: torch.Tensor) -> None:
                super().__init__()
                self.outputs = outputs

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.outputs[: inputs.size(0)].clone()

        mixed_outputs = torch.full((1, len(MIXEDCASE_LABELS)), -10.0)
        mixed_outputs[0, MIXEDCASE_LABELS.index("A")] = 8.0
        mixed_outputs[0, MIXEDCASE_LABELS.index("a")] = 7.5
        mixed_outputs[0, MIXEDCASE_LABELS.index("b")] = 7.5
        mixed_outputs[0, MIXEDCASE_LABELS.index("c")] = 7.5
        folded_outputs = torch.full((1, len(LABELS)), -10.0)
        folded_outputs[0, LABELS.index("A")] = 9.0
        model = HybridMixedcaseModel(
            FixedModel(mixed_outputs),
            FixedModel(folded_outputs),
            factorized_gate_enabled=True,
            factorized_folded_confidence_threshold=0.5,
            factorized_type_confidence_threshold=0.99,
        )

        prediction = model(torch.zeros(1, 1, 28, 28)).argmax(dim=1).item()

        self.assertEqual(MIXEDCASE_LABELS[prediction], "A")

    def test_attach_mixedcase_hybrid_rejects_stale_checkpoint_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixed_weights = root / "mixed.pt"
            folded_weights = root / "folded.pt"
            hybrid_path = root / "mixedcase_hybrid.json"
            mixed_weights.write_bytes(b"current mixed")
            folded_weights.write_bytes(b"current folded")
            hybrid_path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "labels": MIXEDCASE_LABELS,
                        "mixedcase_checkpoint_sha256": "stale",
                        "folded_checkpoint_sha256": "stale",
                        "best_checkpoint": {"test_accuracy": 99.0},
                    }
                ),
                encoding="utf-8",
            )
            model = nn.Identity()

            wrapped = attach_mixedcase_hybrid(
                model,
                list(MIXEDCASE_LABELS),
                torch.device("cpu"),
                hybrid_path,
                mixed_weights,
                folded_weights,
            )

        self.assertIs(wrapped, model)

    def test_loads_image_folder_classes_into_label_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for label in ("0", "A"):
                class_dir = root / label
                class_dir.mkdir()
                image = Image.new("L", (18, 18), 255)
                draw = ImageDraw.Draw(image)
                draw.text((4, 2), label, fill=0)
                image.save(class_dir / f"{label}.png")

            images, targets = load_labeled_image_folder(root, ["0", "1", "A"], tiny_transform)

        self.assertEqual(tuple(images.shape), (2, 1, 28, 28))
        self.assertEqual(targets.tolist(), [0, 2])

    def test_rejects_unknown_class_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "not-a-label").mkdir()

            with self.assertRaisesRegex(RuntimeError, "unsupported class folders"):
                load_labeled_image_folder(root, ["0", "A"], tiny_transform)

    def test_chars74k_labels_fold_to_alphanumeric_targets(self) -> None:
        self.assertEqual(_chars74k_sample_label(Path("Sample001")), 0)
        self.assertEqual(_chars74k_sample_label(Path("Sample010")), 9)
        self.assertEqual(_chars74k_sample_label(Path("Sample011")), 10)
        self.assertEqual(_chars74k_sample_label(Path("Sample036")), 35)
        self.assertEqual(_chars74k_sample_label(Path("Sample037")), 10)
        self.assertEqual(_chars74k_sample_label(Path("Sample062")), 35)
        self.assertIsNone(_chars74k_sample_label(Path("Sample063")))

    def test_mixedcase_labels_keep_uppercase_and_lowercase_separate(self) -> None:
        self.assertEqual(len(MIXEDCASE_LABELS), 62)
        self.assertEqual(MIXEDCASE_LABELS.index("S"), 28)
        self.assertEqual(MIXEDCASE_LABELS.index("s"), 54)

    def test_residual_cnn_model_type_is_available_for_mixedcase_experiments(self) -> None:
        """The deeper candidate must keep the same input/output contract."""

        model = MODEL_CLASSES["rescnn"](num_classes=len(MIXEDCASE_LABELS))
        outputs = model(torch.zeros((2, 1, 28, 28), dtype=torch.float32))

        self.assertEqual(tuple(outputs.shape), (2, len(MIXEDCASE_LABELS)))

    def test_mixedcase_auxiliary_targets_fold_case_and_type(self) -> None:
        """Auxiliary losses should use stable digit/case/type target mappings."""

        targets = torch.tensor([0, 10, 36, 35, 61])

        self.assertEqual(mixedcase_folded_targets(targets).tolist(), [0, 10, 10, 35, 35])
        self.assertEqual(mixedcase_type_targets(targets).tolist(), [0, 1, 2, 1, 2])

    def test_mixedcase_auxiliary_logits_and_loss_are_finite(self) -> None:
        """Folded/type auxiliary losses should be differentiable from class logits."""

        outputs = torch.zeros((3, len(MIXEDCASE_LABELS)), requires_grad=True)
        targets = torch.tensor([1, 10, 36])

        self.assertEqual(tuple(mixedcase_folded_logits(outputs).shape), (3, 36))
        self.assertEqual(tuple(mixedcase_type_logits(outputs).shape), (3, 3))

        loss = mixedcase_auxiliary_loss(outputs, targets, folded_weight=0.2, type_weight=0.3)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(outputs.grad)

    def test_mixedcase_distillation_loss_preserves_teacher_logits(self) -> None:
        """Distillation should be zero for matching logits and positive for drift."""

        student = torch.tensor([[2.0, 0.0], [0.2, 1.1]], dtype=torch.float32)
        teacher = student.clone()
        shifted = torch.tensor([[0.0, 2.0], [1.1, 0.2]], dtype=torch.float32)

        self.assertEqual(float(mixedcase_distillation_loss(student, teacher, weight=0.0).item()), 0.0)
        self.assertAlmostEqual(
            float(mixedcase_distillation_loss(student, teacher, weight=1.0).item()),
            0.0,
            places=6,
        )
        self.assertGreater(float(mixedcase_distillation_loss(shifted, teacher, weight=1.0).item()), 0.0)

    def test_mixedcase_ascii_folder_loader_preserves_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "mixed"
            for label in ("A", "a"):
                class_dir = root / str(ord(label))
                class_dir.mkdir(parents=True)
                image = Image.new("L", (24, 24), 255)
                draw = ImageDraw.Draw(image)
                draw.text((5, 4), label, fill=0)
                image.save(class_dir / f"{label}.png")

            images, targets = build_or_load_mixedcase_ascii_folder_cache(root)

        self.assertEqual(tuple(images.shape), (2, 1, 28, 28))
        self.assertEqual(targets.tolist(), [10, 36])

    def test_load_mixedcase_extra_cache_accepts_tensor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "mixedcase_cache.pt"
            images = torch.zeros((2, 1, 28, 28), dtype=torch.float64)
            targets = torch.tensor([0, 61], dtype=torch.int32)
            torch.save({"images": images, "targets": targets}, cache_path)

            loaded_images, loaded_targets = load_mixedcase_extra_cache(cache_path)

        self.assertEqual(tuple(loaded_images.shape), (2, 1, 28, 28))
        self.assertEqual(loaded_images.dtype, torch.float32)
        self.assertEqual(loaded_targets.tolist(), [0, 61])
        self.assertEqual(loaded_targets.dtype, torch.long)

    def test_load_mixedcase_extra_cache_rejects_invalid_tensor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "mixedcase_cache.pt"
            torch.save(
                {
                    "images": torch.zeros((1, 1, 28, 27), dtype=torch.float32),
                    "targets": torch.tensor([0], dtype=torch.long),
                },
                cache_path,
            )

            with self.assertRaises(RuntimeError):
                load_mixedcase_extra_cache(cache_path)

    def test_limit_mixedcase_extra_cache_caps_each_class(self) -> None:
        images = torch.arange(6 * 28 * 28, dtype=torch.float32).reshape(6, 1, 28, 28)
        targets = torch.tensor([10, 10, 10, 36, 36, 36], dtype=torch.long)

        limited_images, limited_targets = limit_mixedcase_extra_cache(images, targets, 2, seed=7)

        self.assertEqual(tuple(limited_images.shape), (4, 1, 28, 28))
        counts = torch.bincount(limited_targets, minlength=len(MIXEDCASE_LABELS))
        self.assertEqual(counts[10].item(), 2)
        self.assertEqual(counts[36].item(), 2)

    def test_freeze_feature_layers_keeps_only_final_classifier_trainable(self) -> None:
        model = MODEL_CLASSES["widecnn"](num_classes=len(MIXEDCASE_LABELS))

        frozen_count = freeze_feature_layers(model)

        self.assertGreater(frozen_count, 0)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertEqual(trainable, ["network.10.weight", "network.10.bias"])

    def test_freeze_feature_layers_can_leave_late_tail_trainable(self) -> None:
        model = MODEL_CLASSES["widecnn"](num_classes=len(MIXEDCASE_LABELS))

        freeze_feature_layers(model, trainable_tail_modules=2)

        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertEqual(
            trainable,
            [
                "network.7.weight",
                "network.7.bias",
                "network.10.weight",
                "network.10.bias",
            ],
        )

    def test_mixedcase_train_dataset_can_enable_tensor_augmentation(self) -> None:
        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)

        plain = _mixedcase_train_dataset(images, targets, augment=False)
        augmented = _mixedcase_train_dataset(images, targets, augment=True)

        self.assertIsInstance(plain, TensorDataset)
        self.assertIsInstance(augmented, AugmentedTensorDataset)
        self.assertEqual(len(augmented), 2)
        image, target = augmented[0]
        self.assertEqual(tuple(image.shape), (1, 28, 28))
        self.assertEqual(int(target), 0)

    def test_mixedcase_loss_weights_can_target_case_and_weak_labels(self) -> None:
        weights = mixedcase_loss_weights(
            ["0", "A", "a", "s"],
            upper_weight=1.2,
            lower_weight=1.1,
            weak_labels="0s",
            weak_weight=1.5,
        )

        self.assertIsNotNone(weights)
        assert weights is not None
        for actual, expected in zip(weights.tolist(), [1.5, 1.2, 1.1, 1.65]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertIsNone(mixedcase_loss_weights(["A", "a"]))

    def test_mixedcase_loss_weights_can_blend_class_balance(self) -> None:
        weights = mixedcase_loss_weights(
            ["common", "rare"],
            class_counts=np.array([100, 25]),
            class_balance_strength=0.5,
        )

        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertLess(weights[0].item(), weights[1].item())
        self.assertAlmostEqual(weights[0].item(), 0.7905694, places=5)
        self.assertAlmostEqual(weights[1].item(), 1.5811388, places=5)

    def test_focal_cross_entropy_zero_gamma_matches_cross_entropy(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]])
        targets = torch.tensor([0, 2])

        focal_loss = FocalCrossEntropyLoss(gamma=0.0)(logits, targets)
        cross_entropy = nn.functional.cross_entropy(logits, targets)

        self.assertAlmostEqual(float(focal_loss.item()), float(cross_entropy.item()), places=6)

    def test_focal_cross_entropy_downweights_easy_examples(self) -> None:
        logits = torch.tensor([[5.0, -1.0], [0.1, 0.0]])
        targets = torch.tensor([0, 1])

        plain_loss = FocalCrossEntropyLoss(gamma=0.0)(logits, targets)
        focal_loss = FocalCrossEntropyLoss(gamma=1.5)(logits, targets)

        self.assertLess(float(focal_loss.item()), float(plain_loss.item()))

    def test_mixedcase_checkpoint_score_can_use_balanced_group_accuracy(self) -> None:
        """Balanced checkpoint selection should optimize the weakest split."""

        metrics = {
            "test_accuracy": 90.0,
            "digit_test_accuracy": 98.0,
            "upper_test_accuracy": 84.0,
            "lower_test_accuracy": 73.0,
        }

        self.assertEqual(mixedcase_checkpoint_score(metrics, "test_accuracy"), 90.0)
        self.assertEqual(mixedcase_checkpoint_score(metrics, "balanced_group_accuracy"), 73.0)

    def test_mixedcase_checkpoint_floors_reject_group_regressions(self) -> None:
        """Checkpoint floors should stop lowercase gains that collapse uppercase."""

        metrics = {
            "case_or_ambiguity_aware_test_accuracy": 98.1,
            "digit_test_accuracy": 97.9,
            "upper_test_accuracy": 67.2,
            "lower_test_accuracy": 86.4,
        }

        self.assertFalse(
            mixedcase_checkpoint_meets_floors(
                metrics,
                min_case_or_visual=98.0,
                min_digit=95.0,
                min_upper=84.0,
                min_lower=73.0,
            )
        )

    def test_mixedcase_checkpoint_floor_failures_name_missed_floors(self) -> None:
        """Rejected candidate diagnostics should explain which gate failed."""

        metrics = {
            "case_or_ambiguity_aware_test_accuracy": 95.9,
            "digit_test_accuracy": 96.0,
            "upper_test_accuracy": 73.2,
            "lower_test_accuracy": 84.2,
        }

        self.assertEqual(
            mixedcase_checkpoint_floor_failures(
                metrics,
                min_case_or_visual=96.0,
                min_digit=95.0,
                min_upper=80.0,
                min_lower=80.0,
            ),
            [
                "case_or_visual 95.90% < floor 96.00%",
                "upper 73.20% < floor 80.00%",
            ],
        )

    def test_parse_mixedcase_benchmark_gate_names_defaults_to_mixedcase_gates(self) -> None:
        """Empty mixed-case benchmark gate values should use the mixed-case gates."""

        self.assertEqual(
            parse_mixedcase_benchmark_gate_names(" mixedcase_exact, mixedcase_lower_exact "),
            ("mixedcase_exact", "mixedcase_lower_exact"),
        )
        self.assertEqual(parse_mixedcase_benchmark_gate_names(""), DEFAULT_MIXEDCASE_BENCHMARK_GATES)

    def test_mixedcase_benchmark_gate_failures_report_regressions_and_targets(self) -> None:
        """Post-training mixed-case gates should explain rejected candidates."""

        failures = mixedcase_benchmark_gate_failures(
            {"mixedcase_exact": 87.8, "mixedcase_lower_exact": 73.1},
            {"mixedcase_exact": 87.9, "mixedcase_lower_exact": 72.9},
            target=95.0,
        )

        self.assertEqual(
            failures,
            [
                "mixedcase_exact 87.9000% < target 95.0000%",
                "mixedcase_lower_exact 72.9000% < baseline 73.1000%",
                "mixedcase_lower_exact 72.9000% < target 95.0000%",
            ],
        )

    def test_mixedcase_cli_can_require_saved_benchmark_gates(self) -> None:
        """Protected mixed-case training should check saved benchmark gates."""

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "sys.argv",
                [
                    "alnum_model.py",
                    "--mixed-case",
                    "--mixedcase-require-benchmark-gates",
                    "--mixedcase-benchmark-gate-names",
                    "mixedcase_exact",
                    "--mixedcase-benchmark-backup-dir",
                    str(Path(directory) / "backup"),
                ],
            ),
            patch(
                "alnum_model.saved_mixedcase_benchmark_values",
                side_effect=[{"mixedcase_exact": 87.7}, {"mixedcase_exact": 87.8}],
            ) as gates,
            patch("alnum_model.backup_mixedcase_artifacts") as backup,
            patch("alnum_model.restore_mixedcase_artifacts") as restore,
            patch("alnum_model.train_mixedcase") as train,
        ):
            alnum_model.main()

        self.assertEqual(gates.call_count, 2)
        self.assertTrue(backup.called)
        self.assertTrue(train.called)
        self.assertFalse(restore.called)

    def test_mixedcase_cli_passes_candidate_output_paths(self) -> None:
        """Candidate mixed-case runs should write to caller-provided artifact paths."""

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "sys.argv",
                [
                    "alnum_model.py",
                    "--mixed-case",
                    "--mixedcase-output-weights-path",
                    str(Path(directory) / "candidate.pt"),
                    "--mixedcase-output-metrics-path",
                    str(Path(directory) / "candidate_metrics.json"),
                ],
            ),
            patch("alnum_model.train_mixedcase") as train,
        ):
            alnum_model.main()

        _, kwargs = train.call_args
        self.assertEqual(kwargs["output_weights_path"], Path(directory) / "candidate.pt")
        self.assertEqual(kwargs["output_metrics_path"], Path(directory) / "candidate_metrics.json")

    def test_mixedcase_cli_can_require_trained_candidate_checkpoint(self) -> None:
        """Candidate loops can reject plain warm-start copies."""

        with (
            patch(
                "sys.argv",
                [
                    "alnum_model.py",
                    "--mixed-case",
                    "--mixedcase-require-trained-checkpoint",
                ],
            ),
            patch("alnum_model.train_mixedcase") as train,
        ):
            alnum_model.main()

        _, kwargs = train.call_args
        self.assertTrue(kwargs["require_trained_checkpoint"])

    def test_mixedcase_cli_passes_distillation_settings(self) -> None:
        """Mixed-case experiments can preserve warm-start logits with distillation."""

        with (
            patch(
                "sys.argv",
                [
                    "alnum_model.py",
                    "--mixed-case",
                    "--mixedcase-distillation-weight",
                    "0.4",
                    "--mixedcase-distillation-temperature",
                    "3.5",
                ],
            ),
            patch("alnum_model.train_mixedcase") as train,
        ):
            alnum_model.main()

        _, kwargs = train.call_args
        self.assertEqual(kwargs["distillation_weight"], 0.4)
        self.assertEqual(kwargs["distillation_temperature"], 3.5)

    def test_mixedcase_cli_rejects_candidate_outputs_with_deployed_gates(self) -> None:
        """Deployed benchmark gates should not be used to approve candidate files."""

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "sys.argv",
                [
                    "alnum_model.py",
                    "--mixed-case",
                    "--mixedcase-require-benchmark-gates",
                    "--mixedcase-output-weights-path",
                    str(Path(directory) / "candidate.pt"),
                ],
            ),
            patch("alnum_model.train_mixedcase") as train,
        ):
            with self.assertRaisesRegex(ValueError, "deployed benchmark artifacts"):
                alnum_model.main()

        self.assertFalse(train.called)

    def test_save_mixedcase_checkpoint_writes_custom_artifact_paths(self) -> None:
        """Mixed-case checkpoint persistence should support isolated candidate files."""

        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "nested" / "candidate.pt"
            metrics_path = Path(directory) / "nested" / "candidate_metrics.json"

            wrote_weights = alnum_model.save_mixedcase_checkpoint(
                history=[{"epoch": 1, "test_accuracy": 12.5, "checkpoint_floor_failures": []}],
                best_state={"classifier.weight": torch.zeros(1, 1)},
                best_accuracy=12.5,
                best_metrics={"test_accuracy": 12.5, "source": "unit_test"},
                best_observed_metrics={"test_accuracy": 13.0, "source": "observed_unit_test"},
                model_type="cnn",
                learning_rate=0.001,
                seed=123,
                device=torch.device("cpu"),
                samples_per_class=1,
                output_weights_path=weights_path,
                output_metrics_path=metrics_path,
            )

            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertTrue(wrote_weights)
        self.assertEqual(checkpoint["test_accuracy"], 12.5)
        self.assertEqual(metrics["output_weights_path"], str(weights_path))
        self.assertEqual(metrics["output_metrics_path"], str(metrics_path))
        self.assertTrue(metrics["wrote_weights"])
        self.assertTrue(metrics["latest_checkpoint_floor_passed"])
        self.assertEqual(metrics["latest_checkpoint_floor_failures"], [])
        self.assertEqual(metrics["best_observed_checkpoint"]["test_accuracy"], 13.0)
        self.assertEqual(metrics["trainable_tail_modules"], 1)

    def test_save_mixedcase_checkpoint_reports_missing_weights(self) -> None:
        """Metrics should say when no checkpoint satisfied save floors."""

        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "candidate.pt"
            metrics_path = Path(directory) / "candidate_metrics.json"

            wrote_weights = alnum_model.save_mixedcase_checkpoint(
                history=[
                    {
                        "epoch": 1,
                        "test_accuracy": 12.5,
                        "checkpoint_floor_failures": ["upper 12.50% < floor 84.70%"],
                    }
                ],
                best_state=None,
                best_accuracy=12.5,
                best_metrics=None,
                best_observed_metrics={"test_accuracy": 12.5, "source": "epoch_1"},
                model_type="cnn",
                learning_rate=0.001,
                seed=123,
                device=torch.device("cpu"),
                samples_per_class=1,
                output_weights_path=weights_path,
                output_metrics_path=metrics_path,
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertFalse(wrote_weights)
        self.assertFalse(weights_path.exists())
        self.assertFalse(metrics["wrote_weights"])
        self.assertFalse(metrics["latest_checkpoint_floor_passed"])
        self.assertEqual(metrics["latest_checkpoint_floor_failures"], ["upper 12.50% < floor 84.70%"])
        self.assertEqual(metrics["best_observed_checkpoint"]["source"], "epoch_1")

    def test_train_mixedcase_rejects_when_no_checkpoint_is_written(self) -> None:
        """Candidate runs should fail clearly if all checkpoints miss floors."""

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("alnum_model.get_device", return_value=torch.device("cpu")),
                patch("alnum_model.make_mixedcase_loaders") as loaders,
            ):
                loaders.return_value = (
                    DataLoader(TensorDataset(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long))),
                    DataLoader(TensorDataset(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long))),
                    DataLoader(TensorDataset(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long))),
                    DataLoader(TensorDataset(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long))),
                    DataLoader(TensorDataset(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long))),
                    np.ones(len(MIXEDCASE_LABELS), dtype=np.int64),
                )

                with self.assertRaisesRegex(RuntimeError, "no weights were saved"):
                    alnum_model.train_mixedcase(
                        epochs=0,
                        batch_size=1,
                        min_accuracy=0.0,
                        learning_rate=0.001,
                        seed=123,
                        model_type="cnn",
                        samples_per_class=1,
                        device_name="auto",
                        output_weights_path=Path(directory) / "candidate.pt",
                        output_metrics_path=Path(directory) / "candidate_metrics.json",
                    )

    def test_train_mixedcase_can_reject_warm_start_only_candidate(self) -> None:
        """Candidate runs should not save a copied warm start when asked for a real epoch."""

        with tempfile.TemporaryDirectory() as directory:
            warm_start_path = Path(directory) / "warm_start.pt"
            weights_path = Path(directory) / "candidate.pt"
            metrics_path = Path(directory) / "candidate_metrics.json"
            with (
                patch("alnum_model.get_device", return_value=torch.device("cpu")),
                patch("alnum_model.make_mixedcase_loaders") as loaders,
                patch("alnum_model.MIXEDCASE_WEIGHTS_PATH", warm_start_path),
                patch(
                    "torch.load",
                    return_value={
                        "model_state_dict": MODEL_CLASSES["cnn"](len(MIXEDCASE_LABELS)).state_dict(),
                        "labels": list(MIXEDCASE_LABELS),
                        "model_type": "cnn",
                    },
                ),
                patch("alnum_model.evaluate", return_value=(0.0, 80.0)),
                patch(
                    "alnum_model.evaluate_mixedcase_breakdown",
                    return_value={
                        "test_accuracy": 80.0,
                        "case_or_ambiguity_aware_test_accuracy": 97.0,
                        "casefold_test_accuracy": 87.0,
                        "ambiguity_aware_test_accuracy": 90.0,
                        "digit_test_accuracy": 83.0,
                        "upper_test_accuracy": 72.0,
                        "lower_test_accuracy": 84.0,
                    },
                ),
                patch("alnum_model.evaluate_per_class", return_value={}),
            ):
                weights_path.write_bytes(b"warm-start")
                loaders.return_value = (
                    DataLoader(TensorDataset(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long))),
                    DataLoader(TensorDataset(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long))),
                    DataLoader(TensorDataset(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long))),
                    DataLoader(TensorDataset(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long))),
                    DataLoader(TensorDataset(torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long))),
                    np.ones(len(MIXEDCASE_LABELS), dtype=np.int64),
                )

                warm_start_path.write_bytes(b"warm-start")
                with self.assertRaisesRegex(RuntimeError, "warm start"):
                    alnum_model.train_mixedcase(
                        epochs=1,
                        batch_size=1,
                        min_accuracy=0.0,
                        learning_rate=0.001,
                        seed=123,
                        model_type="cnn",
                        samples_per_class=1,
                        device_name="auto",
                        warm_start=True,
                        min_checkpoint_case_or_visual=95.0,
                        min_checkpoint_digit=80.0,
                        min_checkpoint_upper=70.0,
                        min_checkpoint_lower=80.0,
                        output_weights_path=weights_path,
                        output_metrics_path=metrics_path,
                        require_trained_checkpoint=True,
                    )

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertFalse(metrics["wrote_weights"])
        self.assertEqual(metrics["best_checkpoint"]["source"], "warm_start_seed")
        self.assertFalse(weights_path.exists())

    def test_mixedcase_cli_restores_artifacts_when_benchmark_gate_regresses(self) -> None:
        """Protected mixed-case training should restore artifacts after regression."""

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "sys.argv",
                [
                    "alnum_model.py",
                    "--mixed-case",
                    "--mixedcase-require-benchmark-gates",
                    "--mixedcase-benchmark-gate-names",
                    "mixedcase_exact",
                    "--mixedcase-benchmark-backup-dir",
                    str(Path(directory) / "backup"),
                ],
            ),
            patch(
                "alnum_model.saved_mixedcase_benchmark_values",
                side_effect=[{"mixedcase_exact": 87.8}, {"mixedcase_exact": 87.7}],
            ),
            patch("alnum_model.backup_mixedcase_artifacts"),
            patch("alnum_model.restore_mixedcase_artifacts") as restore,
            patch("alnum_model.train_mixedcase"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Rejected mixed-case training checkpoint"):
                alnum_model.main()

        self.assertTrue(restore.called)

    def test_mixedcase_cli_restores_artifacts_when_training_raises(self) -> None:
        """Protected mixed-case training should restore artifacts after hard failures."""

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "sys.argv",
                [
                    "alnum_model.py",
                    "--mixed-case",
                    "--mixedcase-require-benchmark-gates",
                    "--mixedcase-benchmark-gate-names",
                    "mixedcase_exact",
                    "--mixedcase-benchmark-backup-dir",
                    str(Path(directory) / "backup"),
                ],
            ),
            patch("alnum_model.saved_mixedcase_benchmark_values", return_value={"mixedcase_exact": 87.8}),
            patch("alnum_model.backup_mixedcase_artifacts"),
            patch("alnum_model.restore_mixedcase_artifacts") as restore,
            patch("alnum_model.train_mixedcase", side_effect=RuntimeError("training failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                alnum_model.main()

        self.assertTrue(restore.called)

    def test_mixedcase_warm_start_rejects_model_type_mismatch(self) -> None:
        """Warm-start training should fail fast instead of random-initializing."""

        with self.assertRaisesRegex(RuntimeError, "model type"):
            validate_mixedcase_warm_start_checkpoint(
                {"labels": list(MIXEDCASE_LABELS), "model_type": "cnn", "model_state_dict": {}},
                "rescnn",
            )

    def test_mixedcase_warm_start_accepts_matching_checkpoint(self) -> None:
        """A matching mixed-case checkpoint should pass warm-start validation."""

        validate_mixedcase_warm_start_checkpoint(
            {"labels": list(MIXEDCASE_LABELS), "model_type": "cnn", "model_state_dict": {}},
            "cnn",
        )

    def test_mixedcase_warm_start_metrics_must_meet_checkpoint_floors(self) -> None:
        """A raw warm-start seed below deployed floors should not be saveable."""

        raw_warm_start_metrics = {
            "case_or_ambiguity_aware_test_accuracy": 97.02,
            "digit_test_accuracy": 94.92,
            "upper_test_accuracy": 84.07,
            "lower_test_accuracy": 72.65,
        }

        self.assertFalse(
            mixedcase_checkpoint_meets_floors(
                raw_warm_start_metrics,
                min_case_or_visual=98.04,
                min_digit=95.02,
                min_upper=84.70,
                min_lower=73.14,
            )
        )

    def test_mixedcase_ambiguity_groups_match_known_lookalikes(self) -> None:
        self.assertTrue(mixedcase_labels_match_with_ambiguity("S", "s"))
        self.assertTrue(mixedcase_labels_match_with_ambiguity("0", "O"))
        self.assertTrue(mixedcase_labels_match_with_ambiguity("1", "l"))
        self.assertTrue(mixedcase_labels_match_with_ambiguity("l", "i"))
        self.assertTrue(mixedcase_labels_match_with_ambiguity("q", "9"))
        self.assertTrue(mixedcase_labels_match_with_ambiguity("T", "7"))
        self.assertFalse(mixedcase_labels_match_with_ambiguity("A", "B"))

    def test_mixedcase_visual_ambiguity_keeps_casefold_separate(self) -> None:
        self.assertFalse(mixedcase_labels_match_with_visual_ambiguity("S", "s"))
        self.assertFalse(mixedcase_labels_match_with_visual_ambiguity("A", "a"))
        self.assertTrue(mixedcase_labels_match_with_visual_ambiguity("0", "O"))
        self.assertTrue(mixedcase_labels_match_with_visual_ambiguity("T", "7"))

    def test_attach_mixedcase_pair_rules_flips_close_visual_twin(self) -> None:
        class FixedLogitModel(nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor(
                    [
                        [0.40, 0.30, 0.00],
                        [0.40, 0.10, 0.00],
                    ],
                    dtype=torch.float32,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "mixedcase_pair_rules.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "labels": ["0", "O", "A"],
                        "rules": [{"from": "0", "to": "O", "threshold": -0.15}],
                    }
                ),
                encoding="utf-8",
            )
            model = FixedLogitModel()

            attached = attach_mixedcase_pair_rules(model, ["0", "O", "A"], torch.device("cpu"), rules_path)

        self.assertTrue(attached)
        self.assertEqual(model(torch.zeros((2, 1, 28, 28))).argmax(dim=1).tolist(), [1, 0])

    def test_attach_mixedcase_pair_rules_rejects_mismatched_checkpoint_hash(self) -> None:
        class FixedLogitModel(nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.40, 0.30, 0.00]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "mixedcase_pair_rules.json"
            weights_path = Path(temp_dir) / "mixedcase_cnn.pt"
            weights_path.write_bytes(b"current weights")
            rules_path.write_text(
                json.dumps(
                    {
                        "labels": ["0", "O", "A"],
                        "checkpoint_sha256": "not-the-current-checkpoint",
                        "rules": [{"from": "0", "to": "O", "threshold": -0.15}],
                    }
                ),
                encoding="utf-8",
            )
            model = FixedLogitModel()

            attached = attach_mixedcase_pair_rules(
                model,
                ["0", "O", "A"],
                torch.device("cpu"),
                rules_path,
                weights_path,
            )

        self.assertFalse(attached)
        self.assertFalse(hasattr(model, "mixedcase_pair_rules"))
        self.assertEqual(int(model(torch.zeros((1, 1, 28, 28))).argmax(dim=1).item()), 0)

    def test_mixedcase_breakdown_reports_casefold_and_ambiguity_metrics(self) -> None:
        class FixedPredictionModel(nn.Module):
            def __init__(self, predictions: list[int]) -> None:
                super().__init__()
                self.predictions = predictions
                self.offset = 0

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                batch_predictions = self.predictions[self.offset : self.offset + images.size(0)]
                self.offset += images.size(0)
                logits = torch.zeros((images.size(0), len(MIXEDCASE_LABELS)), dtype=torch.float32)
                for row, label_index in enumerate(batch_predictions):
                    logits[row, label_index] = 10.0
                return logits

        expected_labels = ["S", "s", "0", "T"]
        predicted_labels = ["S", "S", "O", "7"]
        targets = torch.tensor([MIXEDCASE_LABELS.index(label) for label in expected_labels])
        predictions = [MIXEDCASE_LABELS.index(label) for label in predicted_labels]
        images = torch.zeros((len(expected_labels), 1, 28, 28), dtype=torch.float32)
        loader = DataLoader(TensorDataset(images, targets), batch_size=2)

        metrics = evaluate_mixedcase_breakdown(
            FixedPredictionModel(predictions),
            loader,
            nn.CrossEntropyLoss(),
            list(MIXEDCASE_LABELS),
            torch.device("cpu"),
        )

        self.assertAlmostEqual(metrics["test_accuracy"], 25.0)
        self.assertAlmostEqual(metrics["casefold_test_accuracy"], 50.0)
        self.assertAlmostEqual(metrics["ambiguity_aware_test_accuracy"], 75.0)
        self.assertAlmostEqual(metrics["case_or_ambiguity_aware_test_accuracy"], 100.0)
        self.assertAlmostEqual(metrics["digit_ambiguity_aware_test_accuracy"], 100.0)
        self.assertAlmostEqual(metrics["upper_ambiguity_aware_test_accuracy"], 100.0)
        self.assertAlmostEqual(metrics["lower_ambiguity_aware_test_accuracy"], 0.0)
        self.assertAlmostEqual(metrics["lower_case_or_ambiguity_aware_test_accuracy"], 100.0)

    def test_mixedcase_transfer_initializes_lowercase_from_folded_letters(self) -> None:
        """Transfer init should duplicate folded uppercase rows into lowercase rows."""

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "folded.pt"
            folded_model = MODEL_CLASSES["cnn"](num_classes=len(LABELS))
            folded_state = folded_model.state_dict()
            output_weight_key = "network.20.weight"
            output_bias_key = "network.20.bias"
            folded_state[output_weight_key].copy_(
                torch.arange(folded_state[output_weight_key].numel(), dtype=torch.float32).reshape_as(
                    folded_state[output_weight_key]
                )
            )
            folded_state[output_bias_key].copy_(torch.arange(len(LABELS), dtype=torch.float32))
            torch.save(
                {
                    "model_state_dict": folded_state,
                    "labels": LABELS,
                    "model_type": "cnn",
                },
                checkpoint_path,
            )

            mixed_model = MODEL_CLASSES["cnn"](num_classes=len(MIXEDCASE_LABELS))
            initialized = initialize_mixedcase_from_folded_checkpoint(
                mixed_model,
                "cnn",
                torch.device("cpu"),
                folded_weights_path=checkpoint_path,
            )

        self.assertTrue(initialized)
        mixed_state = mixed_model.state_dict()
        self.assertTrue(torch.equal(mixed_state[output_weight_key][: len(LABELS)], folded_state[output_weight_key]))
        self.assertTrue(torch.equal(mixed_state[output_bias_key][: len(LABELS)], folded_state[output_bias_key]))
        self.assertTrue(torch.equal(mixed_state[output_weight_key][36], folded_state[output_weight_key][10]))
        self.assertEqual(float(mixed_state[output_bias_key][36]), float(folded_state[output_bias_key][10]))

    def test_nist_sd19_hex_labels_map_to_mixedcase_targets(self) -> None:
        self.assertEqual(_nist_sd19_label_from_hex("30"), 0)
        self.assertEqual(_nist_sd19_label_from_hex("41"), 10)
        self.assertEqual(_nist_sd19_label_from_hex("5a"), 35)
        self.assertEqual(_nist_sd19_label_from_hex("61"), 36)
        self.assertEqual(_nist_sd19_label_from_hex("7a"), 61)
        self.assertIsNone(_nist_sd19_label_from_hex("2f"))

    def test_loads_character_corrections_with_saved_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            image_id = "abc123"
            image = Image.new("RGB", (80, 80), "white")
            draw = ImageDraw.Draw(image)
            draw.line((20, 15, 20, 65), fill="black", width=5)
            draw.line((20, 40, 52, 40), fill="black", width=5)
            draw.line((52, 15, 52, 65), fill="black", width=5)
            image.save(upload_dir / f"{image_id}.png")
            corrections_path = root / "corrections.jsonl"
            corrections_path.write_text(
                json.dumps(
                    {
                        "correction_kind": "character",
                        "image_id": image_id,
                        "corrected_label": "H",
                        "bbox": {"x": 10, "y": 10, "width": 55, "height": 60},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = load_correction_cache(["0", "H"], corrections_path, upload_dir)

        self.assertIsNotNone(loaded)
        images, targets = loaded
        self.assertEqual(tuple(images.shape), (1, 1, 28, 28))
        self.assertEqual(targets.tolist(), [1])

    def test_loads_sequence_corrections_when_boxes_match_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            image_id = "seq123"
            image = Image.new("RGB", (100, 80), "white")
            draw = ImageDraw.Draw(image)
            draw.line((10, 15, 10, 65), fill="black", width=5)
            draw.line((10, 40, 35, 40), fill="black", width=5)
            draw.line((35, 15, 35, 65), fill="black", width=5)
            draw.line((62, 20, 62, 62), fill="black", width=5)
            image.save(upload_dir / f"{image_id}.png")
            corrections_path = root / "corrections.jsonl"
            corrections_path.write_text(
                json.dumps(
                    {
                        "correction_kind": "sequence",
                        "image_id": image_id,
                        "corrected_label": "Hi",
                        "prediction_boxes": [
                            {"original_label": "H", "bbox": {"x": 5, "y": 10, "width": 38, "height": 60, "row": 1}},
                            {"original_label": "L", "bbox": {"x": 54, "y": 10, "width": 20, "height": 60, "row": 1}},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = load_correction_cache(["H", "i"], corrections_path, upload_dir)

        self.assertIsNotNone(loaded)
        images, targets = loaded
        self.assertEqual(tuple(images.shape), (2, 1, 28, 28))
        self.assertEqual(targets.tolist(), [0, 1])

    def test_loads_legacy_sequence_corrections_by_resegmenting_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            image_id = "legacy123"
            image = Image.new("RGB", (120, 80), "white")
            draw = ImageDraw.Draw(image)
            draw.line((20, 15, 20, 65), fill="black", width=5)
            draw.line((75, 15, 75, 65), fill="black", width=5)
            image.save(upload_dir / f"{image_id}.png")
            corrections_path = root / "corrections.jsonl"
            corrections_path.write_text(
                json.dumps(
                    {
                        "correction_kind": "sequence",
                        "image_id": image_id,
                        "corrected_label": "11",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = load_correction_cache(["0", "1"], corrections_path, upload_dir)

        self.assertIsNotNone(loaded)
        images, targets = loaded
        self.assertEqual(tuple(images.shape), (2, 1, 28, 28))
        self.assertEqual(targets.tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
