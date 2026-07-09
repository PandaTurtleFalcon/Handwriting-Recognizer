import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw

from alnum_model import (
    AugmentedTensorDataset,
    LABELS,
    MIXEDCASE_LABELS,
    MODEL_CLASSES,
    _chars74k_sample_label,
    _mixedcase_train_dataset,
    _nist_sd19_label_from_hex,
    build_or_load_mixedcase_ascii_folder_cache,
    evaluate_mixedcase_breakdown,
    initialize_mixedcase_from_folded_checkpoint,
    load_correction_cache,
    mixedcase_auxiliary_loss,
    mixedcase_folded_logits,
    mixedcase_folded_targets,
    mixedcase_loss_weights,
    mixedcase_labels_match_with_ambiguity,
    mixedcase_labels_match_with_visual_ambiguity,
    mixedcase_type_logits,
    mixedcase_type_targets,
)
from extra_alnum_datasets import load_labeled_image_folder


def tiny_transform(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.resize((28, 28)), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class ExtraAlnumDatasetTests(unittest.TestCase):
    """Regression tests for optional local alphanumeric datasets."""

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
