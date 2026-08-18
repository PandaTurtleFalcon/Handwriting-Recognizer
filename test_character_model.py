import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from character_model import (
    _alnum_should_override,
    _combined_cache_path,
    _correction_memory_override_label,
    _digit_beats_ambiguous_letter,
    _letter_should_override,
    _looks_like_four,
    _looks_like_one,
    _looks_like_seven,
    _postprocess_colons,
    _postprocess_dot_height,
    _postprocess_exclamations,
    _postprocess_lowercase_i,
    _punctuation_shape_label,
    _record_with_legacy_correction_boxes,
    _split_touching_character_regions,
    attach_character_pair_rules,
    attach_character_logit_bias,
    build_or_load_combined_cache,
    character_loss_weights,
    FocalCrossEntropyLoss,
    labels_match_with_ambiguity,
    load_correction_memory_exemplars,
    stratified_split_indices,
)
from alnum_model import attach_mixedcase_logit_bias
from mnist_model import DigitRegion, segment_digit_regions
from PIL import Image, ImageDraw
import torch


class CharacterPostprocessingTests(unittest.TestCase):
    """Regression tests for model-independent character cleanup rules."""

    def test_extra_ascii_folder_data_is_added_to_character_cache(self) -> None:
        """Extra ASCII-code folders should merge into the character labels."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "base"
            extra = Path(directory) / "extra"
            (root / "65").mkdir(parents=True)
            (root / "66").mkdir(parents=True)
            (extra / "65").mkdir(parents=True)
            for image_path in [root / "65" / "a.png", root / "66" / "b.png", extra / "65" / "extra-a.png"]:
                image = Image.new("L", (24, 24), 255)
                draw = ImageDraw.Draw(image)
                draw.line((6, 18, 12, 5, 18, 18), fill=0, width=2)
                image.save(image_path)

            cache_path = _combined_cache_path(root, [extra])
            images, targets, labels = build_or_load_combined_cache(root, [extra])

            self.assertTrue(cache_path.exists())
            self.assertEqual(labels, ["A", "B"])
            self.assertEqual(len(images), 3)
            self.assertEqual(targets.tolist().count(0), 2)
            self.assertEqual(targets.tolist().count(1), 1)

    def test_stratified_split_indices_is_deterministic_and_balanced(self) -> None:
        """The local splitter should replace sklearn's stratified split behavior."""

        indices = list(range(30))
        labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 10

        first_train, first_validation = stratified_split_indices(indices, 0.2, 42, labels)
        second_train, second_validation = stratified_split_indices(indices, 0.2, 42, labels)

        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)
        self.assertEqual(len(first_validation), 6)
        validation_labels = [labels[index] for index in first_validation]
        self.assertEqual(validation_labels.count("A"), 2)
        self.assertEqual(validation_labels.count("B"), 2)
        self.assertEqual(validation_labels.count("C"), 2)
        self.assertEqual(set(first_train).isdisjoint(first_validation), True)

    def test_labels_match_with_visual_ambiguity_groups(self) -> None:
        """Ambiguity-aware scoring should accept known handwriting lookalikes."""

        self.assertTrue(labels_match_with_ambiguity("S", "s"))
        self.assertTrue(labels_match_with_ambiguity("0", "O"))
        self.assertTrue(labels_match_with_ambiguity("1", "|"))
        self.assertTrue(labels_match_with_ambiguity("l", "i"))
        self.assertTrue(labels_match_with_ambiguity("_", "-"))
        self.assertTrue(labels_match_with_ambiguity(".", "'"))
        self.assertTrue(labels_match_with_ambiguity(":", "i"))

    def test_character_logit_bias_attaches_to_matching_model(self) -> None:
        """A matching calibration artifact should shift model logits."""

        model = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(model.weight)
        labels = ["A", "B"]
        with tempfile.TemporaryDirectory() as directory:
            bias_path = Path(directory) / "bias.pt"
            torch.save({"labels": labels, "bias": torch.tensor([0.25, -0.25])}, bias_path)

            attached = attach_character_logit_bias(model, labels, torch.device("cpu"), bias_path)
            output = model(torch.ones((1, 2)))

        self.assertTrue(attached)
        self.assertTrue(hasattr(model, "character_logit_bias"))
        self.assertTrue(torch.allclose(output, torch.tensor([[0.25, -0.25]])))

    def test_character_logit_bias_rejects_mismatched_labels(self) -> None:
        """A stale calibration artifact must not alter model outputs."""

        model = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(model.weight)
        labels = ["A", "B"]
        with tempfile.TemporaryDirectory() as directory:
            bias_path = Path(directory) / "bias.pt"
            torch.save({"labels": ["A", "C"], "bias": torch.tensor([0.25, -0.25])}, bias_path)

            attached = attach_character_logit_bias(model, labels, torch.device("cpu"), bias_path)
            output = model(torch.ones((1, 2)))

        self.assertFalse(attached)
        self.assertFalse(hasattr(model, "character_logit_bias"))
        self.assertTrue(torch.allclose(output, torch.zeros((1, 2))))

    def test_character_logit_bias_rejects_mismatched_checkpoint_hash(self) -> None:
        """A fingerprinted calibration artifact must match the active weights."""

        model = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(model.weight)
        labels = ["A", "B"]
        with tempfile.TemporaryDirectory() as directory:
            bias_path = Path(directory) / "bias.pt"
            weights_path = Path(directory) / "character_cnn.pt"
            weights_path.write_bytes(b"current weights")
            torch.save(
                {
                    "labels": labels,
                    "bias": torch.tensor([-0.2, 0.2]),
                    "checkpoint_sha256": "not-the-current-checkpoint",
                },
                bias_path,
            )

            attached = attach_character_logit_bias(model, labels, torch.device("cpu"), bias_path, weights_path)
            output = model(torch.ones((1, 2)))

        self.assertFalse(attached)
        self.assertFalse(hasattr(model, "character_logit_bias"))
        self.assertTrue(torch.allclose(output, torch.zeros((1, 2))))

    def test_character_pair_rules_flip_close_visual_twin(self) -> None:
        """A matching pair-rule artifact should flip only close alternatives."""

        class FixedLogitModel(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor(
                    [
                        [0.40, 0.30, 0.00],
                        [0.40, 0.10, 0.00],
                    ],
                    dtype=torch.float32,
                )

        labels = ["0", "O", "A"]
        with tempfile.TemporaryDirectory() as directory:
            rules_path = Path(directory) / "rules.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "labels": labels,
                        "rules": [{"from": "0", "to": "O", "threshold": -0.15}],
                    }
                ),
                encoding="utf-8",
            )
            model = FixedLogitModel()

            attached = attach_character_pair_rules(model, labels, torch.device("cpu"), rules_path)
            predictions = model(torch.zeros((2, 1, 32, 32))).argmax(dim=1).tolist()

        self.assertTrue(attached)
        self.assertEqual(predictions, [1, 0])
        self.assertTrue(labels_match_with_ambiguity(";", "!"))
        self.assertTrue(labels_match_with_ambiguity("q", "9"))
        self.assertTrue(labels_match_with_ambiguity("T", "7"))
        self.assertFalse(labels_match_with_ambiguity("A", "B"))

    def test_character_pair_rules_reject_mismatched_checkpoint_hash(self) -> None:
        """Stale pair rules must not attach to a different checkpoint."""

        class FixedLogitModel(torch.nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.tensor([[0.40, 0.30]], dtype=torch.float32)

        labels = ["0", "O"]
        with tempfile.TemporaryDirectory() as directory:
            rules_path = Path(directory) / "rules.json"
            weights_path = Path(directory) / "character_cnn.pt"
            weights_path.write_bytes(b"current weights")
            rules_path.write_text(
                json.dumps(
                    {
                        "labels": labels,
                        "checkpoint_sha256": "not-the-current-checkpoint",
                        "rules": [{"from": "0", "to": "O", "threshold": -0.15}],
                    }
                ),
                encoding="utf-8",
            )
            model = FixedLogitModel()

            attached = attach_character_pair_rules(model, labels, torch.device("cpu"), rules_path, weights_path)
            prediction = int(model(torch.zeros((1, 1, 32, 32))).argmax(dim=1).item())

        self.assertFalse(attached)
        self.assertFalse(hasattr(model, "character_pair_rules"))
        self.assertEqual(prediction, 0)

    def test_mixedcase_logit_bias_attaches_to_matching_model(self) -> None:
        """A matching mixed-case calibration artifact should shift model logits."""

        model = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(model.weight)
        labels = ["A", "a"]
        with tempfile.TemporaryDirectory() as directory:
            bias_path = Path(directory) / "mixedcase_bias.pt"
            torch.save({"labels": labels, "bias": torch.tensor([-0.1, 0.3])}, bias_path)

            attached = attach_mixedcase_logit_bias(model, labels, torch.device("cpu"), bias_path)
            output = model(torch.ones((1, 2)))

        self.assertTrue(attached)
        self.assertTrue(hasattr(model, "mixedcase_logit_bias"))
        self.assertTrue(torch.allclose(output, torch.tensor([[-0.1, 0.3]])))

    def test_mixedcase_logit_bias_rejects_mismatched_checkpoint_hash(self) -> None:
        """Mixed-case bias artifacts are tied to the checkpoint they calibrated."""

        model = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(model.weight)
        labels = ["A", "a"]
        with tempfile.TemporaryDirectory() as directory:
            bias_path = Path(directory) / "mixedcase_bias.pt"
            weights_path = Path(directory) / "mixedcase_cnn.pt"
            weights_path.write_bytes(b"current weights")
            torch.save(
                {
                    "labels": labels,
                    "bias": torch.tensor([-0.1, 0.3]),
                    "checkpoint_sha256": "not-the-current-checkpoint",
                },
                bias_path,
            )

            attached = attach_mixedcase_logit_bias(model, labels, torch.device("cpu"), bias_path, weights_path)
            output = model(torch.ones((1, 2)))

        self.assertFalse(attached)
        self.assertFalse(hasattr(model, "mixedcase_logit_bias"))
        self.assertTrue(torch.allclose(output, torch.zeros((1, 2))))

    def test_character_loss_weights_can_emphasize_punctuation(self) -> None:
        weights = character_loss_weights(["A", "7", "!", "."], punctuation_weight=2.5)

        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertEqual(weights.tolist(), [1.0, 1.0, 2.5, 2.5])
        self.assertIsNone(character_loss_weights(["A", "!"], punctuation_weight=1.0))

    def test_character_loss_weights_can_emphasize_weak_labels(self) -> None:
        weights = character_loss_weights(["O", "0", "-", "_"], punctuation_weight=1.5, weak_labels="O-", weak_weight=2.0)

        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertEqual(weights.tolist(), [2.0, 1.0, 3.0, 1.5])

    def test_load_correction_memory_exemplars_uses_sequence_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            image_id = "abc123"
            image = Image.new("L", (80, 60), 255)
            draw = ImageDraw.Draw(image)
            draw.ellipse((20, 15, 55, 45), outline=0, width=4)
            image.save(upload_dir / f"{image_id}.png")
            corrections_path = root / "corrections.jsonl"
            corrections_path.write_text(
                json.dumps(
                    {
                        "correction_kind": "sequence",
                        "image_id": image_id,
                        "corrected_label": "a",
                        "prediction_boxes": [
                            {"bbox": {"x": 15, "y": 10, "width": 45, "height": 40, "row": 1}},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            exemplars = load_correction_memory_exemplars(["2", "a"], corrections_path, upload_dir)

        self.assertIsNotNone(exemplars)
        assert exemplars is not None
        images, targets = exemplars
        self.assertEqual(tuple(images.shape), (1, 1, 32, 32))
        self.assertEqual(targets.tolist(), [1])

    def test_load_correction_memory_exemplars_ignores_sequence_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            image_id = "phrase123"
            image = Image.new("L", (80, 40), 255)
            draw = ImageDraw.Draw(image)
            draw.line((8, 8, 8, 32), fill=0, width=3)
            draw.ellipse((42, 8, 66, 32), outline=0, width=3)
            image.save(upload_dir / f"{image_id}.png")
            corrections_path = root / "corrections.jsonl"
            corrections_path.write_text(
                json.dumps(
                    {
                        "correction_kind": "sequence",
                        "image_id": image_id,
                        "corrected_label": "l o",
                        "prediction_boxes": [
                            {"bbox": {"x": 0, "y": 0, "width": 28, "height": 40, "row": 1}},
                            {"bbox": {"x": 36, "y": 0, "width": 40, "height": 40, "row": 1}},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            exemplars = load_correction_memory_exemplars(["l", "o"], corrections_path, upload_dir)

        self.assertIsNotNone(exemplars)
        assert exemplars is not None
        _, targets = exemplars
        self.assertEqual(targets.tolist(), [0, 1])

    def test_load_correction_memory_exemplars_recovers_legacy_sequence_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            image_id = "legacy123"
            image = Image.new("L", (120, 80), 255)
            draw = ImageDraw.Draw(image)
            draw.line((25, 15, 25, 65), fill=0, width=5)
            draw.line((78, 18, 100, 18), fill=0, width=5)
            draw.line((100, 18, 76, 63), fill=0, width=5)
            draw.line((76, 63, 103, 63), fill=0, width=5)
            image.save(upload_dir / f"{image_id}.png")
            corrections_path = root / "corrections.jsonl"
            corrections_path.write_text(
                json.dumps(
                    {
                        "correction_kind": "sequence",
                        "image_id": image_id,
                        "corrected_label": "17",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            exemplars = load_correction_memory_exemplars(["1", "7"], corrections_path, upload_dir)

        self.assertIsNotNone(exemplars)
        assert exemplars is not None
        images, targets = exemplars
        self.assertEqual(tuple(images.shape), (2, 1, 32, 32))
        self.assertEqual(targets.tolist(), [0, 1])

    def test_legacy_sequence_recovery_uses_touching_character_split(self) -> None:
        image = Image.new("L", (80, 60), 255)
        joined = DigitRegion(image=image.crop((10, 10, 70, 50)), box=(10, 10, 70, 50), row=1)
        left = DigitRegion(image=image.crop((10, 10, 35, 50)), box=(10, 10, 35, 50), row=1)
        right = DigitRegion(image=image.crop((36, 10, 70, 50)), box=(36, 10, 70, 50), row=1)
        record = {"correction_kind": "sequence", "corrected_label": "AB"}

        with patch("character_model.segment_digit_regions", return_value=[joined]):
            with patch("character_model._split_touching_character_regions", return_value=[left, right]) as mock_split:
                recovered = _record_with_legacy_correction_boxes(record, image)

        self.assertEqual(len(recovered["prediction_boxes"]), 2)
        self.assertEqual(recovered["prediction_boxes"][0]["bbox"]["width"], 25)
        mock_split.assert_called_once_with([joined])

    def test_correction_memory_overrides_only_low_confidence_close_matches(self) -> None:
        class MemoryModel:
            pass

        model = MemoryModel()
        model.correction_exemplars = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        model.correction_exemplar_targets = torch.tensor([1], dtype=torch.long)
        tensor = torch.zeros((1, 1, 32, 32), dtype=torch.float32)

        self.assertEqual(_correction_memory_override_label(model, ["2", "a"], tensor, 0.94), ("a", 0.0))
        self.assertIsNone(_correction_memory_override_label(model, ["2", "a"], tensor, 0.95))

    def test_correction_memory_requires_distance_and_label_margin(self) -> None:
        class MemoryModel:
            pass

        model = MemoryModel()
        model.correction_exemplars = torch.stack(
            [
                torch.zeros((1, 32, 32), dtype=torch.float32),
                torch.full((1, 32, 32), 0.03, dtype=torch.float32),
            ]
        )
        model.correction_exemplar_targets = torch.tensor([1, 0], dtype=torch.long)
        tensor = torch.zeros((1, 1, 32, 32), dtype=torch.float32)

        self.assertIsNone(_correction_memory_override_label(model, ["2", "a"], tensor, 0.80))

        model.correction_exemplars = torch.full((1, 1, 32, 32), 1.0, dtype=torch.float32)
        model.correction_exemplar_targets = torch.tensor([1], dtype=torch.long)

        self.assertIsNone(_correction_memory_override_label(model, ["2", "a"], tensor, 0.80))

    def test_focal_cross_entropy_downweights_easy_examples(self) -> None:
        criterion = FocalCrossEntropyLoss(gamma=1.0)
        logits = torch.tensor([[4.0, -1.0], [0.2, 0.0]], dtype=torch.float32)
        targets = torch.tensor([0, 0], dtype=torch.long)

        losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        focal_loss = criterion(logits, targets)

        self.assertLess(focal_loss.item(), losses.mean().item())
        self.assertGreater(focal_loss.item(), 0.0)

    def test_split_dot_above_stem_becomes_lowercase_i(self) -> None:
        """A detached dot above a skinny stem should read as lowercase i."""

        predictions = [
            {"label": "H", "confidence": 0.97, "x": 40, "y": 55, "width": 90, "height": 120, "row": 1},
            {"label": ":", "confidence": 0.91, "x": 184, "y": 40, "width": 12, "height": 12, "row": 1},
            {"label": "L", "confidence": 0.81, "x": 180, "y": 70, "width": 14, "height": 105, "row": 1},
        ]

        cleaned = _postprocess_lowercase_i(predictions)

        self.assertEqual("".join(str(item["label"]) for item in cleaned), "Hi")
        self.assertEqual(len(cleaned), 2)
        self.assertGreaterEqual(float(cleaned[1]["confidence"]), 0.9)

    def test_split_colon_dots_are_merged(self) -> None:
        """Two vertically aligned small dots should become one colon."""

        predictions = [
            {"label": "Q", "confidence": 0.80, "x": 50, "y": 30, "width": 14, "height": 14, "row": 1},
            {"label": "Q", "confidence": 0.80, "x": 51, "y": 76, "width": 14, "height": 14, "row": 1},
        ]

        cleaned = _postprocess_colons(predictions)

        self.assertEqual("".join(str(item["label"]) for item in cleaned), ":")
        self.assertEqual(len(cleaned), 1)

    def test_split_dot_below_stem_becomes_exclamation(self) -> None:
        """A detached dot below a skinny stem should read as exclamation."""

        predictions = [
            {"label": "1", "confidence": 0.98, "x": 50, "y": 15, "width": 12, "height": 80, "row": 1},
            {"label": "0", "confidence": 0.81, "x": 48, "y": 112, "width": 15, "height": 15, "row": 1},
        ]

        cleaned = _postprocess_exclamations(predictions)

        self.assertEqual("".join(str(item["label"]) for item in cleaned), "!")
        self.assertEqual(len(cleaned), 1)

    def test_low_dot_in_text_row_becomes_period(self) -> None:
        """Row position should fix period/apostrophe crop ambiguity."""

        predictions = [
            {"label": "H", "confidence": 0.97, "x": 10, "y": 20, "width": 50, "height": 90, "row": 1},
            {"label": ":", "confidence": 0.72, "x": 70, "y": 98, "width": 10, "height": 10, "row": 1},
        ]

        cleaned = _postprocess_dot_height(predictions)

        self.assertEqual([item["label"] for item in cleaned], ["H", "."])
        self.assertGreaterEqual(float(cleaned[1]["confidence"]), 0.9)

    def test_high_dot_in_text_row_becomes_apostrophe(self) -> None:
        """High standalone dots in a text row should stay apostrophe-like."""

        predictions = [
            {"label": "H", "confidence": 0.97, "x": 10, "y": 30, "width": 50, "height": 90, "row": 1},
            {"label": ".", "confidence": 0.72, "x": 70, "y": 24, "width": 10, "height": 10, "row": 1},
        ]

        cleaned = _postprocess_dot_height(predictions)

        self.assertEqual([item["label"] for item in cleaned], ["H", "'"])

    def test_isolated_dot_keeps_model_label(self) -> None:
        """A dot-only row has no baseline, so the model label should remain."""

        predictions = [{"label": "'", "confidence": 0.72, "x": 70, "y": 98, "width": 10, "height": 10, "row": 1}]

        self.assertEqual(_postprocess_dot_height(predictions)[0]["label"], "'")

    def test_dot_postprocessing_does_not_merge_across_rows(self) -> None:
        """Detached dots should only merge with stems or dots on the same row."""

        colon = _postprocess_colons(
            [
                {"label": "Q", "confidence": 0.80, "x": 50, "y": 30, "width": 14, "height": 14, "row": 1},
                {"label": "Q", "confidence": 0.80, "x": 51, "y": 76, "width": 14, "height": 14, "row": 2},
            ]
        )
        lowercase_i = _postprocess_lowercase_i(
            [
                {"label": ":", "confidence": 0.91, "x": 184, "y": 40, "width": 12, "height": 12, "row": 1},
                {"label": "L", "confidence": 0.81, "x": 180, "y": 70, "width": 14, "height": 105, "row": 2},
            ]
        )
        exclamation = _postprocess_exclamations(
            [
                {"label": "1", "confidence": 0.98, "x": 50, "y": 15, "width": 12, "height": 80, "row": 1},
                {"label": "0", "confidence": 0.81, "x": 48, "y": 112, "width": 15, "height": 15, "row": 2},
            ]
        )

        self.assertEqual(len(colon), 2)
        self.assertEqual(len(lowercase_i), 2)
        self.assertEqual(len(exclamation), 2)

    def test_dot_below_stem_stays_exclamation_mark(self) -> None:
        """A dot below a vertical stem should stay an exclamation mark."""

        image = Image.new("L", (70, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.line((34, 14, 34, 78), fill=0, width=6)
        draw.ellipse((28, 96, 40, 108), fill=0)
        region = DigitRegion(image=image, box=(0, 0, 70, 120), row=1)

        self.assertEqual(_punctuation_shape_label(region), "!")

    def test_dot_above_stem_is_lowercase_i_shape(self) -> None:
        """A merged dot-above-stem shape should be classified as i."""

        image = Image.new("L", (70, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.ellipse((28, 12, 40, 24), fill=0)
        draw.line((34, 44, 34, 108), fill=0, width=6)
        region = DigitRegion(image=image, box=(0, 0, 70, 120), row=1)

        self.assertEqual(_punctuation_shape_label(region), "i")

    def test_parenthesis_shapes_are_detected(self) -> None:
        """Single-stroke parentheses should beat letter/digit guesses."""

        left = Image.new("L", (90, 150), 255)
        right = Image.new("L", (90, 150), 255)
        left_draw = ImageDraw.Draw(left)
        right_draw = ImageDraw.Draw(right)
        left_draw.arc((22, 8, 86, 142), start=105, end=255, fill=0, width=6)
        right_draw.arc((4, 8, 68, 142), start=-75, end=75, fill=0, width=6)

        self.assertEqual(_punctuation_shape_label(DigitRegion(image=left, box=(0, 0, 90, 150), row=1)), "(")
        self.assertEqual(_punctuation_shape_label(DigitRegion(image=right, box=(0, 0, 90, 150), row=1)), ")")

    def test_blank_seam_splits_touching_character_region(self) -> None:
        """A wide region with an internal blank seam should become two regions."""

        image = Image.new("L", (180, 130), 255)
        draw = ImageDraw.Draw(image)
        draw.arc((12, 20, 78, 82), start=20, end=330, fill=0, width=6)
        draw.line((28, 52, 76, 52), fill=0, width=6)
        draw.line((120, 18, 166, 18), fill=0, width=6)
        draw.line((120, 18, 120, 62), fill=0, width=6)
        draw.line((120, 62, 160, 62), fill=0, width=6)
        draw.arc((114, 58, 168, 116), start=-90, end=95, fill=0, width=6)
        region = DigitRegion(image=image, box=(10, 20, 190, 150), row=1)

        split = _split_touching_character_regions([region])

        self.assertEqual(len(split), 2)
        self.assertLess(split[0].box[2], split[1].box[0])

    def test_thin_bridge_splits_touching_character_region(self) -> None:
        """A small connecting stroke should not force two glyphs into one region."""

        image = Image.new("L", (180, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.arc((12, 18, 78, 84), start=15, end=335, fill=0, width=7)
        draw.line((30, 54, 76, 54), fill=0, width=7)
        draw.line((76, 56, 108, 56), fill=0, width=4)
        draw.line((120, 18, 164, 18), fill=0, width=7)
        draw.line((164, 18, 120, 104), fill=0, width=7)
        region = DigitRegion(image=image, box=(20, 40, 200, 160), row=1)

        split = _split_touching_character_regions([region])

        self.assertEqual(len(split), 2)
        self.assertLess(split[0].box[2], split[1].box[0] + 8)

    def test_touching_narrow_punctuation_region_is_split(self) -> None:
        """A tall skinny mark attached to a glyph should survive as its own region."""

        image = Image.new("L", (160, 130), 255)
        draw = ImageDraw.Draw(image)
        draw.line((20, 12, 20, 96), fill=0, width=6)
        draw.ellipse((15, 108, 25, 118), fill=0)
        draw.line((23, 58, 54, 58), fill=0, width=4)
        draw.arc((62, 28, 130, 94), start=15, end=335, fill=0, width=7)
        draw.line((78, 60, 126, 60), fill=0, width=7)
        region = DigitRegion(image=image, box=(30, 30, 190, 160), row=1)

        split = _split_touching_character_regions([region])

        self.assertEqual(len(split), 2)
        self.assertLessEqual(split[0].box[2] - split[0].box[0], 40)

    def test_shape_rule_identifies_plain_one(self) -> None:
        """A plain vertical stroke should remain digit 1, not letter L."""

        image = Image.new("L", (80, 180), 255)
        draw = ImageDraw.Draw(image)
        draw.line((38, 12, 38, 166), fill=0, width=6)
        region = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)[0]

        self.assertTrue(_looks_like_one(region))
        self.assertFalse(_looks_like_seven(region))

    def test_shape_rule_identifies_wide_top_seven(self) -> None:
        """A tall stroke with a wide top bar should be digit 7."""

        image = Image.new("L", (90, 220), 255)
        draw = ImageDraw.Draw(image)
        draw.line((18, 18, 72, 18), fill=0, width=7)
        draw.line((72, 18, 46, 206), fill=0, width=7)
        region = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)[0]

        self.assertTrue(_looks_like_seven(region))
        self.assertFalse(_looks_like_one(region))

    def test_shape_rule_identifies_open_top_four(self) -> None:
        """A handwritten 4 with a crossbar should stay numeric."""

        image = Image.new("L", (100, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.line((70, 10, 70, 108), fill=0, width=7)
        draw.line((70, 10, 25, 62), fill=0, width=7)
        draw.line((25, 62, 82, 62), fill=0, width=7)
        region = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)[0]

        self.assertTrue(_looks_like_four(region))
        self.assertFalse(_looks_like_one(region))

    def test_digit_rescue_handles_common_letter_confusions(self) -> None:
        """Confident digit votes should beat known letter lookalikes."""

        self.assertTrue(_digit_beats_ambiguous_letter("4", 0.98, "Y", 0.979))
        self.assertTrue(_digit_beats_ambiguous_letter("5", 0.95, "J", 0.99))
        self.assertTrue(_digit_beats_ambiguous_letter("2", 0.99, "Z", 0.94))
        self.assertTrue(_digit_beats_ambiguous_letter("1", 0.996, "I", 0.97))
        self.assertTrue(_digit_beats_ambiguous_letter("1", 0.996, "l", 0.97))
        self.assertTrue(_digit_beats_ambiguous_letter("0", 0.996, "O", 0.96))
        self.assertTrue(_digit_beats_ambiguous_letter("8", 0.996, "B", 0.96))
        self.assertTrue(_digit_beats_ambiguous_letter("5", 0.996, "S", 0.96))
        self.assertFalse(_digit_beats_ambiguous_letter("4", 0.91, "Y", 0.979))
        self.assertFalse(_digit_beats_ambiguous_letter("1", 0.996, "I", 0.99))
        self.assertFalse(_digit_beats_ambiguous_letter("0", 0.990, "O", 0.96))

    def test_letter_model_needs_margin_to_replace_digits(self) -> None:
        """A weak letter vote should not steal a stronger digit prediction."""

        self.assertFalse(_letter_should_override("5", 0.95, "S", 0.80, False))
        self.assertFalse(_letter_should_override("4", 0.98, "Y", 0.90, False))
        self.assertTrue(_letter_should_override("5", 0.80, "S", 0.93, False))
        self.assertFalse(_letter_should_override("5", 0.80, "S", 0.99, True))

    def test_letter_model_only_replaces_same_alphabetic_label(self) -> None:
        """Uppercase-only letter votes should not erase case or change letters."""

        self.assertTrue(_letter_should_override("S", 0.72, "S", 0.90, False))
        self.assertFalse(_letter_should_override("s", 0.72, "S", 0.99, False))
        self.assertFalse(_letter_should_override("H", 0.72, "L", 0.99, False))

    def test_alnum_model_needs_margin_to_flip_case(self) -> None:
        """Mixed-case predictions should not erase lowercase on tiny margins."""

        self.assertFalse(_alnum_should_override("s", 0.72, "S", 0.82, 0.0))
        self.assertTrue(_alnum_should_override("s", 0.72, "S", 0.91, 0.0))
        self.assertFalse(_alnum_should_override("S", 0.72, "s", 0.78, 0.0))
        self.assertTrue(_alnum_should_override("S", 0.72, "s", 0.84, 0.0))


if __name__ == "__main__":
    unittest.main()
