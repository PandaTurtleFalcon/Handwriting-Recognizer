import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from alnum_model import (
    MIXEDCASE_LABELS,
    _chars74k_sample_label,
    _nist_sd19_label_from_hex,
    build_or_load_mixedcase_ascii_folder_cache,
    load_correction_cache,
    mixedcase_labels_match_with_ambiguity,
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

    def test_mixedcase_ambiguity_groups_match_known_lookalikes(self) -> None:
        self.assertTrue(mixedcase_labels_match_with_ambiguity("S", "s"))
        self.assertTrue(mixedcase_labels_match_with_ambiguity("0", "O"))
        self.assertTrue(mixedcase_labels_match_with_ambiguity("1", "l"))
        self.assertTrue(mixedcase_labels_match_with_ambiguity("l", "i"))
        self.assertTrue(mixedcase_labels_match_with_ambiguity("q", "9"))
        self.assertTrue(mixedcase_labels_match_with_ambiguity("T", "7"))
        self.assertFalse(mixedcase_labels_match_with_ambiguity("A", "B"))

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
