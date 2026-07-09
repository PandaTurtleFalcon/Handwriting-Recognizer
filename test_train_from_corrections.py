import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from scripts import train_from_corrections
from scripts.train_from_corrections import export_character_correction_folder
from scripts.train_from_corrections import (
    correction_item_label_counts,
    exported_character_crop_counts,
    format_priority_coverage,
)


class TrainFromCorrectionsTests(unittest.TestCase):
    def test_exports_sequence_corrections_as_character_ascii_folders(self) -> None:
        """Daily training should expose saved sequence corrections to character_model."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            image_id = "abc123"
            image = Image.new("RGB", (48, 24), "white")
            draw = ImageDraw.Draw(image)
            draw.line((6, 4, 6, 20), fill="black", width=2)
            draw.line((20, 4, 20, 20), fill="black", width=2)
            draw.line((6, 12, 20, 12), fill="black", width=2)
            image.save(upload_dir / f"{image_id}.png")

            corrections_path = root / "corrections.jsonl"
            corrections_path.write_text(
                json.dumps(
                    {
                        "correction_kind": "sequence",
                        "image_id": image_id,
                        "corrected_label": "Hi",
                        "prediction_boxes": [
                            {"original_label": "H", "bbox": {"x": 0, "y": 0, "width": 16, "height": 24, "row": 1}},
                            {"original_label": "L", "bbox": {"x": 16, "y": 0, "width": 16, "height": 24, "row": 1}},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_root = root / "character_ascii"
            count = export_character_correction_folder(
                ["H", "i"],
                output_root=output_root,
                corrections_path=corrections_path,
                upload_dir=upload_dir,
            )

            self.assertEqual(count, 2)
            self.assertEqual(len(list((output_root / str(ord("H"))).glob("*.png"))), 1)
            self.assertEqual(len(list((output_root / str(ord("i"))).glob("*.png"))), 1)

    def test_parser_help_is_available_without_training(self) -> None:
        """The daily training script should expose safe CLI help."""

        help_text = train_from_corrections.build_parser().format_help()

        self.assertIn("--dry-run", help_text)
        self.assertIn("--min-character-corrections", help_text)
        self.assertIn("--min-alnum-corrections", help_text)
        self.assertIn("--priority-labels", help_text)
        self.assertIn("--mixedcase-priority-labels", help_text)

    def test_counts_exported_character_crops_by_priority_label(self) -> None:
        """Dry-run coverage should show which weak labels have examples."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, count in {"O": 2, "l": 1}.items():
                class_dir = root / str(ord(label))
                class_dir.mkdir(parents=True)
                for index in range(count):
                    Image.new("L", (8, 8), 255).save(class_dir / f"{index}.png")

            counts = exported_character_crop_counts(root)

        self.assertEqual(counts["O"], 2)
        self.assertEqual(counts["l"], 1)
        self.assertEqual(format_priority_coverage(counts, "Olo"), "O:2, l:1, o:0")

    def test_counts_loaded_correction_items_by_label(self) -> None:
        """Dry-run coverage should decode cached target indices into labels."""

        counts = correction_item_label_counts(["0", "1", "A", "a"], (object(), [1, 1, 3, 99]))

        self.assertEqual(counts["1"], 2)
        self.assertEqual(counts["a"], 1)
        self.assertNotIn("A", counts)

    def test_main_skips_tiny_correction_sets_without_force(self) -> None:
        """A tiny user-labeled set should not trigger daily fine-tuning by default."""

        fake_corrections = (object(), [0, 1])
        output = io.StringIO()
        with (
            patch.object(train_from_corrections, "load_correction_cache", return_value=fake_corrections),
            patch.object(train_from_corrections, "load_character_labels", return_value=["A"]),
            patch.object(train_from_corrections, "export_character_correction_folder", return_value=2),
            patch.object(train_from_corrections, "train_character_model") as train_character,
            patch.object(train_from_corrections, "train") as train_folded,
            patch.object(train_from_corrections, "train_mixedcase") as train_mixed,
            contextlib.redirect_stdout(output),
        ):
            train_from_corrections.main([])

        self.assertIn("Only 2 character correction samples", output.getvalue())
        self.assertIn("Only 2 folded alnum correction samples", output.getvalue())
        self.assertIn("Only 2 mixed-case correction samples", output.getvalue())
        train_character.assert_not_called()
        train_folded.assert_not_called()
        train_mixed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
