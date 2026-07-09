import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.train_from_corrections import export_character_correction_folder


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


if __name__ == "__main__":
    unittest.main()
