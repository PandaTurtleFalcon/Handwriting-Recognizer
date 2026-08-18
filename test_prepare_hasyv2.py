import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_hasyv2 import ascii_label, prepare_hasy_subset


class PrepareHasyV2Tests(unittest.TestCase):
    """Regression tests for HASYv2 label preparation."""

    def test_ascii_label_maps_latex_punctuation_aliases(self) -> None:
        """Useful HASY math labels should feed supported ASCII punctuation."""

        self.assertEqual(ascii_label(r"\cdot"), str(ord(".")))
        self.assertEqual(ascii_label(r"\bullet"), str(ord(".")))
        self.assertEqual(ascii_label(r"\prime"), str(ord("'")))
        self.assertEqual(ascii_label(r"\mid"), str(ord("|")))
        self.assertEqual(ascii_label(r"\vdots"), str(ord(":")))
        self.assertEqual(ascii_label(r"\setminus"), str(ord("/")))

    def test_prepare_hasy_subset_copies_aliases_to_ascii_folders(self) -> None:
        """Alias rows should be copied into ASCII-code image-folder targets."""

        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace) / "extracted"
            image_dir = root / "hasy-data"
            output = Path(workspace) / "out"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "sample.png"
            Image.new("L", (4, 4), 255).save(image_path)
            with (root / "hasy-data-labels.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "symbol_id", "latex", "user_id"])
                writer.writeheader()
                writer.writerow(
                    {
                        "path": "hasy-data/sample.png",
                        "symbol_id": "184",
                        "latex": r"\cdot",
                        "user_id": "1",
                    }
                )

            counts = prepare_hasy_subset(root, output, "ascii")
            self.assertEqual(counts[str(ord("."))], 1)
            self.assertTrue((output / str(ord(".")) / "sample.png").exists())


if __name__ == "__main__":
    unittest.main()
