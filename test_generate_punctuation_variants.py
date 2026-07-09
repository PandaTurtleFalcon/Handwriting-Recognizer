import tempfile
import unittest
from pathlib import Path

from scripts.generate_punctuation_variants import TARGET_LABELS, generate_punctuation_variants


class GeneratePunctuationVariantsTests(unittest.TestCase):
    """Regression tests for synthetic punctuation generation."""

    def test_generates_ascii_code_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "punct"

            generate_punctuation_variants(output_root, samples_per_label=2, seed=7)

            for label in TARGET_LABELS:
                class_dir = output_root / str(ord(label))
                self.assertTrue(class_dir.exists())
                self.assertEqual(len(list(class_dir.glob("*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
