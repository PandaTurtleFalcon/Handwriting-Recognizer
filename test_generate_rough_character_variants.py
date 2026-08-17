import tempfile
import unittest
from pathlib import Path

from scripts.generate_rough_character_variants import generate_rough_character_variants, render_rough_character


class GenerateRoughCharacterVariantsTests(unittest.TestCase):
    """Regression tests for rough handwritten character generation."""

    def test_generates_ascii_code_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "rough"

            generate_rough_character_variants(output_root, labels="A1!", samples_per_label=2, seed=7)

            for label in "A1!":
                class_dir = output_root / str(ord(label))
                self.assertTrue(class_dir.exists())
                self.assertEqual(len(list(class_dir.glob("*.png"))), 2)

    def test_rendered_character_has_ink(self) -> None:
        image = render_rough_character("H", seed=13)

        self.assertEqual(image.mode, "L")
        self.assertLess(image.getextrema()[0], 128)

    def test_generation_is_deterministic_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_root = Path(first_dir) / "rough"
            second_root = Path(second_dir) / "rough"

            generate_rough_character_variants(first_root, labels="H", samples_per_label=1, seed=11)
            generate_rough_character_variants(second_root, labels="H", samples_per_label=1, seed=11)

            first_bytes = (first_root / str(ord("H")) / "0000.png").read_bytes()
            second_bytes = (second_root / str(ord("H")) / "0000.png").read_bytes()
            self.assertEqual(first_bytes, second_bytes)

    def test_rejects_multi_character_label_items(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one character"):
            render_rough_character("Hi", seed=1)


if __name__ == "__main__":
    unittest.main()
