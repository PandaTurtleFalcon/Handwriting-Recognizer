import tempfile
import unittest
from pathlib import Path

from scripts.prepare_uji_pen_v2 import convert_uji_dataset, parse_uji_samples, rasterize_sample


class PrepareUjiPenV2Tests(unittest.TestCase):
    """Regression tests for UJI stroke parsing and rasterization."""

    def test_parse_and_rasterize_ascii_sample(self) -> None:
        """A UJI WORD block should become one nonblank grayscale image."""

        text = """
// ASCII char: A
WORD A trn_UJI_W01-01
  NUMSTROKES 2
  POINTS 3 # 0 10 5 0 10 10
  POINTS 2 # 2 6 8 6
WORD euro trn_UJI_W01-01
  NUMSTROKES 1
  POINTS 2 # 0 0 10 10
"""

        samples = parse_uji_samples(text)
        image = rasterize_sample(samples[0])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].label, "A")
        self.assertEqual(image.size, (96, 96))
        self.assertLess(min(image.getdata()), 255)

    def test_convert_writes_ascii_code_folders(self) -> None:
        """Prepared images should use the same ASCII-code folders as character data."""

        text = """
WORD ! trn_UJI_W01-01
  NUMSTROKES 1
  POINTS 2 # 0 0 0 10
"""

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "uji.txt"
            output = Path(directory) / "out"
            input_path.write_text(text, encoding="utf-8")
            counts = convert_uji_dataset(input_path, output)

            self.assertEqual(counts, {"!": 1})
            self.assertEqual(len(list((output / "33").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
