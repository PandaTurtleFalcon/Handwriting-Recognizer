import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.prepare_the_dataset import ORIENTATION_TRANSFORMS, convert_the_csv, parse_the_rows, the_label_to_mixedcase_index


class PrepareTheDatasetTests(unittest.TestCase):
    """Regression tests for T-H-E Dataset Version IV conversion."""

    def test_label_mapping_preserves_mixed_case_order(self) -> None:
        self.assertEqual(the_label_to_mixedcase_index(1), 36)
        self.assertEqual(the_label_to_mixedcase_index(26), 61)
        self.assertEqual(the_label_to_mixedcase_index(40), 10)
        self.assertEqual(the_label_to_mixedcase_index(65), 35)
        self.assertIsNone(the_label_to_mixedcase_index(66))

    def test_parse_rows_builds_normalized_tensor_cache(self) -> None:
        a_pixels = ["0"] * 784
        a_pixels[14 * 28 + 14] = "1"
        z_pixels = ["0"] * 784
        z_pixels[10 * 28 + 8] = "255"

        images, targets = parse_the_rows([["1", *a_pixels], ["65", *z_pixels]])

        self.assertEqual(tuple(images.shape), (2, 1, 28, 28))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(targets.tolist(), [36, 35])

    def test_parse_rows_transposes_version_four_pixels_by_default(self) -> None:
        pixels = np.zeros((28, 28), dtype=np.float32)
        pixels[2, 25] = 1.0

        transposed = ORIENTATION_TRANSFORMS["transpose"](pixels)
        raw = ORIENTATION_TRANSFORMS["raw"](pixels)

        self.assertEqual(float(transposed[25, 2]), 1.0)
        self.assertEqual(float(raw[2, 25]), 1.0)

    def test_convert_csv_writes_cache_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            csv_path = Path(workspace) / "version4.csv"
            output_path = Path(workspace) / "mixedcase.pt"
            pixels = ["0"] * 784
            pixels[8 * 28 + 12] = "1"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["1", *pixels])
                writer.writerow(["40", *pixels])

            report = convert_the_csv(csv_path, output_path)
            cache = torch.load(output_path, map_location="cpu", weights_only=True)

        self.assertEqual(report["images"], 2)
        self.assertEqual(report["classes"], 2)
        self.assertEqual(report["orientation"], "transpose")
        self.assertEqual(cache["targets"].tolist(), [36, 10])

    def test_parse_rows_rejects_bad_column_count(self) -> None:
        with self.assertRaises(ValueError):
            parse_the_rows([["1", "0"]])


if __name__ == "__main__":
    unittest.main()
