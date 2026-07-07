import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mnist_model import WEIGHTS_PATH, load_model, mnist_normalize_image, predict_digits, segment_digit_regions, segment_digits


class MnistPreprocessingTests(unittest.TestCase):
    """Regression tests for digit preprocessing, segmentation, and prediction."""

    def test_normalized_digit_is_28_by_28_and_nonblank(self) -> None:
        """Normalization should produce a centered nonblank MNIST tensor image."""

        image = Image.new("L", (80, 80), 255)
        draw = ImageDraw.Draw(image)
        draw.line((38, 12, 38, 66), fill=0, width=9)

        normalized = mnist_normalize_image(image)
        pixels = np.asarray(normalized)

        self.assertEqual(normalized.size, (28, 28))
        self.assertGreater(pixels.sum(), 0)
        self.assertLessEqual(pixels.max(), 255)

    def test_segment_digits_splits_separated_components_left_to_right(self) -> None:
        """Separated marks should become separate crops in left-to-right order."""

        image = Image.new("L", (160, 72), 255)
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 12, 54, 58), outline=0, width=8)
        draw.line((112, 12, 112, 58), fill=0, width=8)

        digits = segment_digits(image)

        self.assertEqual(len(digits), 2)
        self.assertGreater(digits[0].size[0] * digits[0].size[1], 0)
        self.assertGreater(digits[1].size[0] * digits[1].size[1], 0)

    def test_segment_regions_keep_boxes_and_read_rows_top_to_bottom(self) -> None:
        """Detected boxes should preserve coordinates and row order."""

        image = Image.new("L", (180, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((112, 12, 132, 42), fill=0)
        draw.rectangle((18, 14, 38, 44), fill=0)
        draw.rectangle((20, 76, 40, 106), fill=0)
        draw.rectangle((116, 78, 136, 108), fill=0)

        regions = segment_digit_regions(image)
        boxes = [region.box for region in regions]

        self.assertEqual(len(regions), 4)
        self.assertLess(boxes[0][0], boxes[1][0])
        self.assertLess(boxes[0][1], boxes[2][1])
        self.assertLess(boxes[2][0], boxes[3][0])
        self.assertEqual([region.row for region in regions], [1, 1, 2, 2])

    def test_blank_image_returns_no_regions(self) -> None:
        """Blank uploads should not produce false handwriting regions."""

        image = Image.new("L", (80, 80), 255)

        regions = segment_digit_regions(image)

        self.assertEqual(regions, [])

    def test_segment_regions_supports_light_ink_on_dark_background(self) -> None:
        """Foreground detection should work for light ink on dark paper."""

        image = Image.new("L", (80, 80), 0)
        draw = ImageDraw.Draw(image)
        draw.line((40, 12, 40, 64), fill=255, width=7)

        regions = segment_digit_regions(image)

        self.assertEqual(len(regions), 1)
        self.assertLess(regions[0].image.size[0], 40)
        self.assertGreater(np.asarray(regions[0].image).sum(), 0)

    def test_character_segmentation_keeps_wide_shape_together(self) -> None:
        """Wide single characters should not always be split like digits."""

        image = Image.new("L", (96, 72), 255)
        draw = ImageDraw.Draw(image)
        draw.arc((18, 8, 70, 42), start=195, end=25, fill=0, width=6)
        draw.line((68, 26, 24, 62), fill=0, width=6)
        draw.line((24, 62, 78, 62), fill=0, width=6)

        regions = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)

        self.assertEqual(len(regions), 1)

    def test_character_segmentation_merges_punctuation_dot(self) -> None:
        """Question-mark dots should merge with their parent mark."""

        image = Image.new("L", (80, 96), 255)
        draw = ImageDraw.Draw(image)
        draw.arc((20, 10, 60, 48), start=205, end=40, fill=0, width=5)
        draw.line((52, 32, 40, 56), fill=0, width=5)
        draw.ellipse((38, 76, 46, 84), fill=0)

        regions = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)

        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0].box[3], 80)

    def test_character_segmentation_merges_exclamation_dot(self) -> None:
        """Exclamation dots should merge with their vertical stroke."""

        image = Image.new("L", (100, 140), 255)
        draw = ImageDraw.Draw(image)
        draw.line((48, 16, 48, 86), fill=0, width=6)
        draw.ellipse((42, 106, 54, 118), fill=0)

        regions = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)

        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0].box[3] - regions[0].box[1], 90)

    def test_character_segmentation_merges_disconnected_letter_parts(self) -> None:
        """Disconnected pieces of a hand-drawn letter should stay together."""

        image = Image.new("L", (120, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.line((28, 18, 28, 100), fill=0, width=6)
        draw.arc((24, 18, 86, 60), start=270, end=90, fill=0, width=6)
        draw.arc((24, 58, 92, 100), start=270, end=90, fill=0, width=6)

        regions = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)

        self.assertEqual(len(regions), 1)

    def test_character_segmentation_merges_stacked_digit_parts(self) -> None:
        """Disconnected top and bottom strokes of a 5 should stay together."""

        image = Image.new("L", (120, 150), 255)
        draw = ImageDraw.Draw(image)
        draw.line((75, 25, 35, 25), fill=0, width=7)
        draw.line((35, 25, 35, 70), fill=0, width=7)
        draw.arc((25, 58, 88, 125), start=260, end=80, fill=0, width=7)

        regions = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)

        self.assertEqual(len(regions), 1)

    def test_messy_connected_27_segments_and_predicts(self) -> None:
        """The original connected 27 regression should still read correctly."""

        if not Path(WEIGHTS_PATH).exists():
            self.skipTest("trained weights are not available")

        image = Image.new("L", (1000, 700), 255)
        draw = ImageDraw.Draw(image)
        draw.arc((130, 120, 420, 520), start=250, end=95, fill=0, width=8)
        draw.line((180, 560, 320, 560), fill=0, width=8)
        draw.line((320, 560, 430, 460), fill=0, width=8)
        draw.line((430, 460, 430, 350), fill=0, width=8)
        draw.line((260, 430, 470, 360), fill=0, width=8)
        draw.line((470, 360, 730, 360), fill=0, width=8)
        draw.line((680, 130, 900, 120), fill=0, width=8)
        draw.line((900, 120, 880, 560), fill=0, width=8)

        regions = segment_digit_regions(image)
        predictions = predict_digits(load_model(), image)

        self.assertEqual(len(regions), 2)
        self.assertEqual("".join(str(item["digit"]) for item in predictions), "27")

    def test_broken_top_stroke_15_does_not_fragment(self) -> None:
        """The original 15 regression should remain two characters."""

        if not Path(WEIGHTS_PATH).exists():
            self.skipTest("trained weights are not available")

        screenshot_path = Path.home() / "Desktop" / "Screenshot 2026-07-04 at 13.21.27.png"
        if not screenshot_path.exists():
            self.skipTest("local 15 regression screenshot is not available")

        image = Image.open(screenshot_path)
        regions = segment_digit_regions(image)
        predictions = predict_digits(load_model(), image)

        self.assertEqual(len(regions), 2)
        self.assertEqual("".join(str(item["digit"]) for item in predictions), "15")

    def test_preprocessing_accepts_encoded_image_bytes(self) -> None:
        image = Image.new("RGB", (40, 40), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((14, 8, 25, 31), fill="black")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        decoded = Image.open(io.BytesIO(buffer.getvalue()))
        normalized = mnist_normalize_image(decoded)

        self.assertEqual(normalized.mode, "L")
        self.assertEqual(normalized.size, (28, 28))


if __name__ == "__main__":
    unittest.main()
