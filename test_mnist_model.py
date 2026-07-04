import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mnist_model import WEIGHTS_PATH, load_model, mnist_normalize_image, predict_digits, segment_digit_regions, segment_digits


class MnistPreprocessingTests(unittest.TestCase):
    def test_normalized_digit_is_28_by_28_and_nonblank(self) -> None:
        image = Image.new("L", (80, 80), 255)
        draw = ImageDraw.Draw(image)
        draw.line((38, 12, 38, 66), fill=0, width=9)

        normalized = mnist_normalize_image(image)
        pixels = np.asarray(normalized)

        self.assertEqual(normalized.size, (28, 28))
        self.assertGreater(pixels.sum(), 0)
        self.assertLessEqual(pixels.max(), 255)

    def test_segment_digits_splits_separated_components_left_to_right(self) -> None:
        image = Image.new("L", (160, 72), 255)
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 12, 54, 58), outline=0, width=8)
        draw.line((112, 12, 112, 58), fill=0, width=8)

        digits = segment_digits(image)

        self.assertEqual(len(digits), 2)
        self.assertGreater(digits[0].size[0] * digits[0].size[1], 0)
        self.assertGreater(digits[1].size[0] * digits[1].size[1], 0)

    def test_segment_regions_keep_boxes_and_read_rows_top_to_bottom(self) -> None:
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
        image = Image.new("L", (80, 80), 255)

        regions = segment_digit_regions(image)

        self.assertEqual(regions, [])

    def test_messy_connected_27_segments_and_predicts(self) -> None:
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
