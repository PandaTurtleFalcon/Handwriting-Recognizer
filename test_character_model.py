import unittest

from character_model import (
    _digit_beats_ambiguous_letter,
    _looks_like_one,
    _looks_like_seven,
    _postprocess_colons,
    _postprocess_exclamations,
    _postprocess_lowercase_i,
    _punctuation_shape_label,
)
from mnist_model import DigitRegion, segment_digit_regions
from PIL import Image, ImageDraw


class CharacterPostprocessingTests(unittest.TestCase):
    """Regression tests for model-independent character cleanup rules."""

    def test_split_dot_above_stem_becomes_lowercase_i(self) -> None:
        """A detached dot above a skinny stem should read as lowercase i."""

        predictions = [
            {"label": "H", "confidence": 0.97, "x": 40, "y": 55, "width": 90, "height": 120, "row": 1},
            {"label": ":", "confidence": 0.91, "x": 184, "y": 40, "width": 12, "height": 12, "row": 1},
            {"label": "L", "confidence": 0.81, "x": 180, "y": 70, "width": 14, "height": 105, "row": 2},
        ]

        cleaned = _postprocess_lowercase_i(predictions)

        self.assertEqual("".join(str(item["label"]) for item in cleaned), "Hi")
        self.assertEqual(len(cleaned), 2)
        self.assertGreaterEqual(float(cleaned[1]["confidence"]), 0.9)

    def test_split_colon_dots_are_merged(self) -> None:
        """Two vertically aligned small dots should become one colon."""

        predictions = [
            {"label": "Q", "confidence": 0.80, "x": 50, "y": 30, "width": 14, "height": 14, "row": 1},
            {"label": "Q", "confidence": 0.80, "x": 51, "y": 76, "width": 14, "height": 14, "row": 2},
        ]

        cleaned = _postprocess_colons(predictions)

        self.assertEqual("".join(str(item["label"]) for item in cleaned), ":")
        self.assertEqual(len(cleaned), 1)

    def test_split_dot_below_stem_becomes_exclamation(self) -> None:
        """A detached dot below a skinny stem should read as exclamation."""

        predictions = [
            {"label": "1", "confidence": 0.98, "x": 50, "y": 15, "width": 12, "height": 80, "row": 1},
            {"label": "0", "confidence": 0.81, "x": 48, "y": 112, "width": 15, "height": 15, "row": 2},
        ]

        cleaned = _postprocess_exclamations(predictions)

        self.assertEqual("".join(str(item["label"]) for item in cleaned), "!")
        self.assertEqual(len(cleaned), 1)

    def test_dot_below_stem_stays_exclamation_mark(self) -> None:
        """A dot below a vertical stem should stay an exclamation mark."""

        image = Image.new("L", (70, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.line((34, 14, 34, 78), fill=0, width=6)
        draw.ellipse((28, 96, 40, 108), fill=0)
        region = DigitRegion(image=image, box=(0, 0, 70, 120), row=1)

        self.assertEqual(_punctuation_shape_label(region), "!")

    def test_dot_above_stem_is_lowercase_i_shape(self) -> None:
        """A merged dot-above-stem shape should be classified as i."""

        image = Image.new("L", (70, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.ellipse((28, 12, 40, 24), fill=0)
        draw.line((34, 44, 34, 108), fill=0, width=6)
        region = DigitRegion(image=image, box=(0, 0, 70, 120), row=1)

        self.assertEqual(_punctuation_shape_label(region), "i")

    def test_shape_rule_identifies_plain_one(self) -> None:
        """A plain vertical stroke should remain digit 1, not letter L."""

        image = Image.new("L", (80, 180), 255)
        draw = ImageDraw.Draw(image)
        draw.line((38, 12, 38, 166), fill=0, width=6)
        region = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)[0]

        self.assertTrue(_looks_like_one(region))
        self.assertFalse(_looks_like_seven(region))

    def test_shape_rule_identifies_wide_top_seven(self) -> None:
        """A tall stroke with a wide top bar should be digit 7."""

        image = Image.new("L", (90, 220), 255)
        draw = ImageDraw.Draw(image)
        draw.line((18, 18, 72, 18), fill=0, width=7)
        draw.line((72, 18, 46, 206), fill=0, width=7)
        region = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)[0]

        self.assertTrue(_looks_like_seven(region))
        self.assertFalse(_looks_like_one(region))

    def test_digit_rescue_handles_common_letter_confusions(self) -> None:
        """Confident digit votes should beat known letter lookalikes."""

        self.assertTrue(_digit_beats_ambiguous_letter("4", 0.98, "Y", 0.979))
        self.assertTrue(_digit_beats_ambiguous_letter("5", 0.95, "J", 0.99))
        self.assertTrue(_digit_beats_ambiguous_letter("2", 0.99, "Z", 0.94))
        self.assertFalse(_digit_beats_ambiguous_letter("4", 0.91, "Y", 0.979))


if __name__ == "__main__":
    unittest.main()
