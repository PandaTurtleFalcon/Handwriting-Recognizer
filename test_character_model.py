import unittest

from character_model import (
    _alnum_should_override,
    _digit_beats_ambiguous_letter,
    _letter_should_override,
    _looks_like_four,
    _looks_like_one,
    _looks_like_seven,
    _postprocess_colons,
    _postprocess_exclamations,
    _postprocess_lowercase_i,
    _punctuation_shape_label,
    _split_touching_character_regions,
    labels_match_with_ambiguity,
)
from mnist_model import DigitRegion, segment_digit_regions
from PIL import Image, ImageDraw


class CharacterPostprocessingTests(unittest.TestCase):
    """Regression tests for model-independent character cleanup rules."""

    def test_labels_match_with_visual_ambiguity_groups(self) -> None:
        """Ambiguity-aware scoring should accept known handwriting lookalikes."""

        self.assertTrue(labels_match_with_ambiguity("S", "s"))
        self.assertTrue(labels_match_with_ambiguity("0", "O"))
        self.assertTrue(labels_match_with_ambiguity("1", "|"))
        self.assertTrue(labels_match_with_ambiguity("_", "-"))
        self.assertTrue(labels_match_with_ambiguity(".", "'"))
        self.assertTrue(labels_match_with_ambiguity("q", "9"))
        self.assertFalse(labels_match_with_ambiguity("A", "B"))

    def test_split_dot_above_stem_becomes_lowercase_i(self) -> None:
        """A detached dot above a skinny stem should read as lowercase i."""

        predictions = [
            {"label": "H", "confidence": 0.97, "x": 40, "y": 55, "width": 90, "height": 120, "row": 1},
            {"label": ":", "confidence": 0.91, "x": 184, "y": 40, "width": 12, "height": 12, "row": 1},
            {"label": "L", "confidence": 0.81, "x": 180, "y": 70, "width": 14, "height": 105, "row": 1},
        ]

        cleaned = _postprocess_lowercase_i(predictions)

        self.assertEqual("".join(str(item["label"]) for item in cleaned), "Hi")
        self.assertEqual(len(cleaned), 2)
        self.assertGreaterEqual(float(cleaned[1]["confidence"]), 0.9)

    def test_split_colon_dots_are_merged(self) -> None:
        """Two vertically aligned small dots should become one colon."""

        predictions = [
            {"label": "Q", "confidence": 0.80, "x": 50, "y": 30, "width": 14, "height": 14, "row": 1},
            {"label": "Q", "confidence": 0.80, "x": 51, "y": 76, "width": 14, "height": 14, "row": 1},
        ]

        cleaned = _postprocess_colons(predictions)

        self.assertEqual("".join(str(item["label"]) for item in cleaned), ":")
        self.assertEqual(len(cleaned), 1)

    def test_split_dot_below_stem_becomes_exclamation(self) -> None:
        """A detached dot below a skinny stem should read as exclamation."""

        predictions = [
            {"label": "1", "confidence": 0.98, "x": 50, "y": 15, "width": 12, "height": 80, "row": 1},
            {"label": "0", "confidence": 0.81, "x": 48, "y": 112, "width": 15, "height": 15, "row": 1},
        ]

        cleaned = _postprocess_exclamations(predictions)

        self.assertEqual("".join(str(item["label"]) for item in cleaned), "!")
        self.assertEqual(len(cleaned), 1)

    def test_dot_postprocessing_does_not_merge_across_rows(self) -> None:
        """Detached dots should only merge with stems or dots on the same row."""

        colon = _postprocess_colons(
            [
                {"label": "Q", "confidence": 0.80, "x": 50, "y": 30, "width": 14, "height": 14, "row": 1},
                {"label": "Q", "confidence": 0.80, "x": 51, "y": 76, "width": 14, "height": 14, "row": 2},
            ]
        )
        lowercase_i = _postprocess_lowercase_i(
            [
                {"label": ":", "confidence": 0.91, "x": 184, "y": 40, "width": 12, "height": 12, "row": 1},
                {"label": "L", "confidence": 0.81, "x": 180, "y": 70, "width": 14, "height": 105, "row": 2},
            ]
        )
        exclamation = _postprocess_exclamations(
            [
                {"label": "1", "confidence": 0.98, "x": 50, "y": 15, "width": 12, "height": 80, "row": 1},
                {"label": "0", "confidence": 0.81, "x": 48, "y": 112, "width": 15, "height": 15, "row": 2},
            ]
        )

        self.assertEqual(len(colon), 2)
        self.assertEqual(len(lowercase_i), 2)
        self.assertEqual(len(exclamation), 2)

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

    def test_parenthesis_shapes_are_detected(self) -> None:
        """Single-stroke parentheses should beat letter/digit guesses."""

        left = Image.new("L", (90, 150), 255)
        right = Image.new("L", (90, 150), 255)
        left_draw = ImageDraw.Draw(left)
        right_draw = ImageDraw.Draw(right)
        left_draw.arc((22, 8, 86, 142), start=105, end=255, fill=0, width=6)
        right_draw.arc((4, 8, 68, 142), start=-75, end=75, fill=0, width=6)

        self.assertEqual(_punctuation_shape_label(DigitRegion(image=left, box=(0, 0, 90, 150), row=1)), "(")
        self.assertEqual(_punctuation_shape_label(DigitRegion(image=right, box=(0, 0, 90, 150), row=1)), ")")

    def test_blank_seam_splits_touching_character_region(self) -> None:
        """A wide region with an internal blank seam should become two regions."""

        image = Image.new("L", (180, 130), 255)
        draw = ImageDraw.Draw(image)
        draw.arc((12, 20, 78, 82), start=20, end=330, fill=0, width=6)
        draw.line((28, 52, 76, 52), fill=0, width=6)
        draw.line((120, 18, 166, 18), fill=0, width=6)
        draw.line((120, 18, 120, 62), fill=0, width=6)
        draw.line((120, 62, 160, 62), fill=0, width=6)
        draw.arc((114, 58, 168, 116), start=-90, end=95, fill=0, width=6)
        region = DigitRegion(image=image, box=(10, 20, 190, 150), row=1)

        split = _split_touching_character_regions([region])

        self.assertEqual(len(split), 2)
        self.assertLess(split[0].box[2], split[1].box[0])

    def test_thin_bridge_splits_touching_character_region(self) -> None:
        """A small connecting stroke should not force two glyphs into one region."""

        image = Image.new("L", (180, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.arc((12, 18, 78, 84), start=15, end=335, fill=0, width=7)
        draw.line((30, 54, 76, 54), fill=0, width=7)
        draw.line((76, 56, 108, 56), fill=0, width=4)
        draw.line((120, 18, 164, 18), fill=0, width=7)
        draw.line((164, 18, 120, 104), fill=0, width=7)
        region = DigitRegion(image=image, box=(20, 40, 200, 160), row=1)

        split = _split_touching_character_regions([region])

        self.assertEqual(len(split), 2)
        self.assertLess(split[0].box[2], split[1].box[0] + 8)

    def test_touching_narrow_punctuation_region_is_split(self) -> None:
        """A tall skinny mark attached to a glyph should survive as its own region."""

        image = Image.new("L", (160, 130), 255)
        draw = ImageDraw.Draw(image)
        draw.line((20, 12, 20, 96), fill=0, width=6)
        draw.ellipse((15, 108, 25, 118), fill=0)
        draw.line((23, 58, 54, 58), fill=0, width=4)
        draw.arc((62, 28, 130, 94), start=15, end=335, fill=0, width=7)
        draw.line((78, 60, 126, 60), fill=0, width=7)
        region = DigitRegion(image=image, box=(30, 30, 190, 160), row=1)

        split = _split_touching_character_regions([region])

        self.assertEqual(len(split), 2)
        self.assertLessEqual(split[0].box[2] - split[0].box[0], 40)

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

    def test_shape_rule_identifies_open_top_four(self) -> None:
        """A handwritten 4 with a crossbar should stay numeric."""

        image = Image.new("L", (100, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.line((70, 10, 70, 108), fill=0, width=7)
        draw.line((70, 10, 25, 62), fill=0, width=7)
        draw.line((25, 62, 82, 62), fill=0, width=7)
        region = segment_digit_regions(image, split_wide=False, min_component_pixels=4, merge_marks=True)[0]

        self.assertTrue(_looks_like_four(region))
        self.assertFalse(_looks_like_one(region))

    def test_digit_rescue_handles_common_letter_confusions(self) -> None:
        """Confident digit votes should beat known letter lookalikes."""

        self.assertTrue(_digit_beats_ambiguous_letter("4", 0.98, "Y", 0.979))
        self.assertTrue(_digit_beats_ambiguous_letter("5", 0.95, "J", 0.99))
        self.assertTrue(_digit_beats_ambiguous_letter("2", 0.99, "Z", 0.94))
        self.assertTrue(_digit_beats_ambiguous_letter("1", 0.996, "I", 0.97))
        self.assertTrue(_digit_beats_ambiguous_letter("1", 0.996, "l", 0.97))
        self.assertTrue(_digit_beats_ambiguous_letter("0", 0.996, "O", 0.96))
        self.assertTrue(_digit_beats_ambiguous_letter("8", 0.996, "B", 0.96))
        self.assertTrue(_digit_beats_ambiguous_letter("5", 0.996, "S", 0.96))
        self.assertFalse(_digit_beats_ambiguous_letter("4", 0.91, "Y", 0.979))
        self.assertFalse(_digit_beats_ambiguous_letter("1", 0.996, "I", 0.99))
        self.assertFalse(_digit_beats_ambiguous_letter("0", 0.990, "O", 0.96))

    def test_letter_model_needs_margin_to_replace_digits(self) -> None:
        """A weak letter vote should not steal a stronger digit prediction."""

        self.assertFalse(_letter_should_override("5", 0.95, 0.80, False))
        self.assertFalse(_letter_should_override("4", 0.98, 0.90, False))
        self.assertTrue(_letter_should_override("5", 0.80, 0.93, False))
        self.assertFalse(_letter_should_override("5", 0.80, 0.99, True))

    def test_alnum_model_needs_margin_to_flip_case(self) -> None:
        """Mixed-case predictions should not erase lowercase on tiny margins."""

        self.assertFalse(_alnum_should_override("s", 0.72, "S", 0.82, 0.0))
        self.assertTrue(_alnum_should_override("s", 0.72, "S", 0.91, 0.0))
        self.assertFalse(_alnum_should_override("S", 0.72, "s", 0.78, 0.0))
        self.assertTrue(_alnum_should_override("S", 0.72, "s", 0.84, 0.0))


if __name__ == "__main__":
    unittest.main()
