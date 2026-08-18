import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image, ImageFont

from scripts.evaluate_hardcases import display_matches, evaluate_cases, render_script_case, sequence_matches_with_ambiguity


class HardCaseEvaluationTests(unittest.TestCase):
    def test_sequence_matches_visual_ambiguity(self) -> None:
        """Hard-case evaluation should report exact and visual-twin success separately."""

        self.assertTrue(sequence_matches_with_ambiguity("S5o", "sSO"))
        self.assertTrue(sequence_matches_with_ambiguity("Il1", "1lI"))
        self.assertFalse(sequence_matches_with_ambiguity("Hi", "HL:"))
        self.assertFalse(sequence_matches_with_ambiguity("AB", "A"))

    def test_display_matches_treats_row_breaks_as_phrase_spacing(self) -> None:
        """Whole-phrase hardcases should allow the app's visual row separator."""

        self.assertTrue(display_matches("look behind you", "look behind\nyou"))
        self.assertFalse(display_matches("look behind you", "look beh1nd\nyou"))

    def test_evaluate_cases_reports_per_font_metrics(self) -> None:
        """All-font mode should expose aggregate and per-font accuracy."""

        with patch("scripts.evaluate_hardcases.load_web_models", return_value=(object(), object())):
            with patch("scripts.evaluate_hardcases.iter_fonts", return_value=[("font-a", ImageFont.load_default())]):
                with patch("scripts.evaluate_hardcases.main.classify_files") as classifier:
                    classifier.return_value = [{"sequence": "Hi"}]

                    report = evaluate_cases(["Hi"], all_fonts=True)

        self.assertEqual(report["exact_accuracy"], 100.0)
        self.assertEqual(report["per_font"]["font-a"]["exact_accuracy"], 100.0)
        self.assertEqual(report["results"][0]["font"], "font-a")

    def test_evaluate_cases_can_include_script_cases(self) -> None:
        """Script mode should add deterministic rough handwriting cases."""

        with patch("scripts.evaluate_hardcases.load_web_models", return_value=(object(), object())):
            with patch("scripts.evaluate_hardcases.iter_fonts", return_value=[("font-a", ImageFont.load_default())]):
                with patch("scripts.evaluate_hardcases.main.classify_files") as classifier:
                    classifier.return_value = [{"sequence": "Oo0"}]

                    report = evaluate_cases(["Oo0"], all_fonts=False, script_cases=True)

        self.assertEqual(report["total"], 2)
        self.assertEqual(classifier.call_count, 2)
        self.assertEqual(report["results"][1]["font"], "script")

    def test_script_case_renderer_is_deterministic_png(self) -> None:
        """Generated script hardcases should be stable for regression testing."""

        first = render_script_case("Oo0", seed=7)
        second = render_script_case("Oo0", seed=7)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith(b"\x89PNG"))

    def test_script_case_renderer_draws_mixed_letters_at_handwriting_scale(self) -> None:
        """Mixed fallback-prone labels should produce full-size handwritten ink."""

        payload = render_script_case("abc123", seed=11)
        image = Image.open(BytesIO(payload)).convert("L")
        dark_pixels = [
            (x, y)
            for y in range(image.height)
            for x in range(image.width)
            if image.getpixel((x, y)) < 128
        ]
        xs = [x for x, _ in dark_pixels]
        ys = [y for _, y in dark_pixels]

        self.assertGreater(max(xs) - min(xs), 180)
        self.assertGreater(max(ys) - min(ys), 60)

    def test_script_case_renderer_draws_detectable_period(self) -> None:
        """Script punctuation should be large enough for segmentation."""

        payload = render_script_case("Hi.", seed=1004)
        image = Image.open(BytesIO(payload)).convert("L")
        dark_pixels = [
            (x, y)
            for y in range(image.height)
            for x in range(image.width)
            if image.getpixel((x, y)) < 128
        ]
        lower_right_pixels = [(x, y) for x, y in dark_pixels if x > 145 and y > 105]

        self.assertGreaterEqual(len(lower_right_pixels), 30)

    def test_script_case_renderer_separates_upper_and_lowercase_height(self) -> None:
        """Case-pair hardcases should not render both glyphs identically."""

        upper = Image.open(BytesIO(render_script_case("C", seed=12))).convert("L")
        lower = Image.open(BytesIO(render_script_case("c", seed=12))).convert("L")
        upper_ys = [
            y
            for y in range(upper.height)
            for x in range(upper.width)
            if upper.getpixel((x, y)) < 128
        ]
        lower_ys = [
            y
            for y in range(lower.height)
            for x in range(lower.width)
            if lower.getpixel((x, y)) < 128
        ]

        self.assertGreater((max(upper_ys) - min(upper_ys)) - (max(lower_ys) - min(lower_ys)), 12)

    def test_live_all_font_hardcases_stay_above_target(self) -> None:
        """The shipped website recognizer should keep hard cases above 95%."""

        report = evaluate_cases(all_fonts=True)

        self.assertGreaterEqual(report["exact_accuracy"], 95.0)
        self.assertGreaterEqual(report["ambiguity_aware_accuracy"], 95.0)


if __name__ == "__main__":
    unittest.main()
