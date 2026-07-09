import unittest
from unittest.mock import patch

from PIL import ImageFont

from scripts.evaluate_hardcases import evaluate_cases, sequence_matches_with_ambiguity


class HardCaseEvaluationTests(unittest.TestCase):
    def test_sequence_matches_visual_ambiguity(self) -> None:
        """Hard-case evaluation should report exact and visual-twin success separately."""

        self.assertTrue(sequence_matches_with_ambiguity("S5o", "sSO"))
        self.assertTrue(sequence_matches_with_ambiguity("Il1", "1lI"))
        self.assertFalse(sequence_matches_with_ambiguity("Hi", "HL:"))
        self.assertFalse(sequence_matches_with_ambiguity("AB", "A"))

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

    def test_live_all_font_hardcases_stay_above_target(self) -> None:
        """The shipped website recognizer should keep hard cases above 95%."""

        report = evaluate_cases(all_fonts=True)

        self.assertGreaterEqual(report["exact_accuracy"], 95.0)
        self.assertGreaterEqual(report["ambiguity_aware_accuracy"], 95.0)


if __name__ == "__main__":
    unittest.main()
