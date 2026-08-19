import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_character_specialist_probes import best_threshold_row, summarize_probes


class SummarizeCharacterSpecialistProbeTests(unittest.TestCase):
    """Regression tests for compact specialist probe summaries."""

    def test_best_threshold_row_prefers_gain_then_fixed_minus_broken(self) -> None:
        rows = [
            {"gain": 0.1, "replacement_report": {"fixed": 2, "broken": 2}},
            {"gain": 0.1, "replacement_report": {"fixed": 3, "broken": 1}},
            {"gain": 0.0, "replacement_report": {"fixed": 5, "broken": 0}},
        ]

        self.assertIs(best_threshold_row(rows), rows[1])

    def test_summarize_probes_counts_promotable_and_confirmed_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(
                json.dumps(
                    {
                        "promotable": True,
                        "thresholds": {"confidence": 0.9, "margin": 0.0},
                        "threshold_selection": {
                            "evaluated_thresholds": [
                                {"gain": 0.1, "replacement_report": {"fixed": 2, "broken": 0}},
                            ],
                        },
                        "confirmation": {
                            "gain": 0.1,
                            "confirmed": True,
                            "replacement_report": {"fixed": 1, "broken": 0},
                        },
                        "delta": {
                            "validation_accuracy": 0.2,
                            "letter_validation_accuracy": 0.3,
                            "digit_validation_accuracy": 0.0,
                            "punctuation_validation_accuracy": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "promotable": False,
                        "threshold_selection": {"evaluated_thresholds": []},
                        "confirmation": None,
                        "delta": {"validation_accuracy": 0.0},
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_probes([first, second])

        self.assertEqual(summary["probe_count"], 2)
        self.assertEqual(summary["promotable_count"], 1)
        self.assertEqual(summary["confirmed_count"], 1)
        self.assertEqual(summary["summaries"][0]["validation_delta"], 0.2)
        self.assertIsNone(summary["summaries"][1]["best_selection"])


if __name__ == "__main__":
    unittest.main()
