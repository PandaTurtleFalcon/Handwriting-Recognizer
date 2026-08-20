import unittest

from scripts.analyze_character_headroom import headroom_report


class CharacterHeadroomTests(unittest.TestCase):
    """Unit coverage for 93-class character headroom accounting."""

    def test_headroom_counts_visual_recoverable_errors_by_split(self) -> None:
        """The report should separate exact misses from known visual twins."""

        report = headroom_report(
            expected_labels=["A", "0", "-", "B"],
            predicted_labels=["A", "O", "_", "X"],
            families=[frozenset("0Oo"), frozenset("-_")],
        )

        self.assertEqual(report["total"], 4)
        self.assertEqual(report["target_accuracy"], 95.0)
        self.assertEqual(report["exact_accuracy"], 25.0)
        self.assertEqual(report["ambiguity_aware_accuracy"], 75.0)
        self.assertEqual(report["visual_oracle_accuracy"], 75.0)
        self.assertEqual(report["accuracy_gap_to_target"], 70.0)
        self.assertEqual(report["visual_oracle_gap_to_target"], 20.0)
        self.assertEqual(report["visual_recoverable_errors"], 2)
        self.assertEqual(report["remaining_non_family_errors"], 1)
        self.assertEqual(report["splits"]["digit"]["recoverable_errors"], 1)
        self.assertEqual(report["splits"]["punctuation"]["recoverable_errors"], 1)
        self.assertEqual(report["families"][0]["accuracy_gain"], 25.0)
        self.assertEqual(report["families"][0]["split_recoverable_errors"]["digit"], 1)
        self.assertEqual(report["cumulative_family_oracle"][0]["cumulative_accuracy"], 50.0)
        self.assertIsNone(report["families_to_reach_95"])

    def test_headroom_reports_first_family_set_to_reach_target(self) -> None:
        """The roadmap should identify the first cumulative family crossing the target."""

        report = headroom_report(
            expected_labels=["A"] * 19 + ["0"],
            predicted_labels=["A"] * 19 + ["O"],
            families=[frozenset("0Oo")],
            target_accuracy=90.0,
        )

        self.assertEqual(report["families_to_reach_target"]["families"], ["0Oo"])
        self.assertEqual(report["families_to_reach_95"]["families"], ["0Oo"])
        self.assertTrue(report["families_to_reach_target"]["reaches_target"])
        self.assertTrue(report["families_to_reach_95"]["reaches_95"])

    def test_headroom_rejects_mismatched_lengths(self) -> None:
        """Expected and predicted labels must be aligned lists."""

        with self.assertRaisesRegex(ValueError, "same length"):
            headroom_report(["A"], [])


if __name__ == "__main__":
    unittest.main()
