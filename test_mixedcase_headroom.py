import unittest

from scripts.analyze_mixedcase_headroom import headroom_report


class MixedcaseHeadroomTests(unittest.TestCase):
    """Unit coverage for mixed-case headroom accounting."""

    def test_headroom_counts_case_and_visual_recoverable_errors(self) -> None:
        """The report should separate exact accuracy from oracle-recoverable misses."""

        report = headroom_report(
            expected_labels=["A", "a", "0", "Z", "B"],
            predicted_labels=["A", "A", "O", "2", "X"],
            families=[frozenset("0Oo"), frozenset("2Zz")],
        )

        self.assertEqual(report["total"], 5)
        self.assertEqual(report["target_accuracy"], 95.0)
        self.assertEqual(report["exact_accuracy"], 20.0)
        self.assertEqual(report["case_oracle_accuracy"], 40.0)
        self.assertEqual(report["visual_oracle_accuracy"], 60.0)
        self.assertEqual(report["case_or_visual_oracle_accuracy"], 80.0)
        self.assertEqual(report["accuracy_gap_to_target"], 75.0)
        self.assertEqual(report["case_or_visual_oracle_gap_to_target"], 15.0)
        self.assertEqual(report["case_or_visual_recoverable_errors"], 3)
        self.assertEqual(report["remaining_non_family_errors"], 1)
        self.assertEqual(report["error_types"]["case_only"]["count"], 1)
        self.assertEqual(report["error_types"]["case_only"]["split_counts"]["lower"], 1)
        self.assertEqual(report["error_types"]["visual_family"]["count"], 2)
        self.assertEqual(report["error_types"]["other_identity"]["count"], 1)
        self.assertEqual(report["splits"]["upper"]["recoverable_errors"], 1)
        self.assertEqual(report["splits"]["digit"]["recoverable_errors"], 1)
        self.assertEqual(report["families"][0]["family"], "0Oo")
        self.assertEqual(report["families"][0]["accuracy_gain"], 20.0)
        self.assertEqual(report["families"][0]["split_recoverable_errors"]["digit"], 1)
        self.assertEqual(report["cumulative_family_oracle"][0]["cumulative_accuracy"], 40.0)

    def test_headroom_reports_minimum_family_set_to_reach_target(self) -> None:
        """The report should identify the first cumulative oracle crossing the target."""

        expected = ["A"] * 80 + ["0"] * 10 + ["Z"] * 5 + ["B"] * 5
        predicted = ["A"] * 80 + ["O"] * 10 + ["2"] * 5 + ["X"] * 5

        report = headroom_report(
            expected_labels=expected,
            predicted_labels=predicted,
            families=[frozenset("0Oo"), frozenset("2Zz")],
            target_accuracy=90.0,
        )

        self.assertEqual(report["exact_accuracy"], 80.0)
        self.assertEqual(report["families_to_reach_target"]["families"], ["0Oo"])
        self.assertEqual(report["families_to_reach_95"]["families"], ["0Oo", "2Zz"])
        self.assertEqual(report["families_to_reach_target"]["cumulative_accuracy"], 90.0)
        self.assertEqual(report["families_to_reach_95"]["cumulative_accuracy"], 95.0)

    def test_headroom_rejects_mismatched_lengths(self) -> None:
        """Expected and predicted labels must be aligned lists."""

        with self.assertRaisesRegex(ValueError, "same length"):
            headroom_report(["A"], [])


if __name__ == "__main__":
    unittest.main()
