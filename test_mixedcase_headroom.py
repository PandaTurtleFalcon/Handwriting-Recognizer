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
        self.assertEqual(report["exact_accuracy"], 20.0)
        self.assertEqual(report["case_oracle_accuracy"], 40.0)
        self.assertEqual(report["visual_oracle_accuracy"], 60.0)
        self.assertEqual(report["case_or_visual_oracle_accuracy"], 80.0)
        self.assertEqual(report["case_or_visual_recoverable_errors"], 3)
        self.assertEqual(report["remaining_non_family_errors"], 1)
        self.assertEqual(report["splits"]["upper"]["recoverable_errors"], 1)
        self.assertEqual(report["splits"]["digit"]["recoverable_errors"], 1)

    def test_headroom_rejects_mismatched_lengths(self) -> None:
        """Expected and predicted labels must be aligned lists."""

        with self.assertRaisesRegex(ValueError, "same length"):
            headroom_report(["A"], [])


if __name__ == "__main__":
    unittest.main()
