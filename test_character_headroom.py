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
        self.assertEqual(report["exact_accuracy"], 25.0)
        self.assertEqual(report["ambiguity_aware_accuracy"], 75.0)
        self.assertEqual(report["visual_recoverable_errors"], 2)
        self.assertEqual(report["remaining_non_family_errors"], 1)
        self.assertEqual(report["splits"]["digit"]["recoverable_errors"], 1)
        self.assertEqual(report["splits"]["punctuation"]["recoverable_errors"], 1)

    def test_headroom_rejects_mismatched_lengths(self) -> None:
        """Expected and predicted labels must be aligned lists."""

        with self.assertRaisesRegex(ValueError, "same length"):
            headroom_report(["A"], [])


if __name__ == "__main__":
    unittest.main()
