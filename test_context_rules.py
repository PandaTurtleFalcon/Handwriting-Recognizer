import unittest

from context_rules import cleanup_context


class ContextRulesTests(unittest.TestCase):
    """Focused tests for conservative text cleanup rules."""

    def test_balances_trailing_parenthesis_when_opening_exists(self) -> None:
        """A likely edge glyph can close an already-open parenthesized group."""

        cleanup = cleanup_context("T3(e57")

        self.assertEqual(cleanup.display, "T3(e5)")
        self.assertIn("Balanced", cleanup.notes[0])

    def test_balances_leading_parenthesis_when_closing_exists(self) -> None:
        """A likely leading L can become an opener only with a closing pair."""

        cleanup = cleanup_context("Lab)")

        self.assertEqual(cleanup.display, "(ab)")

    def test_does_not_invent_parentheses_without_pair_context(self) -> None:
        """Unmatched candidate glyphs should remain visible when context is weak."""

        cleanup = cleanup_context("T3L87")

        self.assertEqual(cleanup.display, "T3L87")
        self.assertEqual(cleanup.notes, [])

    def test_conservative_test_cleanup_uses_whole_word_shape(self) -> None:
        """A four-character Test-shaped row can use common glyph lookalikes."""

        cleanup = cleanup_context("7:5T")

        self.assertEqual(cleanup.display, "Test")
        self.assertIn("Test", cleanup.notes[0])

    def test_conservative_test_cleanup_rejects_longer_strings(self) -> None:
        """The Test cleanup should not rewrite arbitrary mixed strings."""

        cleanup = cleanup_context("7:5T9")

        self.assertEqual(cleanup.display, "7:5T9")
        self.assertEqual(cleanup.notes, [])

    def test_conservative_numeric_pair_cleanup_handles_saved_15_case(self) -> None:
        """A whole-row p5 shape can be the saved 15 correction."""

        cleanup = cleanup_context("p5")

        self.assertEqual(cleanup.display, "15")
        self.assertIn("15", cleanup.notes[0])

    def test_conservative_numeric_pair_cleanup_rejects_longer_strings(self) -> None:
        """The p5 cleanup should not rewrite word-like strings."""

        cleanup = cleanup_context("p50")

        self.assertEqual(cleanup.display, "p50")
        self.assertEqual(cleanup.notes, [])

    def test_conservative_hi_cleanup_allows_punctuation_tail(self) -> None:
        """HL! is a safe greeting correction because no word tail is guessed."""

        cleanup = cleanup_context("HL!")

        self.assertEqual(cleanup.display, "Hi!")
        self.assertIn("Hi", cleanup.notes[0])

    def test_conservative_hi_cleanup_rejects_word_tail(self) -> None:
        """HL5 should not become Hi5 because that changes real content."""

        cleanup = cleanup_context("HL5")

        self.assertEqual(cleanup.display, "HL5")
        self.assertEqual(cleanup.notes, [])

    def test_row_strings_stay_separated_in_display(self) -> None:
        """Multi-row uploads should not collapse into one ambiguous string."""

        cleanup = cleanup_context("HL!123", ["HL!", "123"])

        self.assertEqual(cleanup.display, "Hi!\n123")
        self.assertEqual(cleanup.rows, ["Hi!", "123"])
        self.assertTrue(any("2 detected rows" in note for note in cleanup.notes))

    def test_drops_isolated_colon_after_hi_row(self) -> None:
        """The saved Hi correction should drop its stray punctuation-only row."""

        cleanup = cleanup_context("H1:", ["H1", ":"])

        self.assertEqual(cleanup.display, "Hi")
        self.assertEqual(cleanup.rows, ["Hi"])
        self.assertTrue(any("punctuation row" in note for note in cleanup.notes))

    def test_keeps_other_punctuation_rows(self) -> None:
        """Only the exact Hi + colon stray row is dropped."""

        cleanup = cleanup_context("OK:", ["OK", ":"])

        self.assertEqual(cleanup.display, "OK\n:")
        self.assertEqual(cleanup.rows, ["OK", ":"])


if __name__ == "__main__":
    unittest.main()
