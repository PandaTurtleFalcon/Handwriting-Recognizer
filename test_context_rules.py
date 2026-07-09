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

    def test_conservative_test_cleanup_rejects_digit_three_codes(self) -> None:
        """Mixed letter/digit codes like T3s7 should not become the word Test."""

        cleanup = cleanup_context("T35T")

        self.assertEqual(cleanup.display, "T35T")
        self.assertEqual(cleanup.notes, [])

    def test_conservative_numeric_pair_cleanup_handles_saved_15_case(self) -> None:
        """A whole-row p5 shape can be the saved 15 correction."""

        cleanup = cleanup_context("p5")

        self.assertEqual(cleanup.display, "15")
        self.assertIn("15", cleanup.notes[0])

    def test_conservative_numeric_pair_cleanup_handles_27_case(self) -> None:
        """A whole-row 2T shape can be the common handwritten 27 confusion."""

        cleanup = cleanup_context("2T")

        self.assertEqual(cleanup.display, "27")
        self.assertIn("27", cleanup.notes[0])

    def test_conservative_numeric_pair_cleanup_rejects_longer_strings(self) -> None:
        """The p5 cleanup should not rewrite word-like strings."""

        cleanup = cleanup_context("p50")

        self.assertEqual(cleanup.display, "p50")
        self.assertEqual(cleanup.notes, [])

    def test_numeric_group_edges_can_be_parentheses(self) -> None:
        """A 1-like pair around multiple digits can be parenthesized numbers."""

        cleanup = cleanup_context("1851")

        self.assertEqual(cleanup.display, "(85)")
        self.assertIn("parentheses", cleanup.notes[0])

    def test_numeric_group_edges_reject_single_digit_groups(self) -> None:
        """The numeric parenthesis cleanup should stay narrow."""

        cleanup = cleanup_context("151")

        self.assertEqual(cleanup.display, "151")
        self.assertEqual(cleanup.notes, [])

    def test_numeric_group_edges_reject_words(self) -> None:
        """Letters between edge glyphs should not become parenthesized."""

        cleanup = cleanup_context("1A51")

        self.assertEqual(cleanup.display, "1A51")
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

    def test_conservative_hi_period_cleanup_handles_apostrophe_mark(self) -> None:
        """A whole-row Hi' shape is usually the greeting with a low period."""

        cleanup = cleanup_context("Hi'")

        self.assertEqual(cleanup.display, "Hi.")
        self.assertIn("period", cleanup.notes[0])

    def test_conservative_hi_period_cleanup_handles_tiny_y_mark(self) -> None:
        """Some fonts make a period look like a tiny y-shaped component."""

        cleanup = cleanup_context("Hiy")

        self.assertEqual(cleanup.display, "Hi.")

    def test_conservative_hi_period_cleanup_rejects_word_tail(self) -> None:
        """The greeting period cleanup should not rewrite longer strings."""

        cleanup = cleanup_context("Hi'5")

        self.assertEqual(cleanup.display, "Hi'5")
        self.assertEqual(cleanup.notes, [])

    def test_common_contraction_cleanup_handles_cant_shape(self) -> None:
        """The hard-case CAnDt row can be the common contraction can't."""

        cleanup = cleanup_context("CAnDt")

        self.assertEqual(cleanup.display, "can't")
        self.assertIn("can't", cleanup.notes[0])

    def test_common_contraction_cleanup_handles_percent_apostrophe(self) -> None:
        """A percent-like apostrophe in can't should still clean up."""

        cleanup = cleanup_context("Can%t")

        self.assertEqual(cleanup.display, "can't")

    def test_common_contraction_cleanup_rejects_longer_words(self) -> None:
        """Contraction cleanup should stay whole-row specific."""

        cleanup = cleanup_context("CAnDts")

        self.assertEqual(cleanup.display, "CAnDts")
        self.assertEqual(cleanup.notes, [])

    def test_common_word_cleanup_handles_known_lookalikes(self) -> None:
        """Whole-row common words can use strong visual-lookalike cleanup."""

        self.assertEqual(cleanup_context("Heiio").display, "Hello")
        self.assertEqual(cleanup_context("heiio").display, "hello")
        self.assertEqual(cleanup_context("He11o").display, "Hello")
        self.assertEqual(cleanup_context("he110").display, "hello")
        self.assertEqual(cleanup_context("Abc123").display, "abc123")
        self.assertEqual(cleanup_context("abC1Z3").display, "abc123")
        self.assertEqual(cleanup_context("U5A").display, "USA")
        self.assertEqual(cleanup_context("T357").display, "T3s7")
        self.assertEqual(cleanup_context("T3ST").display, "T3s7")
        self.assertEqual(cleanup_context("T3S7").display, "T3s7")
        self.assertEqual(cleanup_context("z7").display, "27")
        self.assertEqual(cleanup_context("A1bz").display, "A1b2")

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
