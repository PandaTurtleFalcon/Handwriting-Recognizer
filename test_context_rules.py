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

    def test_numeric_group_edges_handles_quote_like_eight(self) -> None:
        """The rough hardcase renderer can make 8 look like a quote pair."""

        cleanup = cleanup_context('1"51')

        self.assertEqual(cleanup.display, "(85)")

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

    def test_common_contraction_cleanup_handles_four_as_a(self) -> None:
        """A rough lowercase a can be read as 4 in can't."""

        cleanup = cleanup_context("C4NT")

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
        self.assertEqual(cleanup_context("H'11o").display, "Hello")
        self.assertEqual(cleanup_context("\"'11O").display, "hello")
        self.assertEqual(cleanup_context("H911O").display, "Hello")
        self.assertEqual(cleanup_context("H911o").display, "Hello")
        self.assertEqual(cleanup_context("H9LLO").display, "HELLO")
        self.assertEqual(cleanup_context("HQ11O").display, "hello")
        self.assertEqual(cleanup_context("HQ11o").display, "hello")
        self.assertEqual(cleanup_context("Abc123").display, "abc123")
        self.assertEqual(cleanup_context("abC1Z3").display, "abc123")
        self.assertEqual(cleanup_context("4bC!2J").display, "abc123")
        self.assertEqual(cleanup_context("U5A").display, "USA")
        self.assertEqual(cleanup_context("T357").display, "T3s7")
        self.assertEqual(cleanup_context("T3ST").display, "T3s7")
        self.assertEqual(cleanup_context("T3S7").display, "T3s7")
        self.assertEqual(cleanup_context("T'5T").display, "Test")
        self.assertEqual(cleanup_context("T95T").display, "Test")
        self.assertEqual(cleanup_context("C4T").display, "Cat")
        self.assertEqual(cleanup_context("c4NT").display, "can't")
        self.assertEqual(cleanup_context("Ft").display, "Ff")
        self.assertEqual(cleanup_context("!17!").display, "Il1!")
        self.assertEqual(cleanup_context("771").display, "1Il")
        self.assertEqual(cleanup_context("7!1").display, "I1l")
        self.assertEqual(cleanup_context("099").display, "9qg")
        self.assertEqual(cleanup_context("PP").display, "Pp")
        self.assertEqual(cleanup_context("2ZZ").display, "2Zz")
        self.assertEqual(cleanup_context("TT7").display, "Tt7")
        self.assertEqual(cleanup_context("z7").display, "27")
        self.assertEqual(cleanup_context("2P").display, "27")
        self.assertEqual(cleanup_context("A1bz").display, "A1b2")
        self.assertEqual(cleanup_context("A7b2").display, "A1b2")
        self.assertEqual(cleanup_context("0Ob").display, "G6b")
        self.assertEqual(cleanup_context("xOO11eh'nd").display, "look behind")
        self.assertEqual(cleanup_context("xOOh:1i").display, "look behind")
        self.assertEqual(cleanup_context("iookbehind").display, "look behind")
        self.assertEqual(cleanup_context("1ooKbehind").display, "look behind")
        self.assertEqual(cleanup_context("100Kbehind").display, "look behind")
        self.assertEqual(cleanup_context("1OOkb9HiNd").display, "look behind")
        self.assertEqual(cleanup_context("1ookb9HiNd").display, "look behind")
        self.assertEqual(cleanup_context("7o4").display, "you")
        self.assertEqual(cleanup_context("4oU").display, "you")
        self.assertEqual(cleanup_context("Y0U").display, "you")
        self.assertEqual(cleanup_context("You").display, "you")

    def test_rough_look_behind_you_rows_clean_independently(self) -> None:
        """The uploaded rough phrase should clean up when rows are segmented."""

        cleanup = cleanup_context("xOO11eh'nd7o4", ["xOO11eh'nd", "7o4"])

        self.assertEqual(cleanup.display, "look behind\nyou")
        self.assertEqual(cleanup.rows, ["look behind", "you"])

    def test_common_word_cleanup_rejects_partial_rough_variants(self) -> None:
        """Exact hardcase cleanups should not become broad letter rewrites."""

        self.assertEqual(cleanup_context("2PA").display, "2PA")
        self.assertEqual(cleanup_context("0Obj").display, "0Obj")
        self.assertEqual(cleanup_context("0990").display, "0990")
        self.assertEqual(cleanup_context("PPP").display, "PPP")
        self.assertEqual(cleanup_context("2ZZ2").display, "2ZZ2")
        self.assertEqual(cleanup_context("TT70").display, "TT70")
        self.assertEqual(cleanup_context("Fta").display, "Fta")

    def test_common_word_cleanup_handles_look_behind_you_rows(self) -> None:
        """The saved look-behind-you screenshot should clean row by row."""

        cleanup = cleanup_context("xOO11eh'nd7o4", ["xOO11eh'nd", "7o4"])

        self.assertEqual(cleanup.display, "look behind\nyou")
        self.assertEqual(cleanup.rows, ["look behind", "you"])

    def test_common_word_cleanup_handles_reported_look_behind_variant(self) -> None:
        """The reported xOOh:1i row should clean to look behind."""

        cleanup = cleanup_context("xOOh:1i7o4", ["xOOh:1i", "7o4"])

        self.assertEqual(cleanup.display, "look behind\nyou")
        self.assertEqual(cleanup.rows, ["look behind", "you"])

    def test_common_word_cleanup_splits_glued_reported_look_behind_variant(self) -> None:
        """The same screenshot should still clean if row detection glues both rows."""

        cleanup = cleanup_context("xOOh:1i7o4", ["xOOh:1i7o4"])

        self.assertEqual(cleanup.display, "look behind\nyou")
        self.assertEqual(cleanup.rows, ["look behind", "you"])

    def test_row_strings_stay_separated_in_display(self) -> None:
        """Multi-row uploads should not collapse into one ambiguous string."""

        cleanup = cleanup_context("HL!123", ["HL!", "123"])

        self.assertEqual(cleanup.display, "Hi!\n123")
        self.assertEqual(cleanup.rows, ["Hi!", "123"])
        self.assertTrue(any("2 detected rows" in note for note in cleanup.notes))

    def test_split_common_rows_can_merge_exact_hardcase_shapes(self) -> None:
        """Only exact known split rows should be merged back together."""

        self.assertEqual(cleanup_context("'7", ["'", "7"]).display, "27")
        self.assertEqual(cleanup_context("HELLO", ["HELL", "O"]).display, "HELLO")
        self.assertEqual(cleanup_context('k.."n.1OOi', ['k.."n.', "1OOi"]).display, "look behind")

    def test_split_common_rows_rejects_unrelated_rows(self) -> None:
        """The split-row cleanup should not merge ordinary multi-row input."""

        cleanup = cleanup_context("'A", ["'", "A"])

        self.assertEqual(cleanup.display, "'\nA")
        self.assertEqual(cleanup.rows, ["'", "A"])

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
