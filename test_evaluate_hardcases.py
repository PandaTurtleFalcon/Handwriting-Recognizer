import unittest

from scripts.evaluate_hardcases import sequence_matches_with_ambiguity


class HardCaseEvaluationTests(unittest.TestCase):
    def test_sequence_matches_visual_ambiguity(self) -> None:
        """Hard-case evaluation should report exact and visual-twin success separately."""

        self.assertTrue(sequence_matches_with_ambiguity("S5o", "sSO"))
        self.assertTrue(sequence_matches_with_ambiguity("Il1", "1lI"))
        self.assertFalse(sequence_matches_with_ambiguity("Hi", "HL:"))
        self.assertFalse(sequence_matches_with_ambiguity("AB", "A"))


if __name__ == "__main__":
    unittest.main()
