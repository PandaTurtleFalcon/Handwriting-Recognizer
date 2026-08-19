import unittest

from scripts.sweep_mixedcase_case_resolver import (
    best_sweep_row,
    compact_probe_report,
    parse_choice_values,
    parse_float_values,
    parse_int_values,
)


class SweepMixedcaseCaseResolverTests(unittest.TestCase):
    """Regression tests for confirmed mixed-case case-resolver sweeps."""

    def test_parse_values_require_non_empty_lists(self) -> None:
        """Sweep parsers should reject empty value lists."""

        self.assertEqual(parse_int_values("1, 2"), [1, 2])
        self.assertEqual(parse_float_values("0.1, .2"), [0.1, 0.2])
        self.assertEqual(parse_choice_values("exact, balanced", ("exact", "balanced"), "objective"), ["exact", "balanced"])
        with self.assertRaisesRegex(ValueError, "At least one"):
            parse_int_values(" , ")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_choice_values("odd", ("exact",), "objective")

    def test_compact_probe_report_tracks_final_candidate_safety(self) -> None:
        """Compact reports should preserve final selected candidate safety."""

        report = {
            "promotable": False,
            "test_delta": 0.0,
            "final_selected_candidate": {"safe": False, "test_delta": -0.5},
        }

        compact = compact_probe_report(report, {"seed": 1})

        self.assertFalse(compact["promotable"])
        self.assertFalse(compact["final_selected_safe"])
        self.assertEqual(compact["final_selected_delta"], -0.5)

    def test_best_sweep_row_prefers_promotable_then_final_safe(self) -> None:
        """Ranking should prefer deployable rows before exploratory deltas."""

        rows = [
            {"promotable": False, "test_delta": 0.0, "final_selected_safe": False, "final_selected_delta": 1.0},
            {"promotable": False, "test_delta": 0.0, "final_selected_safe": True, "final_selected_delta": 0.1},
            {"promotable": True, "test_delta": 0.01, "final_selected_safe": True, "final_selected_delta": 0.01},
        ]

        self.assertIs(best_sweep_row(rows), rows[2])


if __name__ == "__main__":
    unittest.main()
