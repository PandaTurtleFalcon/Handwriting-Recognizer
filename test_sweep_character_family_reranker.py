import unittest
from pathlib import Path

from scripts.sweep_character_family_reranker import (
    best_sweep_row,
    parse_float_values,
    parse_int_values,
    parse_source_group_sets,
    run_sweep,
)


class SweepCharacterFamilyRerankerTests(unittest.TestCase):
    """Regression tests for the character family sweep wrapper."""

    def test_parse_int_values_requires_values(self) -> None:
        self.assertEqual(parse_int_values("1, 2,3"), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "At least one"):
            parse_int_values(" , ")

    def test_parse_float_values_requires_values(self) -> None:
        self.assertEqual(parse_float_values("0.1, 2"), [0.1, 2.0])
        with self.assertRaisesRegex(ValueError, "At least one"):
            parse_float_values(" , ")

    def test_parse_source_group_sets_accepts_all_and_specific_groups(self) -> None:
        self.assertEqual(parse_source_group_sets("letter;punctuation;all"), [("letter",), ("punctuation",), None])

    def test_best_sweep_row_prefers_promotable_then_delta(self) -> None:
        rows = [
            {"promotable": False, "validation_delta": 5.0},
            {"promotable": True, "validation_delta": 0.1},
            {"promotable": True, "validation_delta": 0.2},
        ]

        self.assertIs(best_sweep_row(rows), rows[2])

    def test_run_sweep_can_limit_planned_grid(self) -> None:
        calls = []

        def fake_run_probe(**kwargs):
            calls.append(kwargs)
            return {"promotable": False, "validation_delta": 0.0, "base": {}, "reranked": {}}

        original = run_sweep.__globals__["run_probe"]
        run_sweep.__globals__["run_probe"] = fake_run_probe
        try:
            report = run_sweep(
                batch_size=4,
                epochs=[1, 2],
                learning_rates=[0.1],
                hidden_units=[0, 4],
                families=("1Ili|!/",),
                source_group_sets=[("letter",), None],
                calibration_ratio=0.2,
                confirmation_ratio=0.5,
                min_family_delta=0.0,
                seed=7,
                train_only_extra_roots=(Path("cvl.pt"),),
                max_runs=3,
            )
        finally:
            run_sweep.__globals__["run_probe"] = original

        self.assertEqual(len(calls), 3)
        self.assertEqual(report["planned_runs"], 8)
        self.assertEqual(report["completed_runs"], 3)
        self.assertTrue(report["truncated"])
        self.assertEqual(calls[0]["train_only_extra_roots"], (Path("cvl.pt"),))
        self.assertEqual(report["train_only_extra_roots"], ["cvl.pt"])


if __name__ == "__main__":
    unittest.main()
