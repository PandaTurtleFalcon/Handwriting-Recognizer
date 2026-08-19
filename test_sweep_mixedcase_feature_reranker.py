import unittest
from pathlib import Path

from scripts.sweep_mixedcase_feature_reranker import (
    best_sweep_row,
    parse_float_values,
    parse_int_values,
    parse_source_group_sets,
    run_sweep,
)


class SweepMixedcaseFeatureRerankerTests(unittest.TestCase):
    """Regression tests for the mixed-case family sweep wrapper."""

    def test_parse_int_values_requires_values(self) -> None:
        self.assertEqual(parse_int_values("1, 2,3"), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "At least one"):
            parse_int_values(" , ")

    def test_parse_float_values_requires_values(self) -> None:
        self.assertEqual(parse_float_values("0.1, 2"), [0.1, 2.0])
        with self.assertRaisesRegex(ValueError, "At least one"):
            parse_float_values(" , ")

    def test_parse_source_group_sets_accepts_semicolon_sets(self) -> None:
        self.assertEqual(parse_source_group_sets("digit,upper;lower"), [("digit", "upper"), ("lower",)])

    def test_best_sweep_row_prefers_promotable_then_delta(self) -> None:
        rows = [
            {"promotable": False, "test_delta": 5.0},
            {"promotable": True, "test_delta": 0.1},
            {"promotable": True, "test_delta": 0.2},
        ]

        self.assertIs(best_sweep_row(rows), rows[2])

    def test_run_sweep_can_limit_planned_grid(self) -> None:
        calls = []

        def fake_run_probe(**kwargs):
            calls.append(kwargs)
            return {"promotable": False, "test_delta": 0.0, "base": {}, "reranked": {}, "families": []}

        original = run_sweep.__globals__["run_probe"]
        run_sweep.__globals__["run_probe"] = fake_run_probe
        try:
            report = run_sweep(
                batch_size=4,
                epochs=[1, 2],
                learning_rates=[0.1],
                hidden_units=[0, 4],
                source_group_sets=[("digit",), ("upper",)],
                probe_confidences=[0.0],
                probe_margins=[0.0, 0.1],
                train_sample_limit=10,
                family_limit=None,
                families=("0Oo",),
                calibration_ratio=0.2,
                confirmation_ratio=0.5,
                min_family_delta=0.0,
                seed=7,
                extra_roots=[Path("cvl.pt")],
                extra_samples_per_class=3,
                include_digit_features=True,
                include_pixel_features=True,
                min_digit=95.0,
                min_upper=84.0,
                min_lower=73.0,
                min_case_or_visual=98.0,
                max_runs=3,
            )
        finally:
            run_sweep.__globals__["run_probe"] = original

        self.assertEqual(len(calls), 3)
        self.assertEqual(report["planned_runs"], 16)
        self.assertEqual(report["completed_runs"], 3)
        self.assertTrue(report["truncated"])
        self.assertEqual(calls[0]["family_names"], ("0Oo",))
        self.assertEqual(calls[0]["extra_roots"], [Path("cvl.pt")])
        self.assertEqual(calls[0]["min_digit"], 95.0)
        self.assertTrue(calls[0]["include_digit_features"])
        self.assertTrue(calls[0]["include_pixel_features"])
        self.assertEqual(report["extra_roots"], ["cvl.pt"])
        self.assertTrue(report["include_pixel_features"])


if __name__ == "__main__":
    unittest.main()
