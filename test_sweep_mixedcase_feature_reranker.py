import unittest
from pathlib import Path

from scripts.sweep_mixedcase_feature_reranker import (
    aggregate_family_rejection_reason_counts,
    best_sweep_row,
    family_rejection_reason_counts,
    parse_float_values,
    parse_int_values,
    parse_optional_float_values,
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

    def test_parse_optional_float_values_accepts_none(self) -> None:
        self.assertEqual(parse_optional_float_values("none, 0.25, null"), [None, 0.25, None])
        with self.assertRaisesRegex(ValueError, "At least one"):
            parse_optional_float_values(" , ")

    def test_parse_source_group_sets_accepts_semicolon_sets(self) -> None:
        self.assertEqual(parse_source_group_sets("digit,upper;lower"), [("digit", "upper"), ("lower",)])

    def test_best_sweep_row_prefers_promotable_then_balanced_delta(self) -> None:
        rows = [
            {"promotable": False, "test_delta": 5.0, "balanced_delta": 0.0},
            {"promotable": False, "test_delta": 0.1, "balanced_delta": 0.2},
            {"promotable": True, "test_delta": 0.1, "balanced_delta": -1.0},
            {"promotable": True, "test_delta": 0.2, "balanced_delta": 0.1},
        ]

        self.assertIs(best_sweep_row(rows), rows[3])

    def test_family_rejection_reason_counts_skips_accepted_families(self) -> None:
        report = {
            "families": [
                {"accepted": False, "rejection_reason": "selection_delta_below_floor"},
                {"accepted": True, "rejection_reason": "should_not_count"},
                {"accepted": False, "rejection_reason": "selection_delta_below_floor"},
                {"accepted": False},
            ]
        }

        self.assertEqual(
            family_rejection_reason_counts(report),
            {"selection_delta_below_floor": 2, "unknown": 1},
        )

    def test_aggregate_family_rejection_reason_counts_combines_rows(self) -> None:
        rows = [
            {"family_rejection_reasons": {"selection_delta_below_floor": 2}},
            {
                "family_rejection_reasons": {
                    "selection_delta_below_floor": 1,
                    "final_delta_below_floor": 3,
                }
            },
            {"family_rejection_reasons": []},
        ]

        self.assertEqual(
            aggregate_family_rejection_reason_counts(rows),
            {"selection_delta_below_floor": 3, "final_delta_below_floor": 3},
        )

    def test_run_sweep_can_limit_planned_grid(self) -> None:
        calls = []
        prepare_calls = []
        fake_data = object()

        def fake_prepare_feature_probe_data(**kwargs):
            prepare_calls.append(kwargs)
            return fake_data

        def fake_run_probe_from_data(**kwargs):
            calls.append(kwargs)
            return {
                "promotable": False,
                "test_delta": 0.0,
                "balanced_delta": -1.0,
                "balanced_score": 73.0,
                "base": {},
                "reranked": {},
                "families": [{"accepted": False, "rejection_reason": "selection_delta_below_floor"}],
            }

        original_prepare = run_sweep.__globals__["prepare_feature_probe_data"]
        original_probe = run_sweep.__globals__["run_probe_from_data"]
        run_sweep.__globals__["prepare_feature_probe_data"] = fake_prepare_feature_probe_data
        run_sweep.__globals__["run_probe_from_data"] = fake_run_probe_from_data
        try:
            report = run_sweep(
                batch_size=4,
                epochs=[1, 2],
                learning_rates=[0.1],
                hidden_units=[0, 4],
                source_group_sets=[("digit",), ("upper",)],
                probe_confidences=[0.0],
                probe_margins=[0.0, 0.1],
                base_confidence_maxes=[None, 0.35],
                base_margin_maxes=[None],
                digit_protect_confidences=[None, 0.95],
                upper_protect_confidences=[None],
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
                include_embedding_features=True,
                min_digit=95.0,
                min_upper=84.0,
                min_lower=73.0,
                min_case_or_visual=98.0,
                max_probe_train_samples=128,
                mini_batch_size=32,
                max_runs=3,
            )
        finally:
            run_sweep.__globals__["prepare_feature_probe_data"] = original_prepare
            run_sweep.__globals__["run_probe_from_data"] = original_probe

        self.assertEqual(len(prepare_calls), 1)
        self.assertEqual(len(calls), 3)
        self.assertEqual(report["planned_runs"], 64)
        self.assertEqual(report["completed_runs"], 3)
        self.assertTrue(report["truncated"])
        self.assertTrue(report["prepared_once"])
        self.assertIs(calls[0]["data"], fake_data)
        self.assertEqual(calls[0]["family_names"], ("0Oo",))
        self.assertEqual(prepare_calls[0]["extra_roots"], [Path("cvl.pt")])
        self.assertEqual(calls[0]["min_digit"], 95.0)
        self.assertTrue(prepare_calls[0]["include_digit_features"])
        self.assertTrue(calls[0]["include_pixel_features"])
        self.assertTrue(prepare_calls[0]["include_embedding_features"])
        self.assertEqual(calls[0]["max_probe_train_samples"], 128)
        self.assertEqual(calls[0]["mini_batch_size"], 32)
        self.assertIsNone(calls[0]["base_confidence_max"])
        self.assertIsNone(calls[0]["digit_protect_confidence"])
        self.assertEqual(report["extra_roots"], ["cvl.pt"])
        self.assertTrue(report["include_pixel_features"])
        self.assertTrue(report["include_embedding_features"])
        self.assertEqual(report["base_confidence_maxes"], [None, 0.35])
        self.assertEqual(report["digit_protect_confidences"], [None, 0.95])
        self.assertEqual(report["max_probe_train_samples"], 128)
        self.assertEqual(report["mini_batch_size"], 32)
        self.assertEqual(report["rows"][0]["family_rejection_reasons"], {"selection_delta_below_floor": 1})
        self.assertEqual(report["family_rejection_reasons"], {"selection_delta_below_floor": 3})


if __name__ == "__main__":
    unittest.main()
