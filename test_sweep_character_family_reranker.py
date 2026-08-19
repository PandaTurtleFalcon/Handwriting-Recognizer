import unittest
from pathlib import Path

from scripts.sweep_character_family_reranker import (
    accepted_family_counts,
    best_sweep_row,
    parse_float_values,
    parse_int_values,
    parse_source_group_sets,
    rejection_reason_counts,
    run_sweep,
    top_family_rows,
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

    def test_family_diagnostics_summarize_whole_sweep(self) -> None:
        rows = [
            {
                "families": [
                    {
                        "family": "0Oo",
                        "accepted": False,
                        "delta": 0.2,
                        "rejection_reason": "selection_validation_delta_below_floor",
                    },
                    {"family": "5Ss", "accepted": True, "delta": 0.1},
                ]
            },
            {
                "families": [
                    {
                        "family": "0Oo",
                        "accepted": False,
                        "delta": 0.0,
                        "rejection_reason": "selection_validation_delta_below_floor",
                    }
                ]
            },
        ]

        self.assertEqual(top_family_rows(rows, limit=1)[0]["family"], "0Oo")
        self.assertEqual(rejection_reason_counts(rows), {"selection_validation_delta_below_floor": 2})
        self.assertEqual(accepted_family_counts(rows), {"5Ss": 1})

    def test_run_sweep_can_limit_planned_grid(self) -> None:
        calls = []
        prepared = object()

        def fake_run_probe(**kwargs):
            calls.append(kwargs)
            return {"promotable": False, "validation_delta": 0.0, "base": {}, "reranked": {}}

        original = run_sweep.__globals__["run_probe"]
        original_prepare = run_sweep.__globals__["prepare_probe_data"]
        run_sweep.__globals__["run_probe"] = fake_run_probe
        run_sweep.__globals__["prepare_probe_data"] = lambda **_kwargs: prepared
        try:
            report = run_sweep(
                batch_size=4,
                epochs=[1, 2],
                learning_rates=[0.1],
                hidden_units=[0, 4],
                families=("1Ili|!/",),
                source_group_sets=[("letter",), None],
                probe_confidences=[0.0],
                probe_margins=[0.0, 0.1],
                calibration_ratio=0.2,
                confirmation_ratio=0.5,
                min_family_delta=0.0,
                seed=7,
                train_only_extra_roots=(Path("cvl.pt"),),
                include_pixel_features=True,
                include_embedding_features=True,
                max_probe_train_samples=128,
                mini_batch_size=32,
                max_runs=3,
            )
        finally:
            run_sweep.__globals__["run_probe"] = original
            run_sweep.__globals__["prepare_probe_data"] = original_prepare

        self.assertEqual(len(calls), 3)
        self.assertEqual(report["planned_runs"], 16)
        self.assertEqual(report["completed_runs"], 3)
        self.assertTrue(report["truncated"])
        self.assertEqual(calls[0]["train_only_extra_roots"], (Path("cvl.pt"),))
        self.assertTrue(calls[0]["include_pixel_features"])
        self.assertTrue(calls[0]["include_embedding_features"])
        self.assertEqual(calls[0]["max_probe_train_samples"], 128)
        self.assertEqual(calls[0]["mini_batch_size"], 32)
        self.assertEqual(calls[0]["probe_confidence"], 0.0)
        self.assertEqual(calls[0]["probe_margin"], 0.0)
        self.assertIs(calls[0]["probe_data"], prepared)
        self.assertEqual(report["train_only_extra_roots"], ["cvl.pt"])
        self.assertTrue(report["include_pixel_features"])
        self.assertTrue(report["include_embedding_features"])
        self.assertEqual(report["max_probe_train_samples"], 128)
        self.assertEqual(report["mini_batch_size"], 32)
        self.assertEqual(report["rejection_reason_counts"], {})
        self.assertEqual(report["accepted_family_counts"], {})
        self.assertEqual(report["top_family_rows"], [])


if __name__ == "__main__":
    unittest.main()
