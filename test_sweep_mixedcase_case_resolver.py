import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.sweep_mixedcase_case_resolver import (
    best_sweep_row,
    compact_probe_report,
    parse_choice_values,
    parse_float_values,
    parse_int_values,
    run_sweep,
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

    def test_run_sweep_reuses_prepared_data_per_seed(self) -> None:
        """Repeated hyperparameter rows should share cached model outputs."""

        prepared_by_seed = {101: object(), 202: object()}

        def fake_prepare(**kwargs):
            return prepared_by_seed[kwargs["seed"]]

        def fake_probe(data, **kwargs):
            return {
                "promotable": data is prepared_by_seed[202],
                "test_delta": 0.1 if data is prepared_by_seed[202] else 0.0,
                "final_selected_candidate": {"safe": data is prepared_by_seed[202], "test_delta": 0.1},
            }

        with patch("scripts.sweep_mixedcase_case_resolver.prepare_case_resolver_data", side_effect=fake_prepare) as prepare:
            with patch("scripts.sweep_mixedcase_case_resolver.run_probe_from_data", side_effect=fake_probe) as probe:
                report = run_sweep(
                    batch_size=16,
                    train_sample_limit=40,
                    epochs=[1, 2],
                    learning_rates=[0.01],
                    hidden_units=[0],
                    objectives=["exact"],
                    class_weightings=["none"],
                    confidence_thresholds=[0.0],
                    margin_thresholds=[0.0],
                    calibration_ratio=0.25,
                    confirmation_ratio=0.5,
                    seeds=[101, 202],
                    extra_roots=[Path("tmp/example.pt")],
                    extra_samples_per_class=3,
                    include_embedding_features=True,
                )

        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(probe.call_count, 4)
        self.assertEqual(report["cached_seed_count"], 2)
        self.assertEqual(report["completed_runs"], 4)
        self.assertEqual(report["promotable_count"], 2)


if __name__ == "__main__":
    unittest.main()
