import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_mixedcase_probes import summarize_probes


class SummarizeMixedcaseProbeTests(unittest.TestCase):
    """Regression tests for compact mixed-case probe summaries."""

    def test_summarizes_sweep_residual_and_ensemble_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep = root / "sweep.json"
            feature = root / "feature.json"
            residual = root / "residual.json"
            ensemble = root / "ensemble.json"
            sweep.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "promotable": False,
                                "test_delta": -0.1,
                                "families": [
                                    {
                                        "family": "flat",
                                        "accepted": False,
                                        "delta": -0.1,
                                        "rejection_reason": "selection_delta_below_floor",
                                    }
                                ],
                            },
                            {
                                "promotable": True,
                                "test_delta": 0.2,
                                "parameters": {"epochs": 80},
                                "base": {"test_accuracy": 87.0},
                                "reranked": {"test_accuracy": 87.2},
                                "families": [
                                    {
                                        "family": "near",
                                        "accepted": False,
                                        "delta": 0.4,
                                        "rejection_reason": "final_digit_test_accuracy_regressed",
                                    },
                                    {"family": "kept", "accepted": True, "delta": 0.2},
                                ],
                            },
                        ],
                        "best": {
                            "promotable": True,
                            "test_delta": 0.2,
                            "parameters": {"epochs": 80},
                            "base": {"test_accuracy": 87.0},
                            "reranked": {"test_accuracy": 87.2},
                            "families": [
                                {
                                    "family": "flat",
                                    "accepted": False,
                                    "delta": 0.0,
                                    "rejection_reason": "final_delta_below_floor",
                                },
                                {
                                    "family": "near",
                                    "accepted": False,
                                    "delta": 0.3,
                                    "rejection_reason": "final_digit_test_accuracy_regressed",
                                },
                            ],
                        },
                        "promotable_count": 1,
                        "completed_runs": 2,
                        "planned_runs": 4,
                        "families": ["0Oo"],
                    }
                ),
                encoding="utf-8",
            )
            feature.write_text(
                json.dumps(
                    {
                        "promotable": False,
                        "test_delta": 0.05,
                        "balanced_delta": -0.1,
                        "balanced_score": 73.0,
                        "base": {"test_accuracy": 87.0},
                        "reranked": {"test_accuracy": 87.05},
                        "family_names": ["0Oo"],
                        "source_groups": ["digit", "upper"],
                        "probe_thresholds": {"confidence": 0.0},
                        "minimum_gates": {"digit_test_accuracy": 95.0},
                        "families": [
                            {
                                "family": "0Oo",
                                "accepted": False,
                                "delta": 0.05,
                                "rejection_reason": "final_lower_test_accuracy_regressed",
                            },
                            {"family": "9gq", "accepted": True, "delta": 0.01},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            residual.write_text(
                json.dumps(
                    {
                        "clusters": [
                            {
                                "cluster": "0Oo",
                                "accepted": False,
                                "selection_delta": 0.1,
                                "confirmation_delta": -0.1,
                                "rejection_reason": "confirmation_regressed",
                            },
                            {
                                "cluster": "9gq",
                                "accepted": True,
                                "selection_delta": 0.1,
                                "confirmation_delta": 0.1,
                                "delta": 0.05,
                            },
                        ],
                        "promotable": True,
                        "test_delta": 0.05,
                    }
                ),
                encoding="utf-8",
            )
            ensemble.write_text(
                json.dumps(
                    {
                        "candidate_count": 3,
                        "unique_checkpoint_count": 4,
                        "duplicate_checkpoint_count": 2,
                        "baseline": {"test_accuracy": 87.0},
                        "best": {"path": "mixedcase_cnn.pt", "metrics": {"test_accuracy": 87.3}},
                        "candidates": [{"accepted": False}, {"accepted": True}],
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_probes([sweep, feature, residual, ensemble])

        self.assertEqual(summary["probe_count"], 4)
        self.assertEqual(summary["promotable_count"], 2)
        self.assertEqual(summary["accepted_count"], 3)
        self.assertEqual(summary["summaries"][0]["kind"], "sweep")
        self.assertEqual(summary["summaries"][0]["best_parameters"], {"epochs": 80})
        self.assertEqual(summary["summaries"][0]["top_family_rows"][0]["family"], "near")
        self.assertEqual(
            summary["summaries"][0]["top_family_rows"][0]["rejection_reason"],
            "final_digit_test_accuracy_regressed",
        )
        self.assertEqual(summary["summaries"][0]["top_family_rows_all_runs"][0]["family"], "near")
        self.assertEqual(
            summary["summaries"][0]["rejection_reason_counts"],
            {"final_digit_test_accuracy_regressed": 1, "selection_delta_below_floor": 1},
        )
        self.assertEqual(summary["summaries"][0]["accepted_family_counts"], {"kept": 1})
        self.assertEqual(summary["summaries"][1]["kind"], "feature_probe")
        self.assertFalse(summary["summaries"][1]["promotable"])
        self.assertEqual(summary["summaries"][1]["accepted_family_counts"], {"9gq": 1})
        self.assertEqual(
            summary["summaries"][1]["rejection_reason_counts"],
            {"final_lower_test_accuracy_regressed": 1},
        )
        self.assertEqual(summary["summaries"][1]["top_family_rows"][0]["family"], "0Oo")
        self.assertEqual(summary["summaries"][2]["accepted_clusters"], ["9gq"])
        self.assertAlmostEqual(summary["summaries"][3]["best_test_delta"], 0.3)


if __name__ == "__main__":
    unittest.main()
