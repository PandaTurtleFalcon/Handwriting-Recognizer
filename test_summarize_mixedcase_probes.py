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
            residual = root / "residual.json"
            ensemble = root / "ensemble.json"
            sweep.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"promotable": False, "test_delta": -0.1},
                            {
                                "promotable": True,
                                "test_delta": 0.2,
                                "parameters": {"epochs": 80},
                                "base": {"test_accuracy": 87.0},
                                "reranked": {"test_accuracy": 87.2},
                            },
                        ],
                        "best": {
                            "promotable": True,
                            "test_delta": 0.2,
                            "parameters": {"epochs": 80},
                            "base": {"test_accuracy": 87.0},
                            "reranked": {"test_accuracy": 87.2},
                        },
                        "promotable_count": 1,
                        "completed_runs": 2,
                        "planned_runs": 4,
                        "families": ["0Oo"],
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

            summary = summarize_probes([sweep, residual, ensemble])

        self.assertEqual(summary["probe_count"], 3)
        self.assertEqual(summary["promotable_count"], 2)
        self.assertEqual(summary["accepted_count"], 2)
        self.assertEqual(summary["summaries"][0]["kind"], "sweep")
        self.assertEqual(summary["summaries"][0]["best_parameters"], {"epochs": 80})
        self.assertEqual(summary["summaries"][1]["accepted_clusters"], ["9gq"])
        self.assertAlmostEqual(summary["summaries"][2]["best_test_delta"], 0.3)


if __name__ == "__main__":
    unittest.main()
