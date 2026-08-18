import tempfile
import unittest
from pathlib import Path

import torch

from alnum_model import LABELS, MIXEDCASE_LABELS
from scripts.calibrate_mixedcase_hybrid import (
    _load_hybrid_artifact,
    hybrid_metrics,
    hybrid_predictions,
)


class MixedcaseHybridCalibrationTests(unittest.TestCase):
    """Unit coverage for repeatable mixed-case hybrid threshold calibration."""

    def test_hybrid_predictions_match_folded_identity_case_probe(self) -> None:
        """Folded identity should only override alphabetic mixed-case predictions."""

        mixed_outputs = torch.full((3, len(MIXEDCASE_LABELS)), -10.0)
        mixed_outputs[0, MIXEDCASE_LABELS.index("5")] = 8.0
        mixed_outputs[1, MIXEDCASE_LABELS.index("Q")] = 7.0
        mixed_outputs[1, MIXEDCASE_LABELS.index("q")] = 7.4
        mixed_outputs[2, MIXEDCASE_LABELS.index("B")] = 7.4
        mixed_outputs[2, MIXEDCASE_LABELS.index("b")] = 7.0
        folded_outputs = torch.full((3, len(LABELS)), -10.0)
        folded_outputs[0, LABELS.index("S")] = 9.0
        folded_outputs[1, LABELS.index("A")] = 9.0
        folded_outputs[2, LABELS.index("B")] = 9.0
        artifact = {
            "letter_case_threshold": 0.0,
            "folded_confidence_threshold": 0.25,
            "folded_margin_threshold": 0.5,
            "letter_case_thresholds": {"A": 1.0},
        }

        predictions = hybrid_predictions(mixed_outputs, folded_outputs, artifact)

        self.assertEqual(MIXEDCASE_LABELS[int(predictions[0])], "5")
        self.assertEqual(MIXEDCASE_LABELS[int(predictions[1])], "A")
        self.assertEqual(MIXEDCASE_LABELS[int(predictions[2])], "B")

    def test_hybrid_metrics_include_balanced_group_accuracy(self) -> None:
        """The calibration objective should expose weakest digit/upper/lower split."""

        predictions = torch.tensor(
            [
                MIXEDCASE_LABELS.index("1"),
                MIXEDCASE_LABELS.index("A"),
                MIXEDCASE_LABELS.index("b"),
                MIXEDCASE_LABELS.index("B"),
            ]
        )
        targets = torch.tensor(
            [
                MIXEDCASE_LABELS.index("1"),
                MIXEDCASE_LABELS.index("A"),
                MIXEDCASE_LABELS.index("b"),
                MIXEDCASE_LABELS.index("c"),
            ]
        )

        metrics = hybrid_metrics(predictions, targets, list(MIXEDCASE_LABELS))

        self.assertEqual(metrics["digit_test_accuracy"], 100.0)
        self.assertEqual(metrics["upper_test_accuracy"], 100.0)
        self.assertEqual(metrics["lower_test_accuracy"], 50.0)
        self.assertEqual(metrics["balanced_group_accuracy"], 50.0)

    def test_load_hybrid_artifact_rejects_wrong_label_order(self) -> None:
        """A stale or malformed artifact should fall back to safe defaults."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mixedcase_hybrid.json"
            path.write_text('{"labels":["bad"],"letter_case_threshold":9}', encoding="utf-8")

            artifact = _load_hybrid_artifact(path)

        self.assertEqual(artifact["labels"], list(MIXEDCASE_LABELS))
        self.assertEqual(artifact["letter_case_threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
