import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from alnum_model import LABELS, MIXEDCASE_LABELS
from scripts.calibrate_mixedcase_hybrid import (
    _load_hybrid_artifact,
    calibrate_hybrid,
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

    def test_calibrate_hybrid_defaults_to_non_regression_floors(self) -> None:
        """A lower split gain should not be accepted when upper exact regresses."""

        mixed_outputs = torch.full((4, len(MIXEDCASE_LABELS)), -10.0)
        folded_outputs = torch.full((4, len(LABELS)), -10.0)
        labels = list(MIXEDCASE_LABELS)
        upper_a = labels.index("A")
        lower_a = labels.index("a")
        folded_a = LABELS.index("A")
        targets = torch.tensor([upper_a, upper_a, lower_a, lower_a])
        for index in range(4):
            mixed_outputs[index, upper_a] = 5.0
            mixed_outputs[index, lower_a] = 4.0
            folded_outputs[index, folded_a] = 5.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mixedcase_hybrid.json"
            with (
                patch(
                    "scripts.calibrate_mixedcase_hybrid._model_outputs",
                    return_value=(mixed_outputs, folded_outputs, targets, labels),
                ),
                patch(
                    "scripts.calibrate_mixedcase_hybrid._load_hybrid_artifact",
                    return_value={
                        "enabled": True,
                        "labels": labels,
                        "letter_case_threshold": 0.0,
                        "folded_confidence_threshold": 0.0,
                        "folded_margin_threshold": 0.0,
                        "letter_case_thresholds": {},
                        "folded_confidence_thresholds": {},
                        "folded_margin_thresholds": {},
                    },
                ),
            ):
                report = calibrate_hybrid(
                    output_path=output_path,
                    batch_size=4,
                    labels_to_tune="A",
                    case_thresholds=(-2.0,),
                    confidence_thresholds=(),
                    margin_thresholds=(),
                    rounds=1,
                    objective="lower_test_accuracy",
                    min_improvement=0.01,
                    write=True,
                )

        self.assertFalse(report["wrote"])
        self.assertEqual(report["calibrated_objective"], report["base_objective"])
        self.assertFalse(output_path.exists())

    def test_calibrate_hybrid_accepts_candidate_checkpoint_paths(self) -> None:
        """Candidate calibration should not be forced through deployed weights."""

        mixed_outputs = torch.full((1, len(MIXEDCASE_LABELS)), -10.0)
        folded_outputs = torch.full((1, len(LABELS)), -10.0)
        labels = list(MIXEDCASE_LABELS)
        target = labels.index("A")
        mixed_outputs[0, target] = 5.0
        folded_outputs[0, LABELS.index("A")] = 5.0

        with tempfile.TemporaryDirectory() as temp_dir:
            mixed_path = Path(temp_dir) / "candidate.pt"
            folded_path = Path(temp_dir) / "folded.pt"
            output_path = Path(temp_dir) / "candidate_hybrid.json"
            with patch(
                "scripts.calibrate_mixedcase_hybrid._model_outputs",
                return_value=(mixed_outputs, folded_outputs, torch.tensor([target]), labels),
            ) as model_outputs:
                report = calibrate_hybrid(
                    output_path=output_path,
                    mixedcase_weights_path=mixed_path,
                    folded_weights_path=folded_path,
                    batch_size=4,
                    labels_to_tune="A",
                    case_thresholds=(),
                    confidence_thresholds=(),
                    margin_thresholds=(),
                    rounds=0,
                    min_improvement=0.0,
                    write=False,
                )

        model_outputs.assert_called_once_with(
            4,
            mixedcase_weights_path=mixed_path,
            folded_weights_path=folded_path,
        )
        self.assertEqual(report["mixedcase_weights_path"], str(mixed_path))
        self.assertEqual(report["folded_weights_path"], str(folded_path))


if __name__ == "__main__":
    unittest.main()
