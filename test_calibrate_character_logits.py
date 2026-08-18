import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.calibrate_character_logits import calibrate_character_greedy_bias
from scripts.calibrate_character_logits import calibrate_character_pair_rules
from scripts.calibrate_character_logits import main


class CharacterCalibrationCliTests(unittest.TestCase):
    """Regression tests for character calibration artifact safety gates."""

    def test_require_app_gates_restores_rejected_artifact(self) -> None:
        """A candidate bias should roll back when app hardcases fail."""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "character_logit_bias.pt"
            output_path.write_bytes(b"previous")

            def fake_calibrate(**kwargs):
                Path(kwargs["output_path"]).write_bytes(b"candidate")
                return {
                    "base_accuracy": 92.9,
                    "calibrated_accuracy": 93.4,
                    "best_scale": 0.8,
                    "improvement": 0.5,
                    "best_checkpoint": {"validation_accuracy": 93.4},
                    "wrote": True,
                    "output_path": str(kwargs["output_path"]),
                }

            output = StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "calibrate_character_logits.py",
                        "--output-path",
                        str(output_path),
                        "--require-app-gates",
                    ],
                ),
                patch("scripts.calibrate_character_logits.calibrate_character_logits", side_effect=fake_calibrate),
                patch(
                    "scripts.calibrate_character_logits._app_gate_report",
                    return_value={"clean_exact": 95.45, "script_exact": 92.05, "passed": False},
                ),
                patch("sys.stdout", output),
            ):
                main()

            report = json.loads(output.getvalue())
            self.assertFalse(report["wrote"])
            self.assertTrue(report["restored_after_app_gate_failure"])
            self.assertEqual(output_path.read_bytes(), b"previous")

    def test_require_app_gates_keeps_passing_artifact(self) -> None:
        """A candidate bias can stay when app hardcases remain above target."""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "character_logit_bias.pt"
            output_path.write_bytes(b"previous")

            def fake_calibrate(**kwargs):
                Path(kwargs["output_path"]).write_bytes(b"candidate")
                return {
                    "base_accuracy": 92.9,
                    "calibrated_accuracy": 93.1,
                    "best_scale": 0.2,
                    "improvement": 0.2,
                    "best_checkpoint": {"validation_accuracy": 93.1},
                    "wrote": True,
                    "output_path": str(kwargs["output_path"]),
                }

            output = StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "calibrate_character_logits.py",
                        "--output-path",
                        str(output_path),
                        "--require-app-gates",
                    ],
                ),
                patch("scripts.calibrate_character_logits.calibrate_character_logits", side_effect=fake_calibrate),
                patch(
                    "scripts.calibrate_character_logits._app_gate_report",
                    return_value={"clean_exact": 100.0, "script_exact": 95.45, "passed": True},
                ),
                patch("sys.stdout", output),
            ):
                main()

            report = json.loads(output.getvalue())
            self.assertTrue(report["wrote"])
            self.assertTrue(report["app_gates"]["passed"])
            self.assertEqual(output_path.read_bytes(), b"candidate")

    def test_pair_rules_cli_ignores_pair_rule_bias_flag(self) -> None:
        """The stacked-bias flag should not be sent to pair-rule calibration."""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "character_pair_rules.json"
            output = StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "calibrate_character_logits.py",
                        "--pair-rules",
                        "--include-pair-rules",
                        "--output-path",
                        str(output_path),
                        "--dry-run",
                    ],
                ),
                patch(
                    "scripts.calibrate_character_logits.calibrate_character_pair_rules",
                    return_value={
                        "base_accuracy": 93.0,
                        "calibrated_accuracy": 93.0,
                        "best_scale": "greedy-pair-rules",
                        "improvement": 0.0,
                        "best_checkpoint": {"validation_accuracy": 93.0},
                        "wrote": False,
                        "output_path": str(output_path),
                    },
                ) as calibrate,
                patch("sys.stdout", output),
            ):
                main()

            self.assertNotIn("include_pair_rules", calibrate.call_args.kwargs)

    def test_greedy_bias_tunes_requested_labels(self) -> None:
        """Greedy mode should write a better per-label bias when one exists."""

        logits = torch.tensor(
            [
                [0.10, 0.20, 0.00],
                [0.00, 0.40, 0.10],
                [0.00, 0.30, 0.20],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 1], dtype=torch.long)
        train_targets = torch.tensor([0, 1, 2], dtype=torch.long)
        labels = ["A", "B", "."]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "character_logit_bias.pt"
            with patch("scripts.calibrate_character_logits._validation_logits", return_value=(logits, targets, train_targets, labels)):
                report = calibrate_character_greedy_bias(
                    output_path=output_path,
                    batch_size=3,
                    labels_to_tune="A",
                    deltas=(0.2,),
                    rounds=2,
                    min_improvement=0.01,
                    min_ambiguity=0.0,
                    min_punctuation=0.0,
                    write=True,
                )

            self.assertTrue(report["wrote"])
            self.assertGreater(report["calibrated_accuracy"], report["base_accuracy"])
            artifact = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertEqual(artifact["tuned_labels"], ["A"])
            self.assertEqual(len(artifact["steps"]), 1)

    def test_greedy_bias_can_optimize_letter_split(self) -> None:
        """Greedy mode should support targeting a split metric with floors."""

        logits = torch.tensor(
            [
                [0.50, 0.00, 0.00, 0.00],
                [0.00, 0.50, 0.00, 0.00],
                [0.00, 0.20, 0.10, 0.00],
                [0.00, 0.30, 0.20, 0.00],
                [0.00, 0.00, 0.00, 0.50],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 2, 2, 3], dtype=torch.long)
        train_targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        labels = ["0", "A", "B", "."]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "character_logit_bias.pt"
            with patch("scripts.calibrate_character_logits._validation_logits", return_value=(logits, targets, train_targets, labels)):
                report = calibrate_character_greedy_bias(
                    output_path=output_path,
                    batch_size=5,
                    labels_to_tune="B",
                    deltas=(0.2,),
                    rounds=2,
                    min_improvement=0.01,
                    objective="letter_validation_accuracy",
                    min_validation=0.0,
                    min_ambiguity=0.0,
                    min_digit=100.0,
                    min_letter=0.0,
                    min_punctuation=100.0,
                    write=True,
                )

            self.assertTrue(report["wrote"])
            self.assertEqual(report["objective"], "letter_validation_accuracy")
            self.assertGreater(report["calibrated_objective"], report["base_objective"])
            artifact = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertEqual(artifact["objective"], "letter_validation_accuracy")

    def test_greedy_bias_can_evaluate_after_existing_pair_rules(self) -> None:
        """Stacked greedy mode should stamp the pair-rule artifact it measured."""

        logits = torch.tensor(
            [
                [0.40, 0.39, 0.00],
                [0.60, 0.30, 0.00],
                [0.29, 0.30, 0.00],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([1, 0, 2], dtype=torch.long)
        train_targets = torch.tensor([0, 1, 2], dtype=torch.long)
        labels = ["A", "B", "."]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "character_logit_bias.pt"
            pair_rules_path = root / "character_pair_rules.json"
            pair_rules_path.write_text(
                json.dumps(
                    {
                        "labels": labels,
                        "rules": [{"from": "A", "to": "B", "threshold": -0.02}],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("scripts.calibrate_character_logits._validation_logits", return_value=(logits, targets, train_targets, labels)),
                patch("scripts.calibrate_character_logits.PAIR_RULES_PATH", pair_rules_path),
            ):
                report = calibrate_character_greedy_bias(
                    output_path=output_path,
                    batch_size=3,
                    labels_to_tune=".",
                    deltas=(0.32,),
                    rounds=1,
                    min_improvement=0.01,
                    min_ambiguity=0.0,
                    min_punctuation=0.0,
                    include_pair_rules=True,
                    write=True,
                )

            self.assertTrue(report["wrote"])
            self.assertTrue(report["includes_pair_rules"])
            artifact = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertTrue(artifact["includes_pair_rules"])
            self.assertIsNotNone(artifact["pair_rules_sha256"])

    def test_greedy_bias_rejects_unknown_objective(self) -> None:
        """Invalid objective names should fail before writing an artifact."""

        logits = torch.tensor([[0.50, 0.00]], dtype=torch.float32)
        targets = torch.tensor([0], dtype=torch.long)
        train_targets = torch.tensor([0], dtype=torch.long)
        labels = ["A", "B"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "character_logit_bias.pt"
            with patch("scripts.calibrate_character_logits._validation_logits", return_value=(logits, targets, train_targets, labels)):
                with self.assertRaises(ValueError):
                    calibrate_character_greedy_bias(
                        output_path=output_path,
                        batch_size=1,
                        labels_to_tune="B",
                        objective="not_a_metric",
                    )
            self.assertFalse(output_path.exists())

    def test_pair_rules_write_letter_improving_visual_twin_flips(self) -> None:
        """Pair-rule mode should save close-logit flips that improve letters."""

        logits = torch.tensor(
            [
                [0.40, 0.30, 0.00],
                [0.40, 0.10, 0.00],
                [0.10, 0.50, 0.00],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([1, 0, 1], dtype=torch.long)
        train_targets = torch.tensor([0, 1, 2], dtype=torch.long)
        labels = ["0", "O", "."]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "character_pair_rules.json"
            with (
                patch("scripts.calibrate_character_logits._validation_logits", return_value=(logits, targets, train_targets, labels)),
                patch("scripts.calibrate_character_logits.LOGIT_BIAS_PATH", Path(temp_dir) / "missing.pt"),
            ):
                report = calibrate_character_pair_rules(
                    output_path=output_path,
                    batch_size=3,
                    families=("0O",),
                    thresholds=(-0.15,),
                    rounds=2,
                    min_improvement=0.01,
                    min_ambiguity=0.0,
                    min_digit=0.0,
                    min_letter=0.0,
                    min_punctuation=0.0,
                    write=True,
                )

            self.assertTrue(report["wrote"])
            self.assertGreater(report["calibrated_objective"], report["base_objective"])
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["rules"][0]["from"], "0")
            self.assertEqual(artifact["rules"][0]["to"], "O")


if __name__ == "__main__":
    unittest.main()
