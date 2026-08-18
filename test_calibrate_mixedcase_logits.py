import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.calibrate_mixedcase_logits import calibrate_mixedcase_greedy_bias
from scripts.calibrate_mixedcase_logits import calibrate_mixedcase_pair_rules
from scripts.calibrate_mixedcase_logits import main


class MixedcaseCalibrationCliTests(unittest.TestCase):
    """Regression tests for mixed-case calibration CLI safety switches."""

    def test_dry_run_overrides_write_flag(self) -> None:
        """A probe should not write an artifact when --dry-run is present."""

        output = StringIO()
        report = {
            "base_accuracy": 80.5,
            "calibrated_accuracy": 87.2,
            "best_scale": 1.0,
            "improvement": 6.7,
            "best_checkpoint": {"test_accuracy": 87.2},
            "wrote": False,
            "output_path": "mixedcase_logit_bias.pt",
        }

        with (
            patch(
                "sys.argv",
                [
                    "calibrate_mixedcase_logits.py",
                    "--write",
                    "--dry-run",
                    "--scale",
                    "1.0",
                    "--batch-size",
                    "32",
                ],
            ),
            patch("scripts.calibrate_mixedcase_logits.calibrate_mixedcase_logits", return_value=report) as calibrate,
            patch("sys.stdout", output),
        ):
            main()

        self.assertFalse(calibrate.call_args.kwargs["write"])
        self.assertEqual(calibrate.call_args.kwargs["fixed_scale"], 1.0)
        self.assertEqual(calibrate.call_args.kwargs["batch_size"], 32)
        self.assertEqual(json.loads(output.getvalue())["wrote"], False)

    def test_require_app_gates_restores_rejected_artifact(self) -> None:
        """A candidate bias should roll back when app hardcases fail."""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mixedcase_logit_bias.pt"
            output_path.write_bytes(b"previous")

            def fake_calibrate(**kwargs):
                Path(kwargs["output_path"]).write_bytes(b"candidate")
                return {
                    "base_accuracy": 80.5,
                    "calibrated_accuracy": 84.7,
                    "best_scale": 0.25,
                    "improvement": 4.2,
                    "best_checkpoint": {"test_accuracy": 84.7},
                    "wrote": True,
                    "output_path": str(kwargs["output_path"]),
                }

            output = StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "calibrate_mixedcase_logits.py",
                        "--output-path",
                        str(output_path),
                        "--write",
                        "--require-app-gates",
                    ],
                ),
                patch("scripts.calibrate_mixedcase_logits.calibrate_mixedcase_logits", side_effect=fake_calibrate),
                patch(
                    "scripts.calibrate_mixedcase_logits._app_gate_report",
                    return_value={"clean_exact": 90.91, "script_exact": 88.64, "passed": False},
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
            output_path = Path(temp_dir) / "mixedcase_logit_bias.pt"
            output_path.write_bytes(b"previous")

            def fake_calibrate(**kwargs):
                Path(kwargs["output_path"]).write_bytes(b"candidate")
                return {
                    "base_accuracy": 80.5,
                    "calibrated_accuracy": 80.7,
                    "best_scale": 0.01,
                    "improvement": 0.2,
                    "best_checkpoint": {"test_accuracy": 80.7},
                    "wrote": True,
                    "output_path": str(kwargs["output_path"]),
                }

            output = StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "calibrate_mixedcase_logits.py",
                        "--output-path",
                        str(output_path),
                        "--write",
                        "--require-app-gates",
                    ],
                ),
                patch("scripts.calibrate_mixedcase_logits.calibrate_mixedcase_logits", side_effect=fake_calibrate),
                patch(
                    "scripts.calibrate_mixedcase_logits._app_gate_report",
                    return_value={"clean_exact": 100.0, "script_exact": 95.45, "passed": True},
                ),
                patch("sys.stdout", output),
            ):
                main()

            report = json.loads(output.getvalue())
            self.assertTrue(report["wrote"])
            self.assertTrue(report["app_gates"]["passed"])
            self.assertEqual(output_path.read_bytes(), b"candidate")

    def test_greedy_bias_tunes_requested_labels(self) -> None:
        """Greedy mode should write a better per-label mixed-case bias."""

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
        labels = ["0", "A", "a"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mixedcase_logit_bias.pt"
            with (
                patch("scripts.calibrate_mixedcase_logits._mixedcase_logits", return_value=(logits, targets, train_targets, labels)),
                patch("scripts.calibrate_mixedcase_logits.MIXEDCASE_LABELS", labels),
            ):
                report = calibrate_mixedcase_greedy_bias(
                    output_path=output_path,
                    batch_size=3,
                    labels_to_tune="0",
                    deltas=(0.2,),
                    rounds=2,
                    min_improvement=0.01,
                    min_case_or_visual=0.0,
                    min_digit=0.0,
                    min_upper=0.0,
                    min_lower=0.0,
                    write=True,
                )

            self.assertTrue(report["wrote"])
            self.assertGreater(report["calibrated_accuracy"], report["base_accuracy"])
            artifact = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertEqual(artifact["tuned_labels"], ["0"])
            self.assertEqual(len(artifact["steps"]), 1)

    def test_greedy_bias_can_optimize_lowercase_split(self) -> None:
        """Greedy mode should support targeting lowercase with safety floors."""

        logits = torch.tensor(
            [
                [0.50, 0.00, 0.00],
                [0.00, 0.50, 0.00],
                [0.00, 0.20, 0.10],
                [0.00, 0.30, 0.20],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 2, 2], dtype=torch.long)
        train_targets = torch.tensor([0, 1, 2], dtype=torch.long)
        labels = ["0", "A", "a"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mixedcase_logit_bias.pt"
            with (
                patch("scripts.calibrate_mixedcase_logits._mixedcase_logits", return_value=(logits, targets, train_targets, labels)),
                patch("scripts.calibrate_mixedcase_logits.MIXEDCASE_LABELS", labels),
            ):
                report = calibrate_mixedcase_greedy_bias(
                    output_path=output_path,
                    batch_size=4,
                    labels_to_tune="a",
                    deltas=(0.2,),
                    rounds=2,
                    min_improvement=0.01,
                    objective="lower_test_accuracy",
                    min_test=0.0,
                    min_case_or_visual=0.0,
                    min_digit=100.0,
                    min_upper=100.0,
                    min_lower=0.0,
                    write=True,
                )

            self.assertTrue(report["wrote"])
            self.assertEqual(report["objective"], "lower_test_accuracy")
            self.assertGreater(report["calibrated_objective"], report["base_objective"])
            artifact = torch.load(output_path, map_location="cpu", weights_only=True)
            self.assertEqual(artifact["objective"], "lower_test_accuracy")

    def test_greedy_bias_rejects_unknown_objective(self) -> None:
        """Invalid objective names should fail before writing an artifact."""

        logits = torch.tensor([[0.50, 0.00, 0.00]], dtype=torch.float32)
        targets = torch.tensor([0], dtype=torch.long)
        train_targets = torch.tensor([0], dtype=torch.long)
        labels = ["0", "A", "a"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mixedcase_logit_bias.pt"
            with (
                patch("scripts.calibrate_mixedcase_logits._mixedcase_logits", return_value=(logits, targets, train_targets, labels)),
                patch("scripts.calibrate_mixedcase_logits.MIXEDCASE_LABELS", labels),
            ):
                with self.assertRaises(ValueError):
                    calibrate_mixedcase_greedy_bias(
                        output_path=output_path,
                        batch_size=1,
                        labels_to_tune="a",
                        objective="not_a_metric",
                    )
            self.assertFalse(output_path.exists())

    def test_pair_rules_write_exact_improving_visual_twin_flips(self) -> None:
        """Pair-rule mode should save ordered close-logit visual-twin rules."""

        logits = torch.tensor(
            [
                [0.40, 0.30, 0.00],
                [0.40, 0.20, 0.00],
                [0.10, 0.50, 0.00],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([1, 0, 1], dtype=torch.long)
        train_targets = torch.tensor([0, 1, 2], dtype=torch.long)
        labels = ["0", "O", "A"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mixedcase_pair_rules.json"
            with (
                patch("scripts.calibrate_mixedcase_logits._mixedcase_logits", return_value=(logits, targets, train_targets, labels)),
                patch("scripts.calibrate_mixedcase_logits.MIXEDCASE_LABELS", labels),
                patch("scripts.calibrate_mixedcase_logits.MIXEDCASE_LOGIT_BIAS_PATH", Path(temp_dir) / "missing.pt"),
            ):
                report = calibrate_mixedcase_pair_rules(
                    output_path=output_path,
                    batch_size=3,
                    families=("0O",),
                    thresholds=(-0.15,),
                    rounds=2,
                    min_improvement=0.01,
                    min_case_or_visual=0.0,
                    min_digit=0.0,
                    min_upper=0.0,
                    min_lower=0.0,
                    write=True,
                )

            self.assertTrue(report["wrote"])
            self.assertGreater(report["calibrated_accuracy"], report["base_accuracy"])
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["rules"][0]["from"], "0")
            self.assertEqual(artifact["rules"][0]["to"], "O")

    def test_pair_rules_can_optimize_lowercase_split(self) -> None:
        """Pair-rule mode should support targeting lowercase with safety floors."""

        logits = torch.tensor(
            [
                [0.50, 0.00, 0.00],
                [0.00, 0.50, 0.00],
                [0.00, 0.20, 0.10],
                [0.00, 0.30, 0.20],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 2, 2], dtype=torch.long)
        train_targets = torch.tensor([0, 1, 2], dtype=torch.long)
        labels = ["0", "A", "a"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mixedcase_pair_rules.json"
            with (
                patch("scripts.calibrate_mixedcase_logits._mixedcase_logits", return_value=(logits, targets, train_targets, labels)),
                patch("scripts.calibrate_mixedcase_logits.MIXEDCASE_LABELS", labels),
                patch("scripts.calibrate_mixedcase_logits.MIXEDCASE_LOGIT_BIAS_PATH", Path(temp_dir) / "missing.pt"),
            ):
                report = calibrate_mixedcase_pair_rules(
                    output_path=output_path,
                    batch_size=4,
                    families=("Aa",),
                    thresholds=(-0.15,),
                    rounds=2,
                    min_improvement=0.01,
                    objective="lower_test_accuracy",
                    min_case_or_visual=0.0,
                    min_digit=100.0,
                    min_upper=100.0,
                    min_lower=0.0,
                    write=True,
                )

            self.assertTrue(report["wrote"])
            self.assertEqual(report["objective"], "lower_test_accuracy")
            self.assertGreater(report["calibrated_objective"], report["base_objective"])
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["objective"], "lower_test_accuracy")

    def test_pair_rules_reject_test_gain_when_objective_regresses(self) -> None:
        """Optimizing lowercase should not accept rules that only help total accuracy."""

        logits = torch.tensor(
            [
                [0.30, 0.40, 0.00],
                [0.30, 0.40, 0.00],
                [0.30, 0.40, 0.00],
                [0.00, 0.00, 0.50],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 0, 1, 2], dtype=torch.long)
        train_targets = torch.tensor([0, 1, 2], dtype=torch.long)
        labels = ["0", "a", "A"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mixedcase_pair_rules.json"
            with (
                patch("scripts.calibrate_mixedcase_logits._mixedcase_logits", return_value=(logits, targets, train_targets, labels)),
                patch("scripts.calibrate_mixedcase_logits.MIXEDCASE_LABELS", labels),
                patch("scripts.calibrate_mixedcase_logits.MIXEDCASE_LOGIT_BIAS_PATH", Path(temp_dir) / "missing.pt"),
            ):
                report = calibrate_mixedcase_pair_rules(
                    output_path=output_path,
                    batch_size=4,
                    families=("0a",),
                    thresholds=(-0.15,),
                    rounds=1,
                    min_improvement=0.01,
                    objective="lower_test_accuracy",
                    min_case_or_visual=0.0,
                    min_digit=0.0,
                    min_upper=100.0,
                    min_lower=0.0,
                    write=True,
                )

            self.assertFalse(report["wrote"])
            self.assertEqual(report["calibrated_objective"], report["base_objective"])
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
