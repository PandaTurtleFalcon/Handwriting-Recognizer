import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
