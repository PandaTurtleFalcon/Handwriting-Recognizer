import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
