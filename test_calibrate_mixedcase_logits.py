import json
import unittest
from io import StringIO
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


if __name__ == "__main__":
    unittest.main()
