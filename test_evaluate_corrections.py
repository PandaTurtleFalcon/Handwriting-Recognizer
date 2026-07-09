import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_corrections import CorrectionCase, evaluate_cases, load_cases


class EvaluateCorrectionsTests(unittest.TestCase):
    """Tests for the saved-correction app-level evaluator."""

    def test_load_cases_keeps_sequence_records_with_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            (upload_dir / "abc.png").write_bytes(b"fake-image")
            corrections = root / "corrections.jsonl"
            corrections.write_text(
                json.dumps(
                    {
                        "correction_kind": "sequence",
                        "image_id": "abc",
                        "filename": "sample.png",
                        "corrected_label": "Hi",
                    }
                )
                + "\n"
                + json.dumps({"correction_kind": "character", "image_id": "abc", "corrected_label": "H"})
                + "\n",
                encoding="utf-8",
            )

            cases = load_cases(corrections, upload_dir)

        self.assertEqual(cases, [CorrectionCase(filename="sample.png", image_path=upload_dir / "abc.png", target="Hi")])

    def test_evaluate_cases_reports_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            image_path.write_bytes(b"image")
            cases = [CorrectionCase(filename="sample.png", image_path=image_path, target="Hi")]

            with patch("scripts.evaluate_corrections.load_web_models", return_value=(object(), object())):
                with patch(
                    "scripts.evaluate_corrections.main.classify_files",
                    return_value=[{"sequence": "Hi"}],
                ):
                    report = evaluate_cases(cases)

        self.assertEqual(report["total"], 1)
        self.assertEqual(report["correct"], 1)
        self.assertEqual(report["accuracy"], 100.0)
        self.assertEqual(report["results"][0]["prediction"], "Hi")


if __name__ == "__main__":
    unittest.main()
