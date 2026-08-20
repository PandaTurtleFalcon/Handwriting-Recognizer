import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from scripts.analyze_character_confusions import (
    _metric_extra_roots,
    analyze_confusions,
    family_error_details,
    main,
    parse_families,
)


class CharacterConfusionAnalysisTests(unittest.TestCase):
    """Regression tests for character confusion reporting."""

    def test_metric_extra_roots_reads_existing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            extra = root / "extra"
            extra.mkdir()
            missing = root / "missing"
            metrics = root / "metrics.json"
            metrics.write_text(
                f'{{"extra_roots": ["{extra}", "{missing}"]}}',
                encoding="utf-8",
            )

            with patch("scripts.analyze_character_confusions.METRICS_PATH", metrics):
                roots = _metric_extra_roots()

        self.assertEqual(roots, [extra])

    def test_analyze_confusions_reports_punctuation_group(self) -> None:
        class FixedModel(nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros((images.size(0), 3), dtype=torch.float32)
                predictions = [1, 1, 2]
                for row, label_index in enumerate(predictions[: images.size(0)]):
                    logits[row, label_index] = 10.0
                return logits

        images = torch.zeros((3, 1, 28, 28), dtype=torch.float32)
        targets = torch.tensor([0, 1, 2], dtype=torch.long)
        labels = [".", "'", "A"]

        with (
            patch("scripts.analyze_character_confusions.get_device", return_value=torch.device("cpu")),
            patch("scripts.analyze_character_confusions.load_character_model", return_value=(FixedModel(), labels)),
            patch(
                "scripts.analyze_character_confusions.build_or_load_combined_cache",
                return_value=(images, targets, labels),
            ),
            patch("scripts.analyze_character_confusions.train_test_split", return_value=([], [0, 1, 2])),
        ):
            report = analyze_confusions(batch_size=8, top=3, extra_roots=[], families=(".'",))

        self.assertEqual(report["total"], 3)
        self.assertEqual(report["top_confusions_by_group"]["punctuation"], [{"expected": ".", "predicted": "'", "count": 1}])
        self.assertAlmostEqual(report["group_accuracy"]["punctuation"], 50.0)
        self.assertEqual(report["family_error_details"][0]["top_pairs"][0], {"expected": ".", "predicted": "'", "count": 1})

    def test_family_error_details_reports_confidence_and_margin_bands(self) -> None:
        details = family_error_details(
            expected_labels=["1", "I", "O", "A", "l"],
            predicted_labels=["I", "I", "0", "B", "!"],
            confidences=[0.65, 0.99, 0.91, 0.4, 0.72],
            margins=[0.03, 0.4, 0.2, 0.01, 0.12],
            families=("!/1Iil|", "0Oo"),
            top=3,
        )

        self.assertEqual(details[0]["family"], "!/1Iil|")
        self.assertEqual(details[0]["recoverable_errors"], 2)
        self.assertEqual(details[0]["confidence_bands"], {"<0.70": 1, "0.70-0.90": 1})
        self.assertEqual(details[0]["margin_bands"], {"<0.05": 1, "0.05-0.15": 1})
        self.assertEqual(details[0]["split_recoverable_errors"], {"digits": 1, "letters": 1})
        self.assertEqual(details[1]["family"], "0Oo")
        self.assertEqual(details[1]["recoverable_errors"], 1)

    def test_parse_families_defaults_for_blank_input(self) -> None:
        self.assertIn("!/1Iil|", parse_families(None))
        self.assertEqual(parse_families(" 1Ili|!/, 0Oo "), ("1Ili|!/", "0Oo"))

    def test_main_prints_all_confusion_groups_and_worst_labels(self) -> None:
        report = {
            "exact_accuracy": 80.0,
            "ambiguity_aware_accuracy": 95.0,
            "group_accuracy": {"digits": 70.0, "letters": 80.0, "punctuation": 90.0},
            "group_ambiguity_accuracy": {"digits": 95.0, "letters": 96.0, "punctuation": 97.0},
            "top_confusions_by_group": {
                "digits": [{"expected": "0", "predicted": "O", "count": 2}],
                "letters": [{"expected": "s", "predicted": "S", "count": 3}],
                "punctuation": [{"expected": "-", "predicted": "_", "count": 4}],
            },
            "worst_labels": [
                {"label": "0", "accuracy": 70.0, "correct": 7, "total": 10},
                {"label": "s", "accuracy": 80.0, "correct": 8, "total": 10},
            ],
            "family_error_details": [
                {
                    "family": "0Oo",
                    "recoverable_errors": 2,
                    "mean_confidence": 0.8,
                    "mean_margin": 0.2,
                }
            ],
        }
        output = StringIO()

        with (
            patch("scripts.analyze_character_confusions.analyze_confusions", return_value=report),
            patch("sys.argv", ["analyze_character_confusions.py", "--top", "2"]),
            patch("sys.stdout", output),
        ):
            main()

        text = output.getvalue()
        self.assertIn("top digits confusions:", text)
        self.assertIn("top letters confusions:", text)
        self.assertIn("top punctuation confusions:", text)
        self.assertIn("worst labels:", text)
        self.assertIn("0: 70.00% (7/10)", text)
        self.assertIn("visual-family recoverable errors:", text)
        self.assertIn("0Oo: 2", text)


if __name__ == "__main__":
    unittest.main()
