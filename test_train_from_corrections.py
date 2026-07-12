import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from scripts import train_from_corrections
from scripts.train_from_corrections import export_character_correction_folder
from scripts.train_from_corrections import (
    DEFAULT_MIXEDCASE_PRIORITY_LABELS,
    DEFAULT_PRIORITY_LABELS,
    correction_item_label_counts,
    correction_recommendation,
    correction_readiness_summary,
    dry_run_report,
    exportable_character_correction_counts,
    exported_character_crop_counts,
    filter_priority_labels,
    format_next_needed_summary,
    format_priority_coverage,
    format_readiness_summary,
    format_recommendation_summary,
    next_needed_labels,
)
from main import PRACTICE_PRIORITY_LABELS


class TrainFromCorrectionsTests(unittest.TestCase):
    def test_exports_sequence_corrections_as_character_ascii_folders(self) -> None:
        """Daily training should expose saved sequence corrections to character_model."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            image_id = "abc123"
            image = Image.new("RGB", (48, 24), "white")
            draw = ImageDraw.Draw(image)
            draw.line((6, 4, 6, 20), fill="black", width=2)
            draw.line((20, 4, 20, 20), fill="black", width=2)
            draw.line((6, 12, 20, 12), fill="black", width=2)
            image.save(upload_dir / f"{image_id}.png")

            corrections_path = root / "corrections.jsonl"
            corrections_path.write_text(
                json.dumps(
                    {
                        "correction_kind": "sequence",
                        "image_id": image_id,
                        "corrected_label": "Hi",
                        "prediction_boxes": [
                            {"original_label": "H", "bbox": {"x": 0, "y": 0, "width": 16, "height": 24, "row": 1}},
                            {"original_label": "L", "bbox": {"x": 16, "y": 0, "width": 16, "height": 24, "row": 1}},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_root = root / "character_ascii"
            count = export_character_correction_folder(
                ["H", "i"],
                output_root=output_root,
                corrections_path=corrections_path,
                upload_dir=upload_dir,
            )

            self.assertEqual(count, 2)
            self.assertEqual(len(list((output_root / str(ord("H"))).glob("*.png"))), 1)
            self.assertEqual(len(list((output_root / str(ord("i"))).glob("*.png"))), 1)

    def test_parser_help_is_available_without_training(self) -> None:
        """The daily training script should expose safe CLI help."""

        help_text = train_from_corrections.build_parser().format_help()

        self.assertIn("--dry-run", help_text)
        self.assertIn("--min-character-corrections", help_text)
        self.assertIn("--min-alnum-corrections", help_text)
        self.assertIn("--priority-labels", help_text)
        self.assertIn("--mixedcase-priority-labels", help_text)
        self.assertIn("--json", help_text)

    def test_default_priority_labels_match_practice_targets(self) -> None:
        """Dry-run reporting should use the same weak labels as practice mode."""

        practice_labels = "".join(PRACTICE_PRIORITY_LABELS)

        self.assertEqual(DEFAULT_PRIORITY_LABELS, practice_labels)
        self.assertEqual(DEFAULT_MIXEDCASE_PRIORITY_LABELS, practice_labels)

    def test_filters_priority_labels_to_recognizer_label_set(self) -> None:
        """Dry-run coverage should not report labels a recognizer cannot train."""

        self.assertEqual(filter_priority_labels("0Oo-+qQ", ["0", "O", "Q"]), "0OQ")

    def test_correction_readiness_summary_tracks_label_and_sample_gaps(self) -> None:
        """Dry-run readiness should show how far correction data is from training."""

        summary = correction_readiness_summary({"A": 20, "B": 3}, "ABC", target_per_label=20)

        self.assertFalse(summary["ready"])
        self.assertEqual(summary["ready_labels"], 1)
        self.assertEqual(summary["total_labels"], 3)
        self.assertEqual(summary["not_ready_labels"], 2)
        self.assertEqual(summary["samples"], 23)
        self.assertEqual(summary["target_samples"], 60)
        self.assertEqual(summary["needed_samples"], 37)
        self.assertAlmostEqual(summary["coverage_percent"], 38.333333333333336)
        self.assertEqual(
            format_readiness_summary("Character", summary),
            "Character correction readiness: not_ready labels=1/3 not_ready=2 samples=23/60 needed=37 coverage=38.33%",
        )

    def test_next_needed_labels_prioritizes_largest_gaps(self) -> None:
        """Readiness reports should expose concrete labels to collect next."""

        labels = next_needed_labels({"A": 19, "B": 0, "C": 5}, "ABC", target_per_label=20, limit=2)

        self.assertEqual(
            labels,
            [
                {"label": "B", "count": 0, "target": 20, "needed": 20, "coverage_percent": 0.0},
                {"label": "C", "count": 5, "target": 20, "needed": 15, "coverage_percent": 25.0},
            ],
        )

    def test_correction_recommendation_tracks_training_gate(self) -> None:
        """Dry-run automation should know whether to collect or train."""

        blocked = correction_recommendation(
            {"ready": False, "needed_samples": 20},
            [{"label": "s", "count": 0, "target": 20, "needed": 20, "coverage_percent": 0.0}],
        )
        ready = correction_recommendation({"ready": True, "needed_samples": 0}, [])

        self.assertEqual(blocked, {"recommended_action": "collect_corrections", "recommended_label": "s"})
        self.assertEqual(ready, {"recommended_action": "train_corrections", "recommended_label": None})
        self.assertEqual(
            format_recommendation_summary("Character", blocked),
            "Character correction recommendation: action=collect_corrections label=s",
        )
        self.assertEqual(
            format_recommendation_summary("Character", ready),
            "Character correction recommendation: action=train_corrections",
        )
        self.assertEqual(
            format_next_needed_summary("Character", {"next_needed": [{"label": "s", "needed": 20}]}),
            "Character correction next_needed: s:20",
        )
        self.assertEqual(format_next_needed_summary("Character", {"next_needed": []}), "Character correction next_needed: none")

    def test_dry_run_report_exposes_machine_readable_readiness(self) -> None:
        """Automation should be able to read correction readiness without parsing text."""

        report = dry_run_report(
            {"A": 20, "B": 2},
            {"A": 1},
            {"A": 1, "a": 1},
            folded_item_count=1,
            mixed_item_count=2,
            character_priority_labels="A-+",
            mixedcase_priority_labels="Aa-",
        )

        self.assertEqual(report["summary"]["character_crops"], 22)
        self.assertEqual(report["summary"]["folded_items"], 1)
        self.assertEqual(report["summary"]["mixedcase_items"], 2)
        self.assertEqual(report["summary"]["recommended_action"], "collect_corrections")
        self.assertEqual(report["summary"]["recommended_label"], "-")
        self.assertEqual(report["summary"]["recommended_batch_labels"], ["-", "+"])
        self.assertEqual(report["summary"]["recommended_batch_size"], 2)
        self.assertEqual(report["summary"]["recommended_batch_samples"], 0)
        self.assertEqual(report["summary"]["recommended_batch_target_samples"], 40)
        self.assertEqual(report["summary"]["recommended_batch_needed_samples"], 40)
        self.assertAlmostEqual(report["summary"]["recommended_batch_coverage_percent"], 0.0)
        self.assertEqual(report["character"]["readiness"]["needed_samples"], 40)
        self.assertEqual(report["character"]["readiness"]["not_ready_labels"], 2)
        self.assertAlmostEqual(report["character"]["readiness"]["coverage_percent"], 33.333333333333336)
        self.assertEqual(
            report["character"]["next_needed"][0],
            {"label": "-", "count": 0, "target": 20, "needed": 20, "coverage_percent": 0.0},
        )
        self.assertEqual(report["character"]["recommended_action"], "collect_corrections")
        self.assertEqual(report["character"]["recommended_label"], "-")
        self.assertEqual(report["folded_alnum"]["priority_labels"], ["A"])
        self.assertEqual(report["folded_alnum"]["recommended_action"], "collect_corrections")
        self.assertEqual(report["folded_alnum"]["recommended_label"], "A")
        self.assertEqual(report["mixedcase"]["priority_labels"], ["A", "a"])
        self.assertEqual(report["mixedcase"]["recommended_action"], "collect_corrections")
        self.assertEqual(report["mixedcase"]["recommended_label"], "A")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            train_from_corrections.print_text_dry_run_report(report)

        text = output.getvalue()
        self.assertIn("Character correction recommendation: action=collect_corrections label=-", text)
        self.assertIn("Character correction next_needed: -:20, +:20", text)
        self.assertIn("Folded alnum correction recommendation: action=collect_corrections label=A", text)
        self.assertIn("Folded alnum correction next_needed: A:19", text)
        self.assertIn("Mixed-case correction recommendation: action=collect_corrections label=A", text)
        self.assertIn("Mixed-case correction next_needed: A:19, a:19", text)

    def test_counts_exported_character_crops_by_priority_label(self) -> None:
        """Dry-run coverage should show which weak labels have examples."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, count in {"O": 2, "l": 1}.items():
                class_dir = root / str(ord(label))
                class_dir.mkdir(parents=True)
                for index in range(count):
                    Image.new("L", (8, 8), 255).save(class_dir / f"{index}.png")

            counts = exported_character_crop_counts(root)

        self.assertEqual(counts["O"], 2)
        self.assertEqual(counts["l"], 1)
        self.assertEqual(format_priority_coverage(counts, "Olo"), "O:2, l:1, o:0")

    def test_counts_loaded_correction_items_by_label(self) -> None:
        """Dry-run coverage should decode cached target indices into labels."""

        counts = correction_item_label_counts(["0", "1", "A", "a"], (object(), [1, 1, 3, 99]))

        self.assertEqual(counts["1"], 2)
        self.assertEqual(counts["a"], 1)
        self.assertNotIn("A", counts)

    def test_counts_exportable_character_corrections_before_export(self) -> None:
        """Practice samples should appear in dry-run coverage before training export."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            image_id = "practice-1"
            Image.new("RGB", (32, 32), "white").save(upload_dir / f"{image_id}.png")
            corrections_path = root / "corrections.jsonl"
            corrections_path.write_text(
                json.dumps(
                    {
                        "correction_kind": "character",
                        "image_id": image_id,
                        "corrected_label": "O",
                        "bbox": {"x": 0, "y": 0, "width": 32, "height": 32, "row": 1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            counts = exportable_character_correction_counts(
                ["0", "O"],
                corrections_path=corrections_path,
                upload_dir=upload_dir,
            )

        self.assertEqual(counts["O"], 1)

    def test_main_skips_tiny_correction_sets_without_force(self) -> None:
        """A tiny user-labeled set should not trigger daily fine-tuning by default."""

        fake_corrections = (object(), [0, 1])
        output = io.StringIO()
        with (
            patch.object(train_from_corrections, "load_correction_cache", return_value=fake_corrections),
            patch.object(train_from_corrections, "load_character_labels", return_value=["A"]),
            patch.object(train_from_corrections, "export_character_correction_folder", return_value=2),
            patch.object(train_from_corrections, "train_character_model") as train_character,
            patch.object(train_from_corrections, "train") as train_folded,
            patch.object(train_from_corrections, "train_mixedcase") as train_mixed,
            contextlib.redirect_stdout(output),
        ):
            train_from_corrections.main([])

        self.assertIn("Only 2 character correction samples", output.getvalue())
        self.assertIn("Only 2 folded alnum correction samples", output.getvalue())
        self.assertIn("Only 2 mixed-case correction samples", output.getvalue())
        train_character.assert_not_called()
        train_folded.assert_not_called()
        train_mixed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
