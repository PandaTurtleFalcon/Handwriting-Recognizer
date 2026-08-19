import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.probe_character_checkpoint_ensemble import (
    AverageLogitModel,
    artifact_checkpoint_hash,
    calibration_artifacts_match_checkpoint,
    discover_checkpoint_paths_for,
    file_sha256,
    metric_delta,
    rejection_reason,
)


class CharacterCheckpointEnsembleProbeTests(unittest.TestCase):
    """Focused tests for character checkpoint ensemble probe bookkeeping."""

    def test_file_sha256_returns_none_for_missing_file(self) -> None:
        self.assertIsNone(file_sha256(Path("/tmp/missing-character-checkpoint.pt")))

    def test_discover_checkpoint_paths_skips_duplicate_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            deployed = root / "character_cnn.pt"
            backup_root = root / "backups"
            duplicate = backup_root / "run-a" / "character_cnn.pt"
            unique = backup_root / "run-b" / "character_cnn.pt"
            duplicate.parent.mkdir(parents=True)
            unique.parent.mkdir(parents=True)
            deployed.write_bytes(b"same checkpoint")
            duplicate.write_bytes(b"same checkpoint")
            unique.write_bytes(b"different checkpoint")

            paths, duplicates = discover_checkpoint_paths_for(deployed, (backup_root,))

        self.assertEqual(paths, [deployed, unique])
        self.assertEqual(duplicates, 1)

    def test_average_logit_model_averages_outputs(self) -> None:
        first = torch.nn.Linear(2, 2, bias=False)
        second = torch.nn.Linear(2, 2, bias=False)
        first.weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        second.weight.data = torch.tensor([[3.0, 0.0], [0.0, 5.0]])

        output = AverageLogitModel([first, second])(torch.tensor([[2.0, 4.0]]))

        self.assertTrue(torch.allclose(output, torch.tensor([[4.0, 12.0]])))

    def test_metric_delta_subtracts_baseline(self) -> None:
        self.assertEqual(metric_delta({"a": 3.0, "b": 4.0}, {"a": 1.0, "b": 5.0}), {"a": 2.0, "b": -1.0})

    def test_artifact_checkpoint_hash_reads_torch_and_json_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            torch_artifact = root / "bias.pt"
            json_artifact = root / "rules.json"
            torch.save({"checkpoint_sha256": "abc123"}, torch_artifact)
            json_artifact.write_text('{"checkpoint_sha256": "def456"}', encoding="utf-8")

            self.assertEqual(artifact_checkpoint_hash(torch_artifact), "abc123")
            self.assertEqual(artifact_checkpoint_hash(json_artifact), "def456")

    def test_calibration_artifacts_must_match_candidate_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "character_cnn.pt"
            bias = root / "character_logit_bias.pt"
            rules = root / "character_pair_rules.json"
            checkpoint.write_bytes(b"candidate")
            digest = file_sha256(checkpoint)
            torch.save({"checkpoint_sha256": digest}, bias)
            rules.write_text(f'{{"checkpoint_sha256": "{digest}"}}', encoding="utf-8")

            with (
                patch("scripts.probe_character_checkpoint_ensemble.LOGIT_BIAS_PATH", bias),
                patch("scripts.probe_character_checkpoint_ensemble.PAIR_RULES_PATH", rules),
            ):
                self.assertTrue(calibration_artifacts_match_checkpoint(checkpoint))

            rules.write_text('{"checkpoint_sha256": "stale"}', encoding="utf-8")
            with (
                patch("scripts.probe_character_checkpoint_ensemble.LOGIT_BIAS_PATH", bias),
                patch("scripts.probe_character_checkpoint_ensemble.PAIR_RULES_PATH", rules),
            ):
                self.assertFalse(calibration_artifacts_match_checkpoint(checkpoint))

    def test_rejection_reason_requires_gain_and_protected_splits(self) -> None:
        baseline = {
            "validation_accuracy": 94.0,
            "ambiguity_aware_validation_accuracy": 99.0,
            "digit_validation_accuracy": 95.0,
            "letter_validation_accuracy": 93.5,
            "punctuation_validation_accuracy": 96.0,
        }
        small_gain = {**baseline, "validation_accuracy": 94.005}
        letter_regression = {**baseline, "validation_accuracy": 94.02, "letter_validation_accuracy": 93.4}
        safe = {**baseline, "validation_accuracy": 94.02}

        self.assertEqual(rejection_reason(baseline, small_gain, min_delta=0.01), "validation_delta_below_floor")
        self.assertEqual(
            rejection_reason(baseline, letter_regression, min_delta=0.01),
            "letter_validation_accuracy_regressed",
        )
        self.assertIsNone(rejection_reason(baseline, safe, min_delta=0.01))


if __name__ == "__main__":
    unittest.main()
