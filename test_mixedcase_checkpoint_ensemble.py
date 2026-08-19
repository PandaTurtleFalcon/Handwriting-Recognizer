import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from scripts.probe_mixedcase_checkpoint_ensemble import (
    artifact_checkpoint_hash,
    calibration_artifacts_match_checkpoint,
    discover_checkpoint_paths_for,
    file_sha256,
)


class MixedcaseCheckpointEnsembleProbeTests(unittest.TestCase):
    """Focused tests for checkpoint ensemble probe bookkeeping."""

    def test_file_sha256_returns_none_for_missing_file(self) -> None:
        self.assertIsNone(file_sha256(Path("/tmp/missing-mixedcase-checkpoint.pt")))

    def test_discover_checkpoint_paths_skips_duplicate_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            deployed = root / "mixedcase_cnn.pt"
            backup_root = root / "backups"
            duplicate = backup_root / "run-a" / "mixedcase_cnn.pt"
            unique = backup_root / "run-b" / "mixedcase_cnn.pt"
            duplicate.parent.mkdir(parents=True)
            unique.parent.mkdir(parents=True)
            deployed.write_bytes(b"same checkpoint")
            duplicate.write_bytes(b"same checkpoint")
            unique.write_bytes(b"different checkpoint")

            paths, duplicates = discover_checkpoint_paths_for(deployed, (backup_root,))

        self.assertEqual(paths, [deployed, unique])
        self.assertEqual(duplicates, 1)

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
            checkpoint = root / "mixedcase_cnn.pt"
            bias = root / "mixedcase_logit_bias.pt"
            rules = root / "mixedcase_pair_rules.json"
            checkpoint.write_bytes(b"candidate")
            digest = file_sha256(checkpoint)
            torch.save({"checkpoint_sha256": digest}, bias)
            rules.write_text(f'{{"checkpoint_sha256": "{digest}"}}', encoding="utf-8")

            with (
                patch("scripts.probe_mixedcase_checkpoint_ensemble.MIXEDCASE_LOGIT_BIAS_PATH", bias),
                patch("scripts.probe_mixedcase_checkpoint_ensemble.MIXEDCASE_PAIR_RULES_PATH", rules),
            ):
                self.assertTrue(calibration_artifacts_match_checkpoint(checkpoint))

            rules.write_text('{"checkpoint_sha256": "stale"}', encoding="utf-8")
            with (
                patch("scripts.probe_mixedcase_checkpoint_ensemble.MIXEDCASE_LOGIT_BIAS_PATH", bias),
                patch("scripts.probe_mixedcase_checkpoint_ensemble.MIXEDCASE_PAIR_RULES_PATH", rules),
            ):
                self.assertFalse(calibration_artifacts_match_checkpoint(checkpoint))


if __name__ == "__main__":
    unittest.main()
