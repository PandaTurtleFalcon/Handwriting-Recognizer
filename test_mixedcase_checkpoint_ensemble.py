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
    test_tensors as load_probe_test_tensors,
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

    def test_discover_checkpoint_paths_includes_timestamped_backups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            deployed = root / "mixedcase_cnn.pt"
            backup_root = root / "backups"
            timestamped = backup_root / "mixedcase_cnn.20260709_152412.pt"
            candidate = backup_root / "mixedcase_balanced_family_cap250_lr75e8.pt"
            timestamped.parent.mkdir(parents=True)
            deployed.write_bytes(b"current")
            timestamped.write_bytes(b"older candidate")
            candidate.write_bytes(b"named candidate")

            paths, duplicates = discover_checkpoint_paths_for(deployed, (backup_root,))

        self.assertEqual(paths, [deployed, timestamped, candidate])
        self.assertEqual(duplicates, 0)

    def test_artifact_checkpoint_hash_reads_torch_and_json_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            torch_artifact = root / "bias.pt"
            json_artifact = root / "rules.json"
            torch.save({"checkpoint_sha256": "abc123"}, torch_artifact)
            json_artifact.write_text('{"checkpoint_sha256": "def456"}', encoding="utf-8")

            self.assertEqual(artifact_checkpoint_hash(torch_artifact), "abc123")
            self.assertEqual(artifact_checkpoint_hash(json_artifact), "def456")

    def test_test_tensors_can_take_deterministic_sample_limit(self) -> None:
        images = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
        targets = torch.arange(10, dtype=torch.long)

        with (
            patch("scripts.probe_mixedcase_checkpoint_ensemble.build_or_load_mnist_cache", return_value=(images[:4], targets[:4])),
            patch(
                "scripts.probe_mixedcase_checkpoint_ensemble.build_or_load_emnist_byclass_mixedcase_cache",
                return_value=(images[4:], targets[4:]),
            ),
        ):
            first_images, first_targets = load_probe_test_tensors(sample_limit=5, seed=12)
            second_images, second_targets = load_probe_test_tensors(sample_limit=5, seed=12)

        self.assertEqual(int(first_targets.numel()), 5)
        self.assertEqual(first_targets.tolist(), second_targets.tolist())
        self.assertTrue(torch.equal(first_images, second_images))

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
