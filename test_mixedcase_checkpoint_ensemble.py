import tempfile
import unittest
from pathlib import Path

from scripts.probe_mixedcase_checkpoint_ensemble import discover_checkpoint_paths_for, file_sha256


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


if __name__ == "__main__":
    unittest.main()
