import tempfile
import unittest
from pathlib import Path

import torch

from scripts.probe_character_checkpoint_ensemble import (
    AverageLogitModel,
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
