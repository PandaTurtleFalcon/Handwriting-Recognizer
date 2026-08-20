import unittest
from pathlib import Path

import torch

from character_model import load_extra_character_cache
from scripts.prepare_character_hard_family_pack import (
    family_label_indices,
    hard_family_indices,
    parse_families,
    save_pack,
)


class PrepareCharacterHardFamilyPackTests(unittest.TestCase):
    """Focused tests for character hard-family pack preparation."""

    def test_parse_families_uses_defaults_for_blank_input(self) -> None:
        self.assertIn("!/1Iil|", parse_families(None))
        self.assertEqual(parse_families(" 1Ili|!/, 0Oo ,, "), ("1Ili|!/", "0Oo"))

    def test_family_label_indices_deduplicates_requested_labels(self) -> None:
        labels = ["!", "1", "I", "l", "O"]

        indices = family_label_indices(("1Il", "l!"), labels)

        self.assertEqual(indices, (1, 2, 3, 0))

    def test_hard_family_indices_selects_wrong_or_uncertain_examples(self) -> None:
        targets = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
        predictions = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
        confidences = torch.tensor([0.95, 0.95, 0.60, 0.99, 0.80, 0.40])
        margins = torch.tensor([0.90, 0.90, 0.80, 0.02, 0.50, 0.10])

        selected = hard_family_indices(
            targets,
            predictions,
            confidences,
            margins,
            label_indices=(0, 1),
            max_per_label=2,
            max_confidence=0.70,
            max_margin=0.05,
            seed=5,
        )
        selected_targets = targets.index_select(0, selected)

        self.assertLessEqual(selected_targets.tolist().count(0), 2)
        self.assertLessEqual(selected_targets.tolist().count(1), 2)
        self.assertNotIn(2, selected_targets.tolist())
        self.assertIn(1, selected.tolist())
        self.assertIn(2, selected.tolist())
        self.assertIn(3, selected.tolist())

    def test_save_pack_writes_existing_character_cache_format(self) -> None:
        output = Path("tmp/test-character-hard-family-pack.pt")
        images = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
        targets = torch.tensor([1, 2], dtype=torch.long)
        metadata = {
            "families": ["1I"],
            "counts": {"1": 1, "I": 1},
            "cache_labels": ["!", "1", "I"],
        }

        report = save_pack(output, images, targets, metadata)
        loaded = torch.load(output, weights_only=False)
        remapped_images, remapped_targets = load_extra_character_cache(output, ["1", "I"])

        self.assertEqual(report["samples"], 2)
        self.assertTrue(torch.equal(loaded["images"], images))
        self.assertTrue(torch.equal(loaded["targets"], targets))
        self.assertEqual(loaded["labels"], ["!", "1", "I"])
        self.assertEqual(loaded["metadata"], metadata)
        self.assertEqual(tuple(remapped_images.shape), (2, 1, 32, 32))
        self.assertEqual(remapped_targets.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
