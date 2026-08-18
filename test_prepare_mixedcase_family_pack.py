import unittest
from unittest.mock import patch

import torch

from alnum_model import MIXEDCASE_LABELS
from scripts.prepare_mixedcase_family_pack import (
    balanced_indices_for_labels,
    build_family_pack,
    family_label_indices,
    parse_families,
)


class PrepareMixedcaseFamilyPackTests(unittest.TestCase):
    """Focused tests for balanced hard-family pack preparation."""

    def test_parse_families_keeps_requested_order(self) -> None:
        self.assertEqual(parse_families(" 1Iil, 0Oo ,, Ss "), ("1Iil", "0Oo", "Ss"))

    def test_family_label_indices_deduplicates_labels(self) -> None:
        labels = family_label_indices(("1Iil", "lS"))

        self.assertEqual([MIXEDCASE_LABELS[index] for index in labels], ["1", "I", "i", "l", "S"])

    def test_balanced_indices_for_labels_caps_each_label(self) -> None:
        targets = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])

        selected = balanced_indices_for_labels(targets, (0, 2), max_per_label=2, seed=4)
        selected_targets = targets.index_select(0, selected)

        self.assertEqual(selected_targets.tolist().count(0), 2)
        self.assertEqual(selected_targets.tolist().count(2), 2)
        self.assertNotIn(1, selected_targets.tolist())

    def test_build_family_pack_combines_balanced_sources(self) -> None:
        one_index = MIXEDCASE_LABELS.index("1")
        i_index = MIXEDCASE_LABELS.index("I")
        first_images = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
        first_targets = torch.tensor([one_index, one_index, i_index, i_index])
        second_images = torch.arange(4, 8, dtype=torch.float32).view(4, 1, 1, 1)
        second_targets = torch.tensor([one_index, i_index, i_index, i_index])

        with patch(
            "scripts.prepare_mixedcase_family_pack.load_named_source",
            side_effect=[(first_images, first_targets), (second_images, second_targets)],
        ):
            images, targets, metadata = build_family_pack(
                ("first", "second"),
                ("1I",),
                max_per_label_per_source=1,
                seed=7,
            )

        self.assertEqual(int(images.shape[0]), 4)
        self.assertEqual(targets.tolist().count(one_index), 2)
        self.assertEqual(targets.tolist().count(i_index), 2)
        self.assertEqual(metadata["counts"], {"1": 2, "I": 2})


if __name__ == "__main__":
    unittest.main()
