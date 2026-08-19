import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from alnum_model import MIXEDCASE_LABELS
from scripts.prepare_mixedcase_family_pack import (
    ROUGH_SCRIPT_SOURCE,
    balanced_indices_for_labels,
    build_family_pack,
    family_label_indices,
    load_named_source,
    parse_families,
    save_family_pack,
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

    def test_rough_script_source_generates_and_loads_ascii_folder(self) -> None:
        rough_root = Path("/tmp/generated-rough-test")
        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        targets = torch.tensor([MIXEDCASE_LABELS.index("1"), MIXEDCASE_LABELS.index("I")])

        with (
            patch("scripts.prepare_mixedcase_family_pack.generate_rough_character_variants") as generate,
            patch("scripts.prepare_mixedcase_family_pack._mixedcase_ascii_folder_cache_path") as cache_path,
            patch("scripts.prepare_mixedcase_family_pack.load_mixedcase_extra_cache", return_value=(images, targets)) as load,
        ):
            stale_cache = Mock()
            stale_cache.exists.return_value = True
            cache_path.return_value = stale_cache

            loaded_images, loaded_targets = load_named_source(
                ROUGH_SCRIPT_SOURCE,
                rough_root=rough_root,
                rough_samples_per_label=3,
                seed=99,
            )

        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs["samples_per_label"], 3)
        self.assertEqual(generate.call_args.kwargs["seed"], 99)
        stale_cache.unlink.assert_called_once_with()
        load.assert_called_once_with(rough_root)
        self.assertIs(loaded_images, images)
        self.assertIs(loaded_targets, targets)

    def test_save_family_pack_writes_metadata_and_report(self) -> None:
        output = Path("tmp/test-family-pack.pt")
        images = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        metadata = {"seed": 123, "counts": {"0": 1, "1": 1}}

        report = save_family_pack(output, images, targets, metadata)
        loaded = torch.load(output, weights_only=False)

        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["seed"], 123)
        self.assertTrue(torch.equal(loaded["images"], images))
        self.assertTrue(torch.equal(loaded["targets"], targets))
        self.assertEqual(loaded["metadata"], metadata)


if __name__ == "__main__":
    unittest.main()
