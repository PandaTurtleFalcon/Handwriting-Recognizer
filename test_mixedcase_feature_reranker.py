import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.probe_mixedcase_feature_reranker import _fit_tensors
from scripts.probe_mixedcase_feature_reranker import geometry_features
from scripts.probe_mixedcase_feature_reranker import selected_families


class MixedcaseFeatureRerankerTests(unittest.TestCase):
    """Regression tests for the mixed-case feature-reranker probe."""

    def test_geometry_features_are_finite(self) -> None:
        """Blank and inked tensors should produce stable finite feature rows."""

        images = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        images[1, :, 8:20, 10:18] = 1.0

        features = geometry_features(images)

        self.assertEqual(tuple(features.shape), (2, 22))
        self.assertTrue(bool(torch.isfinite(features).all()))

    def test_selected_families_returns_model_label_indices(self) -> None:
        """Family probes should only include labels that exist in the 62-class model."""

        families = selected_families(limit=3)

        self.assertEqual(len(families), 3)
        self.assertTrue(all(len(family) > 1 for family in families))
        self.assertTrue(all(0 <= index < 62 for family in families for index in family))

    def test_fit_tensors_appends_capped_extra_roots(self) -> None:
        """Optional adviser data should be capped before joining fit tensors."""

        train_images = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        train_targets = torch.tensor([10], dtype=torch.long)
        extra_images = torch.ones((4, 1, 28, 28), dtype=torch.float32)
        extra_targets = torch.tensor([36, 36, 36, 37], dtype=torch.long)

        with patch(
            "scripts.probe_mixedcase_feature_reranker.load_mixedcase_extra_cache",
            return_value=(extra_images, extra_targets),
        ):
            images, targets = _fit_tensors(
                train_images,
                train_targets,
                [Path("extra.pt")],
                extra_samples_per_class=1,
                seed=11,
            )

        self.assertEqual(tuple(images.shape), (3, 1, 28, 28))
        self.assertEqual(torch.bincount(targets, minlength=62)[10].item(), 1)
        self.assertEqual(torch.bincount(targets, minlength=62)[36].item(), 1)
        self.assertEqual(torch.bincount(targets, minlength=62)[37].item(), 1)


if __name__ == "__main__":
    unittest.main()
