import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
