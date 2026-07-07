import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from alnum_model import _chars74k_sample_label
from extra_alnum_datasets import load_labeled_image_folder


def tiny_transform(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.resize((28, 28)), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class ExtraAlnumDatasetTests(unittest.TestCase):
    """Regression tests for optional local alphanumeric datasets."""

    def test_loads_image_folder_classes_into_label_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for label in ("0", "A"):
                class_dir = root / label
                class_dir.mkdir()
                image = Image.new("L", (18, 18), 255)
                draw = ImageDraw.Draw(image)
                draw.text((4, 2), label, fill=0)
                image.save(class_dir / f"{label}.png")

            images, targets = load_labeled_image_folder(root, ["0", "1", "A"], tiny_transform)

        self.assertEqual(tuple(images.shape), (2, 1, 28, 28))
        self.assertEqual(targets.tolist(), [0, 2])

    def test_rejects_unknown_class_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "not-a-label").mkdir()

            with self.assertRaisesRegex(RuntimeError, "unsupported class folders"):
                load_labeled_image_folder(root, ["0", "A"], tiny_transform)

    def test_chars74k_labels_fold_to_alphanumeric_targets(self) -> None:
        self.assertEqual(_chars74k_sample_label(Path("Sample001")), 0)
        self.assertEqual(_chars74k_sample_label(Path("Sample010")), 9)
        self.assertEqual(_chars74k_sample_label(Path("Sample011")), 10)
        self.assertEqual(_chars74k_sample_label(Path("Sample036")), 35)
        self.assertEqual(_chars74k_sample_label(Path("Sample037")), 10)
        self.assertEqual(_chars74k_sample_label(Path("Sample062")), 35)
        self.assertIsNone(_chars74k_sample_label(Path("Sample063")))


if __name__ == "__main__":
    unittest.main()
