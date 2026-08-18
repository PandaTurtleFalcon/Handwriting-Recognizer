import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from scripts.prepare_cvl_letters import (
    _filtered_labels,
    matching_image_for_xml,
    parse_cvl_words,
    prepare_cvl_letters,
    split_word_by_ink,
)


class PrepareCvlLettersTests(unittest.TestCase):
    """Regression tests for local CVL word-box letter extraction."""

    def test_parse_words_accepts_common_box_attribute_layouts(self) -> None:
        """CVL-like XML should produce labeled word boxes."""

        with tempfile.TemporaryDirectory() as directory:
            xml_path = Path(directory) / "sample.xml"
            xml_path.write_text(
                """
                <root>
                  <word text="Oo" x="1" y="2" width="30" height="40" />
                  <spot transcription="Il" xMin="5" yMin="6" xMax="35" yMax="46" />
                  <word text="--" x="1" y="2" width="3" height="4" />
                </root>
                """,
                encoding="utf-8",
            )

            words = parse_cvl_words(xml_path)

        self.assertEqual([word.text for word in words], ["Oo", "Il"])
        self.assertEqual(words[0].box, (1, 2, 31, 42))
        self.assertEqual(words[1].box, (5, 6, 35, 46))

    def test_matching_image_falls_back_to_prefix_match(self) -> None:
        """Cropped/full image stems can include suffixes around the XML stem."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "writer-001-cropped.png"
            image_path.touch()

            matched = matching_image_for_xml(root / "writer-001.xml", {"writer-001-cropped": image_path})

        self.assertEqual(matched, image_path)

    def test_filtered_labels_keeps_only_requested_alnum_labels(self) -> None:
        """Spaces and punctuation should not become CVL character targets."""

        self.assertEqual(_filtered_labels("Look, 15!", set("Lo15")), "Loo15")

    def test_split_word_by_ink_returns_one_crop_per_label(self) -> None:
        """Ink-mass splitting should create nonblank per-label crops."""

        image = Image.new("L", (90, 40), 255)
        draw = ImageDraw.Draw(image)
        draw.line((10, 5, 10, 35), fill=0, width=4)
        draw.ellipse((36, 8, 58, 32), outline=0, width=4)
        draw.line((78, 5, 78, 35), fill=0, width=4)

        crops = split_word_by_ink(image, "IOl")

        self.assertEqual([label for label, _ in crops], ["I", "O", "l"])
        self.assertTrue(all(crop.getextrema()[0] < 255 for _, crop in crops))

    def test_prepare_cvl_letters_writes_mixedcase_cache(self) -> None:
        """A local CVL folder should become an images/targets tensor cache."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "writer-001.png"
            image = Image.new("RGB", (120, 60), "white")
            draw = ImageDraw.Draw(image)
            draw.line((15, 10, 15, 50), fill="black", width=4)
            draw.ellipse((45, 12, 75, 48), outline="black", width=4)
            image.save(image_path)
            (root / "writer-001.xml").write_text(
                '<root><word text="IO" x="0" y="0" width="100" height="60" /></root>',
                encoding="utf-8",
            )
            output_path = root / "cvl_letters.pt"

            report = prepare_cvl_letters(root, output_path, labels="IO", limit_per_label=2)
            cache = torch.load(output_path, map_location="cpu", weights_only=True)

        self.assertEqual(report["images"], 2)
        self.assertEqual(report["classes"], 2)
        self.assertEqual(tuple(cache["images"].shape), (2, 1, 28, 28))
        self.assertEqual(cache["targets"].tolist(), [18, 24])


if __name__ == "__main__":
    unittest.main()
