import io
import unittest
from unittest.mock import patch

from PIL import Image

import main


def png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def multipart_body(file_count: int) -> tuple[str, bytes]:
    boundary = "test-boundary"
    chunks = []
    for index in range(file_count):
        chunks.append(
            b"--"
            + boundary.encode("ascii")
            + b'\r\nContent-Disposition: form-data; name="images"; filename="digit'
            + str(index).encode("ascii")
            + b'.png"\r\nContent-Type: image/png\r\n\r\n'
            + png_bytes()
            + b"\r\n"
        )
    chunks.append(b"--" + boundary.encode("ascii") + b"--\r\n")
    return f"MULTIPART/FORM-DATA; boundary={boundary}", b"".join(chunks)


class WebAppRenderingTests(unittest.TestCase):
    def test_build_row_sequences_groups_digits_by_row(self) -> None:
        predictions = [
            {"digit": 2, "row": 2, "x": 10},
            {"digit": 1, "row": 1, "x": 20},
            {"digit": 0, "row": 1, "x": 5},
            {"digit": 3, "row": 2, "x": 30},
        ]

        self.assertEqual(main.build_row_sequences(predictions), ["01", "23"])

    def test_render_result_maps_boxes_to_numbered_cards(self) -> None:
        result = {
            "filename": "digits.png",
            "sequence": "01",
            "row_sequences": ["01"],
            "preview": "data:image/png;base64,abc",
            "image_width": 100,
            "image_height": 50,
            "predictions": [
                {"digit": 0, "confidence": 0.95, "x": 10, "y": 5, "width": 20, "height": 30, "row": 1},
                {"digit": 1, "confidence": 0.75, "x": 50, "y": 5, "width": 10, "height": 30, "row": 1},
            ],
        }

        html = main.render_result(result)

        self.assertIn('class="sequence">01</div>', html)
        self.assertIn('aria-label="Prediction 1: digit 0, confidence 95.0 percent"', html)
        self.assertIn('<span class="digit-index">#2</span> 1', html)

    def test_valid_image_with_bad_extension_is_decoded_by_content(self) -> None:
        fake_predictions = [
            {"digit": 8, "confidence": 0.9, "x": 1, "y": 1, "width": 20, "height": 20, "row": 1}
        ]
        with patch.object(main, "predict_digits", return_value=fake_predictions):
            results = main.classify_files([("not-really-text.txt", png_bytes())], model=object(), device=object())

        self.assertEqual(results[0]["sequence"], "8")
        self.assertIn("preview", results[0])

    def test_parse_multipart_accepts_case_insensitive_content_type(self) -> None:
        content_type, body = multipart_body(1)

        files = main.parse_multipart_files(content_type, body)

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0][0], "digit0.png")

    def test_parse_multipart_rejects_too_many_files(self) -> None:
        content_type, body = multipart_body(main.MAX_FILES + 1)

        with self.assertRaisesRegex(ValueError, "fewer files"):
            main.parse_multipart_files(content_type, body)

    def test_large_image_is_rejected_before_prediction(self) -> None:
        image = Image.new("RGB", (2100, 2100), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        with patch.object(main, "predict_digits") as mock_predict:
            results = main.classify_files([("huge.png", buffer.getvalue())], model=object(), device=object())

        self.assertEqual(results[0]["error"], "Image is too large. Use an image under 4 megapixels.")
        mock_predict.assert_not_called()

    def test_no_predictions_returns_useful_error_with_preview(self) -> None:
        with patch.object(main, "predict_digits", return_value=[]):
            results = main.classify_files([("blank.png", png_bytes())], model=object(), device=object())

        self.assertEqual(results[0]["error"], "No digit-like marks were detected.")
        self.assertIn("preview", results[0])


if __name__ == "__main__":
    unittest.main()
