import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import main


def png_bytes() -> bytes:
    """Create a tiny valid PNG for upload tests."""

    image = Image.new("RGB", (32, 32), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def multipart_body(file_count: int) -> tuple[str, bytes]:
    """Build a minimal multipart body with the requested number of files."""

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
    """Regression tests for upload parsing, classification dispatch, and HTML."""

    def test_build_row_sequences_groups_digits_by_row(self) -> None:
        """Predictions should be grouped into one string per visual row."""

        predictions = [
            {"digit": 2, "row": 2, "x": 10},
            {"digit": 1, "row": 1, "x": 20},
            {"digit": 0, "row": 1, "x": 5},
            {"digit": 3, "row": 2, "x": 30},
        ]

        self.assertEqual(main.build_row_sequences(predictions), ["01", "23"])

    def test_render_result_maps_boxes_to_numbered_cards(self) -> None:
        """Rendered boxes and cards should use matching prediction numbers."""

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
        self.assertIn('action="/correct"', html)
        self.assertIn('name="corrected_label"', html)
        self.assertIn('name="bbox"', html)
        self.assertIn("data-correction-form", html)
        self.assertIn("data-correction-status", html)
        self.assertIn("Fix the whole result", html)
        self.assertIn('name="correction_kind"', html)
        self.assertIn('value="sequence"', html)

    def test_correction_forms_have_unique_input_ids(self) -> None:
        """Multiple correction fields should be independently focusable."""

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

        self.assertEqual(html.count('name="corrected_label"'), 3)
        self.assertEqual(html.count('id="correction-'), 2)
        self.assertEqual(len(set(part.split('"', 1)[0] for part in html.split('id="')[1:])), 3)

    def test_full_result_correction_can_save_entire_text(self) -> None:
        """Whole-result correction should allow fixing every predicted character."""

        html = main.render_result(
            {
                "filename": "phrase.png",
                "sequence": "HL:",
                "row_sequences": ["HL:"],
                "preview": "data:image/png;base64,abc",
                "image_width": 100,
                "image_height": 50,
                "predictions": [
                    {"label": "H", "confidence": 0.95, "x": 10, "y": 5, "width": 20, "height": 30, "row": 1},
                    {"label": "L", "confidence": 0.75, "x": 50, "y": 5, "width": 10, "height": 30, "row": 1},
                    {"label": ":", "confidence": 0.90, "x": 70, "y": 5, "width": 10, "height": 30, "row": 1},
                ],
            }
        )

        self.assertIn('class="full-correction"', html)
        self.assertIn('value="HL:"', html)
        self.assertIn("Save all", html)

    def test_valid_image_with_bad_extension_is_decoded_by_content(self) -> None:
        """Image content should matter more than filename extension."""

        fake_predictions = [
            {"digit": 8, "confidence": 0.9, "x": 1, "y": 1, "width": 20, "height": 20, "row": 1}
        ]
        with patch.object(main, "predict_digits", return_value=fake_predictions):
            results = main.classify_files([("not-really-text.txt", png_bytes())], model=object(), device=object())

        self.assertEqual(results[0]["sequence"], "8")
        self.assertIn("preview", results[0])

    def test_render_page_keeps_corrections_on_page_with_fetch(self) -> None:
        """The page should save corrections in-place instead of replacing results."""

        html = main.render_page()

        self.assertIn("fetch(form.action", html)
        self.assertIn("Saved", html)
        self.assertIn("You can edit it again.", html)

    def test_render_page_shows_digit_specialist_accuracy(self) -> None:
        """The badge should expose separate specialist metrics."""

        html = main.render_page()

        self.assertIn("alnum", html)
        self.assertIn("digit specialist", html)

    def test_best_metric_entry_prefers_checkpoint_eval(self) -> None:
        """Checkpoint eval should count even if the latest run history regressed."""

        metrics = {
            "history": [{"test_accuracy": 95.0}, {"test_accuracy": 96.0}],
            "best_checkpoint": {"test_accuracy": 97.0},
        }

        self.assertEqual(main.best_metric_entry(metrics)["test_accuracy"], 97.0)

    def test_classify_files_applies_context_cleanup_to_display(self) -> None:
        """Obvious context cleanup should affect display text, not predictions."""

        fake_predictions = [
            {"label": "H", "confidence": 0.9, "x": 1, "y": 1, "width": 20, "height": 20, "row": 1},
            {"label": "L", "confidence": 0.8, "x": 25, "y": 1, "width": 10, "height": 20, "row": 1},
            {"label": "!", "confidence": 0.9, "x": 40, "y": 1, "width": 8, "height": 20, "row": 1},
        ]
        previous_kind = main.MnistWebHandler.recognizer_kind
        previous_labels = main.MnistWebHandler.labels
        main.MnistWebHandler.recognizer_kind = "characters"
        main.MnistWebHandler.labels = ["H", "L", "!"]
        try:
            with patch.object(main, "predict_characters", return_value=fake_predictions):
                with patch.object(main, "load_model", return_value=object()):
                    with patch.object(main, "predict_digits", return_value=[]):
                        results = main.classify_files([("greeting.png", png_bytes())], model=object(), device=object())
        finally:
            main.MnistWebHandler.recognizer_kind = previous_kind
            main.MnistWebHandler.labels = previous_labels

        self.assertEqual(results[0]["sequence"], "Hi!")
        self.assertEqual([item["label"] for item in results[0]["predictions"]], ["H", "L", "!"])
        self.assertTrue(results[0]["context_notes"])

    def test_parse_multipart_accepts_case_insensitive_content_type(self) -> None:
        """Multipart content type parsing should be case-insensitive."""

        content_type, body = multipart_body(1)

        files = main.parse_multipart_files(content_type, body)

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0][0], "digit0.png")

    def test_parse_multipart_rejects_too_many_files(self) -> None:
        """Large batches should fail before prediction work starts."""

        content_type, body = multipart_body(main.MAX_FILES + 1)

        with self.assertRaisesRegex(ValueError, "fewer files"):
            main.parse_multipart_files(content_type, body)

    def test_large_image_is_rejected_before_prediction(self) -> None:
        """Oversized images should be rejected before model inference."""

        image = Image.new("RGB", (2100, 2100), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        with patch.object(main, "predict_digits") as mock_predict:
            results = main.classify_files([("huge.png", buffer.getvalue())], model=object(), device=object())

        self.assertEqual(results[0]["error"], "Image is too large. Use an image under 4 megapixels.")
        mock_predict.assert_not_called()

    def test_no_predictions_returns_useful_error_with_preview(self) -> None:
        """Blank uploads should show a useful error and still render preview."""

        with patch.object(main, "predict_digits", return_value=[]):
            results = main.classify_files([("blank.png", png_bytes())], model=object(), device=object())

        self.assertEqual(results[0]["error"], "No handwriting-like marks were detected.")
        self.assertIn("preview", results[0])

    def test_character_mode_uses_character_predictor(self) -> None:
        """Character mode should dispatch to the expanded recognizer stack."""

        fake_predictions = [
            {"label": "A", "confidence": 0.88, "x": 1, "y": 1, "width": 20, "height": 20, "row": 1}
        ]
        previous_kind = main.MnistWebHandler.recognizer_kind
        previous_labels = main.MnistWebHandler.labels
        previous_letter_model = main.MnistWebHandler.letter_model
        previous_letter_labels = main.MnistWebHandler.letter_labels
        previous_alnum_model = main.MnistWebHandler.alnum_model
        previous_alnum_labels = main.MnistWebHandler.alnum_labels
        letter_model = object()
        alnum_model = object()
        main.MnistWebHandler.recognizer_kind = "characters"
        main.MnistWebHandler.labels = ["A"]
        main.MnistWebHandler.letter_model = letter_model
        main.MnistWebHandler.letter_labels = ["A"]
        main.MnistWebHandler.alnum_model = alnum_model
        main.MnistWebHandler.alnum_labels = ["0", "A"]
        try:
            with patch.object(main, "load_model", return_value=object()):
                with patch.object(main, "predict_digits", return_value=[]) as mock_digits:
                    with patch.object(main, "predict_characters", return_value=fake_predictions) as mock_characters:
                        results = main.classify_files([("letter.png", png_bytes())], model=object(), device=object())
        finally:
            main.MnistWebHandler.recognizer_kind = previous_kind
            main.MnistWebHandler.labels = previous_labels
            main.MnistWebHandler.letter_model = previous_letter_model
            main.MnistWebHandler.letter_labels = previous_letter_labels
            main.MnistWebHandler.alnum_model = previous_alnum_model
            main.MnistWebHandler.alnum_labels = previous_alnum_labels

        mock_digits.assert_called_once()
        self.assertIs(mock_characters.call_args.kwargs["letter_model"], letter_model)
        self.assertEqual(mock_characters.call_args.kwargs["letter_labels"], ["A"])
        self.assertIs(mock_characters.call_args.kwargs["alnum_model"], alnum_model)
        self.assertEqual(mock_characters.call_args.kwargs["alnum_labels"], ["0", "A"])
        self.assertEqual(results[0]["sequence"], "A")

    def test_digit_specialist_router_takes_digit_like_predictions(self) -> None:
        """High-confidence digit predictions should win all-digit-like uploads."""

        character_predictions = [
            {"label": "Y", "confidence": 0.98, "row": 1},
            {"label": "J", "confidence": 0.97, "row": 1},
        ]
        digit_predictions = [
            {"digit": 4, "confidence": 0.99, "row": 1},
            {"digit": 5, "confidence": 0.98, "row": 1},
        ]

        self.assertTrue(main.should_use_digit_specialist_predictions(character_predictions, digit_predictions))

    def test_digit_specialist_router_keeps_real_letters(self) -> None:
        """The digit route should not steal uploads containing clear letters."""

        character_predictions = [
            {"label": "H", "confidence": 0.98, "row": 1},
            {"label": "i", "confidence": 0.97, "row": 1},
        ]
        digit_predictions = [
            {"digit": 4, "confidence": 0.99, "row": 1},
            {"digit": 1, "confidence": 0.98, "row": 1},
        ]

        self.assertFalse(main.should_use_digit_specialist_predictions(character_predictions, digit_predictions))

    def test_result_cards_show_top_three_guesses(self) -> None:
        """Ambiguous predictions should expose the strongest alternatives."""

        html = main.render_result(
            {
                "filename": "case.png",
                "sequence": "S",
                "row_sequences": ["S"],
                "preview": "data:image/png;base64,",
                "image_width": 100,
                "image_height": 100,
                "predictions": [
                    {
                        "label": "S",
                        "confidence": 0.32,
                        "x": 1,
                        "y": 1,
                        "width": 20,
                        "height": 20,
                        "row": 1,
                        "alternatives": [
                            {"label": "s", "confidence": 0.58},
                            {"label": "S", "confidence": 0.32},
                        ],
                    }
                ],
            }
        )

        self.assertIn("top guesses:", html)
        self.assertIn("<b>s</b> 58.0%", html)
        self.assertIn("<b>S</b> 32.0%", html)
        self.assertIn('class="digit uncertain"', html)
        self.assertIn('class="digit-box uncertain"', html)

    def test_render_result_shows_context_notes(self) -> None:
        """Context cleanup notes should be visible in the result panel."""

        html = main.render_result(
            {
                "filename": "context.png",
                "sequence": "Hi!",
                "row_sequences": ["Hi!"],
                "context_notes": ["Read H followed by a skinny stroke as the greeting 'Hi'."],
                "preview": "data:image/png;base64,",
                "image_width": 100,
                "image_height": 100,
                "predictions": [
                    {"label": "H", "confidence": 0.9, "x": 1, "y": 1, "width": 20, "height": 20, "row": 1}
                ],
            }
        )

        self.assertIn("skinny stroke", html)

    def test_result_cards_limit_top_guesses_to_three(self) -> None:
        """The result card should stay compact when many alternatives exist."""

        html = main.render_result(
            {
                "filename": "many.png",
                "sequence": "8",
                "row_sequences": ["8"],
                "preview": "data:image/png;base64,",
                "image_width": 100,
                "image_height": 100,
                "predictions": [
                    {
                        "label": "8",
                        "confidence": 0.91,
                        "x": 1,
                        "y": 1,
                        "width": 20,
                        "height": 20,
                        "row": 1,
                        "alternatives": [
                            {"label": "8", "confidence": 0.91},
                            {"label": "B", "confidence": 0.40},
                            {"label": "3", "confidence": 0.22},
                            {"label": "S", "confidence": 0.18},
                        ],
                    }
                ],
            }
        )

        self.assertIn("<b>8</b> 91.0%", html)
        self.assertIn("<b>B</b> 40.0%", html)
        self.assertIn("<b>3</b> 22.0%", html)
        self.assertNotIn("<b>S</b> 18.0%", html)

    def test_low_confidence_prediction_is_marked_uncertain(self) -> None:
        """Low-confidence predictions should be visually marked as uncertain."""

        html = main.render_result(
            {
                "filename": "low.png",
                "sequence": "7",
                "row_sequences": ["7"],
                "preview": "data:image/png;base64,",
                "image_width": 100,
                "image_height": 100,
                "predictions": [
                    {"label": "7", "confidence": 0.62, "x": 1, "y": 1, "width": 20, "height": 20, "row": 1}
                ],
            }
        )

        self.assertIn('class="digit uncertain"', html)
        self.assertIn("uncertain", html)
        self.assertIn('aria-label="Prediction 1: digit 7, confidence 62.0 percent, uncertain"', html)

    def test_close_alternatives_mark_prediction_uncertain(self) -> None:
        """A close top-two margin should also get the yellow uncertain state."""

        prediction = {
            "label": "S",
            "confidence": 0.86,
            "alternatives": [
                {"label": "S", "confidence": 0.86},
                {"label": "s", "confidence": 0.80},
            ],
        }

        self.assertTrue(main.is_prediction_uncertain(prediction))

    def test_parse_correction_form_builds_training_record(self) -> None:
        """Correction forms should produce stable JSONL-ready records."""

        body = (
            b"filename=sample.png&sequence=T3L87&prediction_index=3&original_label=L"
            b"&corrected_label=%28&confidence=0.904"
            b"&bbox=%7B%22x%22%3A10%2C%22y%22%3A20%2C%22width%22%3A30%2C%22height%22%3A40%2C%22row%22%3A1%7D"
        )

        form = main.parse_correction_form(body)
        record = main.build_correction_record(form)

        self.assertEqual(record["filename"], "sample.png")
        self.assertEqual(record["correction_kind"], "character")
        self.assertEqual(record["sequence"], "T3L87")
        self.assertEqual(record["prediction_index"], 3)
        self.assertEqual(record["original_label"], "L")
        self.assertEqual(record["corrected_label"], "(")
        self.assertEqual(record["confidence"], 0.904)
        self.assertEqual(record["bbox"], {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0, "row": 1})
        self.assertIn("timestamp", record)

    def test_parse_sequence_correction_builds_training_record(self) -> None:
        """Whole-result corrections should preserve the complete corrected text."""

        body = (
            b"correction_kind=sequence&filename=sample.png&sequence=HL%3A&prediction_index=0"
            b"&original_label=HL%3A&corrected_label=Hi%21&confidence=0&bbox=%7B%7D"
        )

        form = main.parse_correction_form(body)
        record = main.build_correction_record(form)

        self.assertEqual(record["correction_kind"], "sequence")
        self.assertEqual(record["prediction_index"], 0)
        self.assertEqual(record["original_label"], "HL:")
        self.assertEqual(record["corrected_label"], "Hi!")

    def test_save_correction_appends_jsonl(self) -> None:
        """Saved corrections should append one JSON object per line."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corrections.jsonl"
            record = {
                "filename": "sample.png",
                "sequence": "S",
                "prediction_index": 1,
                "original_label": "S",
                "corrected_label": "s",
                "confidence": 0.58,
                "bbox": {"x": 1, "y": 2, "width": 3, "height": 4, "row": 1},
                "timestamp": "2026-07-07T18:00:00+00:00",
            }

            main.save_correction(record, path=path)
            main.save_correction({**record, "corrected_label": "5"}, path=path)

            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["corrected_label"], "s")
        self.assertEqual(json.loads(lines[1])["corrected_label"], "5")


if __name__ == "__main__":
    unittest.main()
