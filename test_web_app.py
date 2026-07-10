import base64
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

        def fake_metrics(path=main.METRICS_PATH):
            if path == main.METRICS_PATH:
                return {"history": [{"test_accuracy": 99.65}]}
            if path == main.ALNUM_METRICS_PATH:
                return {
                    "best_checkpoint": {
                        "test_accuracy": 96.66,
                        "digit_test_accuracy": 99.53,
                        "letter_test_accuracy": 95.28,
                    }
                }
            if path == main.MIXEDCASE_METRICS_PATH:
                return {
                    "best_checkpoint": {
                        "test_accuracy": 80.50,
                        "digit_test_accuracy": 99.08,
                        "upper_test_accuracy": 71.90,
                        "lower_test_accuracy": 84.10,
                        "ambiguity_aware_test_accuracy": 96.42,
                    }
                }
            if path == main.CHARACTER_METRICS_PATH:
                return {
                    "best_checkpoint": {
                        "validation_accuracy": 85.57,
                        "punctuation_validation_accuracy": 88.41,
                        "ambiguity_aware_validation_accuracy": 95.68,
                        "punctuation_ambiguity_aware_validation_accuracy": 95.99,
                    }
                }
            return {}

        with patch.object(main, "read_metrics", side_effect=fake_metrics):
            html = main.render_page()

        self.assertIn("alnum", html)
        self.assertIn("visual ambiguity 96.42%", html)
        self.assertIn("digit specialist", html)
        self.assertIn("punctuation 88.41%", html)
        self.assertIn("ambiguity-aware 95.99%", html)

    def test_best_metric_entry_prefers_checkpoint_eval(self) -> None:
        """Checkpoint eval should count even if the latest run history regressed."""

        metrics = {
            "history": [{"test_accuracy": 95.0}, {"test_accuracy": 96.0}],
            "best_checkpoint": {"test_accuracy": 97.0},
        }

        self.assertEqual(main.best_metric_entry(metrics)["test_accuracy"], 97.0)

    def test_best_metric_entry_considers_named_checkpoint_evals(self) -> None:
        """Named side-eval checkpoints should be eligible for badge display."""

        metrics = {
            "history": [{"validation_accuracy": 88.0}],
            "best_checkpoint": {"validation_accuracy": 89.0},
            "combined_extra_best_checkpoint": {"validation_accuracy": 90.5},
        }

        self.assertEqual(main.best_metric_entry(metrics, key="validation_accuracy")["validation_accuracy"], 90.5)

    def test_character_stack_prefers_exact_case_alnum_model(self) -> None:
        """Serving should use the mixed-case helper before the folded helper."""

        character_model = object()
        letter_model = object()
        mixedcase_model = object()
        folded_model = object()

        with patch.object(main, "load_character_model", return_value=(character_model, ["H"])):
            with patch.object(main, "load_letter_model", return_value=(letter_model, ["H"])):
                with patch.object(main, "load_mixedcase_model", return_value=(mixedcase_model, ["H", "i"])):
                    with patch.object(main, "load_alnum_model", return_value=(folded_model, ["H"])) as folded_loader:
                        stack = main.load_character_recognizer_stack(object())

        self.assertEqual(stack, (character_model, ["H"], letter_model, ["H"], mixedcase_model, ["H", "i"]))
        folded_loader.assert_not_called()

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

    def test_classify_files_keeps_visible_boxes_for_dropped_context_rows(self) -> None:
        """Dropped display rows should not make sequence corrections untrainable."""

        fake_predictions = [
            {"label": "H", "confidence": 0.9, "x": 1, "y": 1, "width": 20, "height": 20, "row": 1},
            {"label": "L", "confidence": 0.8, "x": 25, "y": 1, "width": 10, "height": 20, "row": 1},
            {"label": ":", "confidence": 0.9, "x": 12, "y": 40, "width": 5, "height": 18, "row": 2},
        ]
        previous_kind = main.MnistWebHandler.recognizer_kind
        previous_labels = main.MnistWebHandler.labels
        main.MnistWebHandler.recognizer_kind = "characters"
        main.MnistWebHandler.labels = ["H", "L", ":"]
        try:
            with patch.object(main, "predict_characters", return_value=fake_predictions):
                with patch.object(main, "load_model", return_value=object()):
                    with patch.object(main, "predict_digits", return_value=[]):
                        results = main.classify_files([("greeting.png", png_bytes())], model=object(), device=object())
        finally:
            main.MnistWebHandler.recognizer_kind = previous_kind
            main.MnistWebHandler.labels = previous_labels

        self.assertEqual(results[0]["sequence"], "Hi")
        self.assertEqual(results[0]["raw_sequence"], "HL:")
        self.assertEqual([item["label"] for item in results[0]["correction_predictions"]], ["H", "L"])

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

    def test_large_phone_image_is_resized_before_prediction(self) -> None:
        """Normal phone photos should be accepted and downscaled for inference."""

        image = Image.new("RGB", (2100, 2100), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        fake_predictions = [
            {"digit": 8, "confidence": 0.9, "x": 1, "y": 1, "width": 20, "height": 20, "row": 1}
        ]

        with patch.object(main, "predict_digits", return_value=fake_predictions) as mock_predict:
            results = main.classify_files([("huge.png", buffer.getvalue())], model=object(), device=object())

        predicted_image = mock_predict.call_args.args[1]
        self.assertEqual(results[0]["sequence"], "8")
        self.assertLessEqual(predicted_image.width * predicted_image.height, main.MAX_IMAGE_PIXELS)

    def test_no_predictions_returns_useful_error_with_preview(self) -> None:
        """Blank uploads should show a useful error and still render preview."""

        with patch.object(main, "predict_digits", return_value=[]):
            results = main.classify_files([("blank.png", png_bytes())], model=object(), device=object())

        self.assertEqual(results[0]["error"], "No handwriting-like marks were detected.")
        self.assertIn("preview", results[0])

    def test_classify_files_can_skip_saving_source_images(self) -> None:
        """Synthetic evaluators should not pollute the user-correction upload set."""

        with patch.object(main, "predict_digits", return_value=[]):
            with patch.object(main, "save_correction_source_image") as mock_save:
                main.classify_files([("blank.png", png_bytes())], model=object(), device=object(), save_sources=False)

        mock_save.assert_not_called()

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
            {"label": "4", "confidence": 0.98, "row": 1},
            {"label": "5", "confidence": 0.97, "row": 1},
            {"label": "J", "confidence": 0.70, "row": 1},
            {"label": "7", "confidence": 0.96, "row": 1},
        ]
        digit_predictions = [
            {"digit": 4, "confidence": 0.99, "row": 1},
            {"digit": 5, "confidence": 0.98, "row": 1},
            {"digit": 5, "confidence": 0.97, "row": 1},
            {"digit": 7, "confidence": 0.99, "row": 1},
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

    def test_digit_specialist_router_keeps_mixed_ambiguous_words(self) -> None:
        """A short ambiguous mixed string should not be rewritten as digits."""

        character_predictions = [
            {"label": "S", "confidence": 0.70, "row": 1},
            {"label": "0", "confidence": 0.95, "row": 1},
        ]
        digit_predictions = [
            {"digit": 5, "confidence": 0.99, "row": 1},
            {"digit": 0, "confidence": 0.98, "row": 1},
        ]

        self.assertFalse(main.should_use_digit_specialist_predictions(character_predictions, digit_predictions))

    def test_digit_specialist_router_keeps_ambiguous_real_letters(self) -> None:
        """High-confidence B/S/O-style letters should not become all digits."""

        character_predictions = [
            {"label": "B", "confidence": 0.91, "row": 1},
            {"label": "7", "confidence": 0.94, "row": 1},
        ]
        digit_predictions = [
            {"digit": 8, "confidence": 0.99, "row": 1},
            {"digit": 7, "confidence": 0.98, "row": 1},
        ]

        self.assertFalse(main.should_use_digit_specialist_predictions(character_predictions, digit_predictions))

    def test_visual_twin_resolver_uses_row_height_for_case(self) -> None:
        """Short glyphs in a mixed-height row can resolve to lowercase twins."""

        predictions = [
            {"label": "T", "confidence": 0.94, "x": 0, "height": 60, "row": 1, "alternatives": []},
            {"label": "E", "confidence": 0.90, "x": 30, "height": 58, "row": 1, "alternatives": []},
            {
                "label": "S",
                "confidence": 0.71,
                "x": 60,
                "height": 34,
                "row": 1,
                "alternatives": [{"label": "s", "confidence": 0.25}, {"label": "S", "confidence": 0.71}],
            },
            {"label": "T", "confidence": 0.93, "x": 90, "height": 60, "row": 1, "alternatives": []},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual([item["label"] for item in resolved], ["T", "E", "s", "T"])

    def test_visual_twin_resolver_requires_clear_height_evidence_for_case(self) -> None:
        """Case geometry should not rewrite rows whose glyph heights are similar."""

        predictions = [
            {"label": "T", "confidence": 0.94, "x": 0, "height": 60, "row": 1, "alternatives": []},
            {"label": "E", "confidence": 0.90, "x": 30, "height": 56, "row": 1, "alternatives": []},
            {
                "label": "S",
                "confidence": 0.71,
                "x": 60,
                "height": 52,
                "row": 1,
                "alternatives": [{"label": "s", "confidence": 0.25}, {"label": "S", "confidence": 0.71}],
            },
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual([item["label"] for item in resolved], ["T", "E", "S"])

    def test_visual_twin_resolver_uses_pair_geometry_for_case_twins(self) -> None:
        """Two-glyph case pairs should use size when the correct twin is plausible."""

        predictions = [
            {
                "label": "C",
                "confidence": 0.70,
                "x": 0,
                "width": 55,
                "height": 52,
                "row": 1,
                "alternatives": [{"label": "c", "confidence": 0.27}],
            },
            {
                "label": "C",
                "confidence": 0.44,
                "x": 65,
                "width": 36,
                "height": 41,
                "row": 1,
                "alternatives": [{"label": "c", "confidence": 0.27}],
            },
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual([item["label"] for item in resolved], ["C", "c"])

    def test_visual_twin_resolver_does_not_flip_reversed_case_pairs(self) -> None:
        """The pair rule is intentionally limited to upper-then-lower evidence."""

        predictions = [
            {
                "label": "c",
                "confidence": 0.60,
                "x": 0,
                "width": 36,
                "height": 41,
                "row": 1,
                "alternatives": [{"label": "C", "confidence": 0.20}],
            },
            {
                "label": "C",
                "confidence": 0.70,
                "x": 65,
                "width": 55,
                "height": 52,
                "row": 1,
                "alternatives": [{"label": "c", "confidence": 0.27}],
            },
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual([item["label"] for item in resolved], ["c", "C"])

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
        self.assertIn('data-fill-correction="s"', html)
        self.assertIn('title="Use s for this character"', html)
        self.assertIn("ambiguous with s 58.0%", html)
        self.assertIn('class="digit uncertain"', html)
        self.assertIn('class="digit-box uncertain"', html)

    def test_ambiguity_note_ignores_unrelated_alternatives(self) -> None:
        """Only known visual lookalikes should get a special ambiguity note."""

        prediction = {
            "label": "A",
            "confidence": 0.90,
            "alternatives": [
                {"label": "A", "confidence": 0.90},
                {"label": "R", "confidence": 0.30},
            ],
        }

        self.assertEqual(main.ambiguity_note(prediction), "")

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

    def test_correction_inputs_have_room_to_type(self) -> None:
        """Per-character correction inputs should not collapse beside Save."""

        self.assertIn(".digits {", main.PAGE_CSS)
        self.assertIn("minmax(160px, 1fr)", main.PAGE_CSS)
        self.assertIn(".correction-form {\n  display: grid;\n  grid-template-columns: 1fr;", main.PAGE_CSS)
        self.assertIn(".correction-form button {\n  width: 100%;", main.PAGE_CSS)

    def test_static_website_files_define_upload_and_api_flow(self) -> None:
        """The browser UI should live in standalone HTML, CSS, and JS files."""

        html = (main.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        css = (main.WEB_ROOT / "styles.css").read_text(encoding="utf-8")
        js = (main.WEB_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn('<form class="upload-panel" id="upload-form"', html)
        self.assertIn('id="practice-canvas"', html)
        self.assertIn('id="practice-form"', html)
        self.assertIn('id="practice-coverage"', html)
        self.assertIn('id="practice-readiness"', html)
        self.assertIn('id="practice-target-progress"', html)
        self.assertIn('id="practice-auto-next"', html)
        self.assertIn('id="practice-next-needed"', html)
        self.assertIn("HEIC", html)
        self.assertIn(".heif", html)
        self.assertIn('href="/styles.css"', html)
        self.assertIn('src="/app.js"', html)
        self.assertIn(".correction-form", css)
        self.assertIn(".practice-panel", css)
        self.assertIn(".practice-toggle", css)
        self.assertIn(".practice-target-progress", css)
        self.assertIn(".practice-focus", css)
        self.assertIn(".practice-focus-button", css)
        self.assertIn(".coverage-chip", css)
        self.assertIn(".readiness-card", css)
        self.assertIn(".readiness-meter", css)
        self.assertIn(".readiness-meter-fill", css)
        self.assertIn(".readiness-next", css)
        self.assertIn(".readiness-next-button", css)
        self.assertIn("touch-action: none", css)
        self.assertIn("grid-template-columns: minmax(52px, 1fr) auto", css)
        self.assertIn('fetch("/api/predict"', js)
        self.assertIn('fetch("/api/correct"', js)
        self.assertIn('fetch("/api/correction-coverage"', js)
        self.assertIn('fetch("/api/correction-readiness"', js)
        self.assertIn("practiceLabels", js)
        self.assertIn("practiceAutoNextInput", js)
        self.assertIn("practiceLabelValuesFromCoverage", js)
        self.assertIn("renderSelectedPracticeProgress", js)
        self.assertIn("selectedPracticeCoverage", js)
        self.assertIn("repeatPracticeStatus", js)
        self.assertIn("more ${label} needed", js)
        self.assertIn("focus_labels", js)
        self.assertIn("practice-focus-button", js)
        self.assertIn("renderCorrectionReadiness", js)
        self.assertIn("handlePracticeShortcut", js)
        self.assertIn("submitPracticeSample", js)
        self.assertIn('event.key === "Escape"', js)
        self.assertIn('event.altKey && event.key.toLowerCase() === "n"', js)
        self.assertIn("readiness-meter-fill", js)
        self.assertIn("next_needed", js)
        self.assertIn("readiness-next-button", js)
        self.assertIn("renderPracticeLabelButtons", js)
        self.assertIn("nextNeededPracticeLabel", js)
        self.assertIn("refreshPracticeCoverage(true)", js)
        self.assertIn("source_image", js)

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
        self.assertEqual(main.ambiguity_note(prediction), "ambiguous with s 80.0%")

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
            b"&prediction_boxes=%5B"
            b"%7B%22original_label%22%3A%22H%22%2C%22bbox%22%3A%7B%22x%22%3A1%2C%22y%22%3A2%2C%22width%22%3A3%2C%22height%22%3A4%2C%22row%22%3A1%7D%7D%2C"
            b"%7B%22original_label%22%3A%22L%22%2C%22bbox%22%3A%7B%22x%22%3A5%2C%22y%22%3A2%2C%22width%22%3A3%2C%22height%22%3A4%2C%22row%22%3A1%7D%7D%2C"
            b"%7B%22original_label%22%3A%22%3A%22%2C%22bbox%22%3A%7B%22x%22%3A9%2C%22y%22%3A2%2C%22width%22%3A3%2C%22height%22%3A4%2C%22row%22%3A1%7D%7D"
            b"%5D"
        )

        form = main.parse_correction_form(body)
        record = main.build_correction_record(form)

        self.assertEqual(record["correction_kind"], "sequence")
        self.assertEqual(record["prediction_index"], 0)
        self.assertEqual(record["original_label"], "HL:")
        self.assertEqual(record["corrected_label"], "Hi!")
        self.assertEqual(
            record["prediction_boxes"],
            [
                {"original_label": "H", "bbox": {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0, "row": 1}},
                {"original_label": "L", "bbox": {"x": 5.0, "y": 2.0, "width": 3.0, "height": 4.0, "row": 1}},
                {"original_label": ":", "bbox": {"x": 9.0, "y": 2.0, "width": 3.0, "height": 4.0, "row": 1}},
            ],
        )

    def test_sequence_correction_rejects_unaligned_text(self) -> None:
        """Whole-result training records need one corrected label per box."""

        body = (
            b"correction_kind=sequence&filename=sample.png&sequence=H&prediction_index=0"
            b"&original_label=H&corrected_label=Hi%21&confidence=0&bbox=%7B%7D"
            b"&prediction_boxes=%5B%7B%22original_label%22%3A%22H%22%2C%22bbox%22%3A%7B%22x%22%3A1%7D%7D%5D"
        )

        form = main.parse_correction_form(body)

        with self.assertRaisesRegex(ValueError, "detected character count"):
            main.build_correction_record(form)

    def test_unchanged_cleaned_sequence_saves_raw_training_labels(self) -> None:
        """Saving a cleaned display unchanged should not train on cleanup output."""

        body = (
            b"correction_kind=sequence&filename=sample.png&sequence=HL&display_sequence=Hi&prediction_index=0"
            b"&original_label=HL&corrected_label=Hi&confidence=0&bbox=%7B%7D"
            b"&prediction_boxes=%5B"
            b"%7B%22original_label%22%3A%22H%22%2C%22bbox%22%3A%7B%22x%22%3A1%7D%7D%2C"
            b"%7B%22original_label%22%3A%22L%22%2C%22bbox%22%3A%7B%22x%22%3A2%7D%7D"
            b"%5D"
        )

        form = main.parse_correction_form(body)
        record = main.build_correction_record(form)

        self.assertEqual(record["corrected_label"], "HL")

    def test_multiline_sequence_correction_flattens_row_separators(self) -> None:
        """Multi-row corrections should train the four boxes, not include newlines."""

        body = (
            b"correction_kind=sequence&filename=sample.png&sequence=ABCD&display_sequence=AB%0ACD&prediction_index=0"
            b"&original_label=ABCD&corrected_label=AX%0ACY&confidence=0&bbox=%7B%7D"
            b"&prediction_boxes=%5B"
            b"%7B%22original_label%22%3A%22A%22%2C%22bbox%22%3A%7B%22x%22%3A1%7D%7D%2C"
            b"%7B%22original_label%22%3A%22B%22%2C%22bbox%22%3A%7B%22x%22%3A2%7D%7D%2C"
            b"%7B%22original_label%22%3A%22C%22%2C%22bbox%22%3A%7B%22x%22%3A3%7D%7D%2C"
            b"%7B%22original_label%22%3A%22D%22%2C%22bbox%22%3A%7B%22x%22%3A4%7D%7D"
            b"%5D"
        )

        form = main.parse_correction_form(body)
        record = main.build_correction_record(form)

        self.assertEqual(record["corrected_label"], "AXCY")

    def test_full_result_correction_includes_prediction_boxes(self) -> None:
        """Whole-result corrections should include per-glyph boxes for training."""

        html = main.render_full_correction_form(
            {
                "filename": "sample.png",
                "image_id": "img123",
                "sequence": "HL",
                "predictions": [
                    {"label": "H", "x": 1, "y": 2, "width": 30, "height": 40, "row": 1},
                    {"label": "L", "x": 45, "y": 2, "width": 10, "height": 40, "row": 1},
                ],
            }
        )

        self.assertIn('name="prediction_boxes"', html)
        self.assertIn("&quot;original_label&quot;:&quot;H&quot;", html)
        self.assertIn("&quot;width&quot;:30", html)

    def test_full_result_correction_uses_visible_training_boxes(self) -> None:
        """A display-cleaned result should post only boxes that match the text field."""

        html = main.render_full_correction_form(
            {
                "filename": "sample.png",
                "image_id": "img123",
                "sequence": "Hi",
                "raw_sequence": "HL:",
                "correction_predictions": [
                    {"label": "H", "x": 1, "y": 2, "width": 30, "height": 40, "row": 1},
                    {"label": "L", "x": 45, "y": 2, "width": 10, "height": 40, "row": 1},
                ],
                "predictions": [
                    {"label": "H", "x": 1, "y": 2, "width": 30, "height": 40, "row": 1},
                    {"label": "L", "x": 45, "y": 2, "width": 10, "height": 40, "row": 1},
                    {"label": ":", "x": 12, "y": 60, "width": 5, "height": 12, "row": 2},
                ],
            }
        )

        self.assertIn('value="Hi"', html)
        self.assertIn('name="sequence" value="HL"', html)
        self.assertIn('name="display_sequence" value="Hi"', html)
        self.assertIn('name="original_label" value="HL"', html)
        self.assertNotIn("&quot;original_label&quot;:&quot;:&quot;", html)

    def test_character_correction_rejects_multi_character_text(self) -> None:
        """Per-character corrections should be trainable one-character labels."""

        body = (
            b"filename=sample.png&sequence=HL&prediction_index=2&original_label=L"
            b"&corrected_label=Hi&confidence=0.8&bbox=%7B%22x%22%3A1%7D"
        )

        form = main.parse_correction_form(body)

        with self.assertRaisesRegex(ValueError, "exactly one character"):
            main.build_correction_record(form)

    def test_visual_twin_resolver_handles_s5s_shape(self) -> None:
        """A narrow final 5 with S/s alternatives can be lowercase s."""

        predictions = [
            {"label": "5", "confidence": 0.99, "x": 1, "y": 1, "width": 66, "height": 50, "row": 1, "alternatives": [{"label": "S", "confidence": 0.69}]},
            {"label": "5", "confidence": 0.99, "x": 80, "y": 1, "width": 65, "height": 50, "row": 1, "alternatives": [{"label": "S", "confidence": 0.001}]},
            {"label": "5", "confidence": 0.99, "x": 150, "y": 1, "width": 38, "height": 42, "row": 1, "alternatives": [{"label": "S", "confidence": 0.80}, {"label": "s", "confidence": 0.15}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "S5s")

    def test_visual_twin_resolver_handles_ss5_width_pattern(self) -> None:
        """A narrow middle 5 between wide 5s is likely lowercase s."""

        predictions = [
            {"label": "5", "confidence": 0.99, "x": 1, "y": 1, "width": 66, "height": 52, "row": 1, "alternatives": [{"label": "S", "confidence": 0.30}]},
            {"label": "5", "confidence": 0.99, "x": 80, "y": 1, "width": 38, "height": 41, "row": 1, "alternatives": []},
            {"label": "5", "confidence": 0.99, "x": 150, "y": 1, "width": 65, "height": 53, "row": 1, "alternatives": [{"label": "s", "confidence": 0.30}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "Ss5")

    def test_visual_twin_resolver_handles_tighter_5_width_spread(self) -> None:
        """Font stress cases have smaller but still useful S/s/5 width spread."""

        predictions = [
            {"label": "5", "confidence": 0.99, "x": 1, "y": 1, "width": 58, "height": 70, "row": 1, "alternatives": [{"label": "S", "confidence": 0.40}]},
            {"label": "5", "confidence": 0.99, "x": 70, "y": 1, "width": 44, "height": 52, "row": 1, "alternatives": []},
            {"label": "5", "confidence": 0.99, "x": 125, "y": 1, "width": 48, "height": 66, "row": 1, "alternatives": [{"label": "s", "confidence": 0.25}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "Ss5")

    def test_visual_twin_resolver_keeps_numeric_555_with_weak_single_alt(self) -> None:
        """A real numeric 555 row should not become letters from one weak side guess."""

        predictions = [
            {"label": "5", "confidence": 0.99, "x": 1, "y": 1, "width": 58, "height": 70, "row": 1, "alternatives": [{"label": "S", "confidence": 0.11}]},
            {"label": "5", "confidence": 0.99, "x": 70, "y": 1, "width": 44, "height": 52, "row": 1, "alternatives": []},
            {"label": "5", "confidence": 0.99, "x": 125, "y": 1, "width": 48, "height": 66, "row": 1, "alternatives": []},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "555")

    def test_visual_twin_resolver_handles_mixed_s5s_labels(self) -> None:
        """S/s/5 can arrive as mixed labels before geometry cleanup."""

        predictions = [
            {"label": "5", "confidence": 0.99, "x": 1, "y": 1, "width": 49, "height": 63, "row": 1, "alternatives": [{"label": "S", "confidence": 0.65}]},
            {"label": "5", "confidence": 0.99, "x": 60, "y": 1, "width": 45, "height": 63, "row": 1, "alternatives": []},
            {"label": "S", "confidence": 0.41, "x": 120, "y": 1, "width": 36, "height": 49, "row": 1, "alternatives": [{"label": "s", "confidence": 0.26}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "S5s")

    def test_visual_twin_resolver_handles_ooo_shape(self) -> None:
        """Circle glyphs can use row-relative width for O/o/0 ordering."""

        predictions = [
            {"label": "0", "confidence": 0.99, "x": 1, "y": 1, "width": 61, "height": 52, "row": 1, "alternatives": [{"label": "O", "confidence": 0.23}]},
            {"label": "O", "confidence": 0.99, "x": 80, "y": 1, "width": 35, "height": 41, "row": 1, "alternatives": [{"label": "o", "confidence": 0.12}]},
            {"label": "O", "confidence": 0.99, "x": 130, "y": 1, "width": 46, "height": 48, "row": 1, "alternatives": [{"label": "0", "confidence": 0.50}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "Oo0")

    def test_visual_twin_resolver_uses_height_to_break_ooo_ties(self) -> None:
        """When 0 and o have the same width, the shorter glyph should be o."""

        predictions = [
            {"label": "o", "confidence": 0.99, "x": 1, "y": 1, "width": 48, "height": 67, "row": 1, "alternatives": [{"label": "0", "confidence": 0.51}]},
            {"label": "O", "confidence": 0.99, "x": 60, "y": 1, "width": 66, "height": 70, "row": 1, "alternatives": []},
            {"label": "0", "confidence": 0.99, "x": 135, "y": 1, "width": 48, "height": 52, "row": 1, "alternatives": [{"label": "o", "confidence": 0.17}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "0Oo")

    def test_visual_twin_resolver_keeps_numeric_000_without_letter_evidence(self) -> None:
        """Pure numeric 000 should not be rewritten using geometry alone."""

        predictions = [
            {"label": "0", "confidence": 0.99, "x": 1, "y": 1, "width": 48, "height": 67, "row": 1, "alternatives": []},
            {"label": "0", "confidence": 0.99, "x": 60, "y": 1, "width": 66, "height": 70, "row": 1, "alternatives": []},
            {"label": "0", "confidence": 0.99, "x": 135, "y": 1, "width": 48, "height": 52, "row": 1, "alternatives": []},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "000")

    def test_visual_twin_resolver_handles_2zz_shape(self) -> None:
        """Relative width can resolve 2/Z/z triples."""

        predictions = [
            {"label": "2", "confidence": 0.99, "x": 1, "y": 1, "width": 57, "height": 51, "row": 1, "alternatives": []},
            {"label": "2", "confidence": 0.99, "x": 70, "y": 1, "width": 79, "height": 59, "row": 1, "alternatives": [{"label": "Z", "confidence": 0.73}]},
            {"label": "Z", "confidence": 0.99, "x": 160, "y": 1, "width": 52, "height": 45, "row": 1, "alternatives": [{"label": "z", "confidence": 0.05}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "2Zz")

    def test_visual_twin_resolver_handles_tighter_2zz_width_spread(self) -> None:
        """Font stress cases have smaller but still useful 2/Z/z width spread."""

        predictions = [
            {"label": "2", "confidence": 0.99, "x": 1, "y": 1, "width": 49, "height": 66, "row": 1, "alternatives": [{"label": "Z", "confidence": 0.38}]},
            {"label": "Z", "confidence": 0.99, "x": 60, "y": 1, "width": 56, "height": 66, "row": 1, "alternatives": []},
            {"label": "Z", "confidence": 0.99, "x": 125, "y": 1, "width": 44, "height": 48, "row": 1, "alternatives": [{"label": "z", "confidence": 0.01}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "2Zz")

    def test_visual_twin_resolver_keeps_numeric_222_without_letter_evidence(self) -> None:
        """Pure numeric 222 should not be rewritten using geometry alone."""

        predictions = [
            {"label": "2", "confidence": 0.99, "x": 1, "y": 1, "width": 49, "height": 66, "row": 1, "alternatives": []},
            {"label": "2", "confidence": 0.99, "x": 60, "y": 1, "width": 56, "height": 66, "row": 1, "alternatives": []},
            {"label": "2", "confidence": 0.99, "x": 125, "y": 1, "width": 44, "height": 48, "row": 1, "alternatives": []},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "222")

    def test_visual_twin_resolver_handles_il1_shape(self) -> None:
        """Increasing-width skinny strokes before ! can map to I/l/1."""

        predictions = [
            {"label": "1", "confidence": 0.99, "x": 1, "y": 1, "width": 20, "height": 50, "row": 1},
            {"label": "i", "confidence": 0.99, "x": 40, "y": 1, "width": 32, "height": 57, "row": 1},
            {"label": "1", "confidence": 0.99, "x": 80, "y": 1, "width": 46, "height": 46, "row": 1},
            {"label": "!", "confidence": 0.99, "x": 130, "y": 1, "width": 21, "height": 52, "row": 1},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "Il1!")

    def test_visual_twin_resolver_handles_skinny_strokes_before_bang(self) -> None:
        """I/l/1/! rows can have the l and 1 labels swapped by the model."""

        predictions = [
            {"label": "I", "confidence": 0.99, "x": 1, "y": 1, "width": 52, "height": 70, "row": 1, "alternatives": []},
            {"label": "1", "confidence": 0.99, "x": 62, "y": 1, "width": 26, "height": 78, "row": 1, "alternatives": [{"label": "l", "confidence": 0.58}]},
            {"label": "1", "confidence": 0.99, "x": 100, "y": 1, "width": 39, "height": 71, "row": 1, "alternatives": [{"label": "I", "confidence": 0.63}]},
            {"label": "!", "confidence": 0.99, "x": 150, "y": 1, "width": 24, "height": 74, "row": 1, "alternatives": []},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "Il1!")

    def test_visual_twin_resolver_handles_skinny_stroke_triplets(self) -> None:
        """The final narrow stroke in 1Il/I1l rows is often an l."""

        predictions = [
            {"label": "1", "confidence": 0.99, "x": 1, "y": 1, "width": 39, "height": 71, "row": 1, "alternatives": []},
            {"label": "I", "confidence": 0.99, "x": 50, "y": 1, "width": 52, "height": 70, "row": 1, "alternatives": []},
            {"label": "1", "confidence": 0.99, "x": 110, "y": 1, "width": 26, "height": 78, "row": 1, "alternatives": [{"label": "l", "confidence": 0.58}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "1Il")

    def test_visual_twin_resolver_handles_9qg_variants(self) -> None:
        """9/q/g rows should use alternatives to keep the descender as q."""

        predictions = [
            {"label": "G", "confidence": 0.97, "x": 1, "y": 1, "width": 53, "height": 74, "row": 1, "alternatives": [{"label": "9", "confidence": 0.52}]},
            {"label": "Q", "confidence": 0.91, "x": 62, "y": 1, "width": 47, "height": 73, "row": 1, "alternatives": [{"label": "q", "confidence": 0.44}]},
            {"label": "g", "confidence": 0.86, "x": 120, "y": 1, "width": 50, "height": 73, "row": 1, "alternatives": []},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "9qg")

    def test_visual_twin_resolver_handles_y_pair_with_strong_alternative(self) -> None:
        """A leading Y can be classified as 4 while still exposing Y as an alt."""

        predictions = [
            {"label": "4", "confidence": 0.96, "x": 1, "y": 1, "width": 61, "height": 66, "row": 1, "alternatives": [{"label": "Y", "confidence": 0.93}]},
            {"label": "y", "confidence": 0.71, "x": 70, "y": 1, "width": 49, "height": 67, "row": 1, "alternatives": []},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "Yy")

    def test_visual_twin_resolver_handles_y_pair_when_both_read_as_four(self) -> None:
        """Yy can be read as 44 only when both glyphs expose strong Y/y evidence."""

        predictions = [
            {"label": "4", "confidence": 0.99, "x": 1, "y": 1, "width": 58, "height": 71, "row": 1, "alternatives": [{"label": "Y", "confidence": 0.78}, {"label": "y", "confidence": 0.20}]},
            {"label": "4", "confidence": 0.99, "x": 70, "y": 1, "width": 52, "height": 73, "row": 1, "alternatives": [{"label": "Y", "confidence": 0.72}, {"label": "y", "confidence": 0.28}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "Yy")

    def test_visual_twin_resolver_keeps_plain_numeric_forty_four(self) -> None:
        """Plain 44 should not become Yy without Y/y alternatives."""

        predictions = [
            {"label": "4", "confidence": 0.99, "x": 1, "y": 1, "width": 57, "height": 72, "row": 1, "alternatives": [{"label": "4", "confidence": 0.97}]},
            {"label": "4", "confidence": 0.99, "x": 70, "y": 1, "width": 57, "height": 72, "row": 1, "alternatives": [{"label": "4", "confidence": 0.97}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "44")

    def test_visual_twin_resolver_handles_b8_with_strong_b_alternative(self) -> None:
        """A B8 row can start as 88 when only the first glyph exposes B."""

        predictions = [
            {"label": "8", "confidence": 0.99, "x": 1, "y": 1, "width": 65, "height": 55, "row": 1, "alternatives": [{"label": "B", "confidence": 0.96}]},
            {"label": "8", "confidence": 0.99, "x": 80, "y": 1, "width": 48, "height": 53, "row": 1, "alternatives": [{"label": "8", "confidence": 0.91}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "B8")

    def test_visual_twin_resolver_handles_kk_with_strong_leading_k_alternative(self) -> None:
        """Kk can be read as kk when the first glyph still exposes strong K."""

        predictions = [
            {"label": "k", "confidence": 0.82, "x": 1, "y": 1, "width": 65, "height": 56, "row": 1, "alternatives": [{"label": "K", "confidence": 0.85}]},
            {"label": "k", "confidence": 0.98, "x": 80, "y": 1, "width": 56, "height": 63, "row": 1, "alternatives": [{"label": "K", "confidence": 0.08}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "Kk")

    def test_visual_twin_resolver_handles_mm_by_height(self) -> None:
        """Mm can be read as MM when the second glyph is shorter and has m evidence."""

        predictions = [
            {"label": "M", "confidence": 0.98, "x": 1, "y": 1, "width": 64, "height": 66, "row": 1, "alternatives": [{"label": "M", "confidence": 0.98}]},
            {"label": "M", "confidence": 0.51, "x": 80, "y": 1, "width": 66, "height": 53, "row": 1, "alternatives": [{"label": "m", "confidence": 0.45}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "Mm")

    def test_visual_twin_resolver_handles_g6b_with_weak_six_alternative(self) -> None:
        """Arial-like G6b can expose the 6 only as a weak alternative."""

        predictions = [
            {"label": "G", "confidence": 0.99, "x": 1, "y": 1, "width": 64, "height": 70, "row": 1, "alternatives": []},
            {"label": "G", "confidence": 0.97, "x": 75, "y": 1, "width": 49, "height": 67, "row": 1, "alternatives": [{"label": "6", "confidence": 0.03}]},
            {"label": "b", "confidence": 0.91, "x": 135, "y": 1, "width": 47, "height": 67, "row": 1, "alternatives": []},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "G6b")

    def test_visual_twin_resolver_keeps_ggb_with_noise_level_six_alt(self) -> None:
        """A noise-level 6 alternative should not silently corrupt GGb."""

        predictions = [
            {"label": "G", "confidence": 0.99, "x": 1, "y": 1, "width": 64, "height": 70, "row": 1, "alternatives": []},
            {"label": "G", "confidence": 0.97, "x": 75, "y": 1, "width": 63, "height": 70, "row": 1, "alternatives": [{"label": "6", "confidence": 0.03}]},
            {"label": "b", "confidence": 0.91, "x": 135, "y": 1, "width": 47, "height": 67, "row": 1, "alternatives": []},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "GGb")

    def test_visual_twin_resolver_handles_three_skinny_strokes(self) -> None:
        """Width order can separate 1/I/l when the third stroke has l alternatives."""

        predictions = [
            {"label": "1", "confidence": 0.99, "x": 1, "y": 1, "width": 20, "height": 49, "row": 1, "alternatives": [{"label": "I", "confidence": 0.20}]},
            {"label": "1", "confidence": 0.99, "x": 40, "y": 1, "width": 46, "height": 46, "row": 1, "alternatives": [{"label": "I", "confidence": 0.91}]},
            {"label": "i", "confidence": 0.99, "x": 100, "y": 1, "width": 32, "height": 57, "row": 1, "alternatives": [{"label": "L", "confidence": 0.69}, {"label": "l", "confidence": 0.14}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "I1l")

    def test_visual_twin_resolver_handles_known_short_words(self) -> None:
        """Word-shaped rows can resolve repeated skinny-stroke lookalikes."""

        predictions = [
            {"label": "H", "confidence": 0.99, "x": 1, "y": 1, "width": 67, "height": 57, "row": 1, "alternatives": []},
            {"label": "e", "confidence": 0.91, "x": 80, "y": 1, "width": 34, "height": 47, "row": 1, "alternatives": []},
            {"label": "i", "confidence": 0.97, "x": 120, "y": 1, "width": 32, "height": 57, "row": 1, "alternatives": [{"label": "L", "confidence": 0.69}, {"label": "l", "confidence": 0.14}]},
            {"label": "i", "confidence": 0.97, "x": 160, "y": 1, "width": 32, "height": 57, "row": 1, "alternatives": [{"label": "L", "confidence": 0.69}, {"label": "l", "confidence": 0.14}]},
            {"label": "o", "confidence": 0.99, "x": 200, "y": 1, "width": 35, "height": 41, "row": 1, "alternatives": []},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "Hello")

    def test_visual_twin_resolver_handles_t3s7_shape(self) -> None:
        """Strong S/s and 7 alternatives should recover a mixed T3s7 code."""

        predictions = [
            {"label": "T", "confidence": 0.99, "x": 1, "y": 1, "width": 65, "height": 50, "row": 1},
            {"label": "3", "confidence": 0.99, "x": 80, "y": 1, "width": 52, "height": 46, "row": 1},
            {"label": "5", "confidence": 0.99, "x": 140, "y": 1, "width": 38, "height": 41, "row": 1, "alternatives": [{"label": "S", "confidence": 0.80}]},
            {"label": "T", "confidence": 0.99, "x": 190, "y": 1, "width": 71, "height": 68, "row": 1, "alternatives": [{"label": "7", "confidence": 0.68}]},
        ]

        resolved = main.resolve_visual_twin_predictions(predictions)

        self.assertEqual("".join(main.prediction_value(item) for item in resolved), "T3s7")

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

    def test_save_practice_source_image_accepts_png_data_url(self) -> None:
        """Generated practice corrections should save a crop source image."""

        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir) / "uploads"
            data_url = "data:image/png;base64," + base64.b64encode(png_bytes()).decode("ascii")
            with patch.object(main, "CORRECTION_UPLOAD_DIR", upload_dir):
                main.save_practice_source_image({"source_image": data_url}, "practice-abc")

            saved = upload_dir / "practice-abc.png"
            self.assertTrue(saved.exists())
            with Image.open(saved) as image:
                self.assertEqual(image.size, (32, 32))

    def test_save_practice_source_image_rejects_bad_data_url(self) -> None:
        """Practice image upload should reject non-PNG data urls."""

        with self.assertRaises(ValueError):
            main.save_practice_source_image({"source_image": "data:text/plain;base64,abc"}, "practice-abc")

    def test_build_correction_coverage_report_tracks_needed_labels(self) -> None:
        """Practice coverage should show which weak labels still need samples."""

        report = main.build_correction_coverage_report({"O": 20, "0": 3}, labels=["0", "O"], target_per_label=20)

        self.assertEqual(report["ready_labels"], 1)
        self.assertEqual(report["total_labels"], 2)
        self.assertEqual(report["focus_labels"], ["0"])
        self.assertEqual(report["labels"][0]["needed"], 17)
        self.assertFalse(report["labels"][0]["ready"])
        self.assertTrue(report["labels"][1]["ready"])

    def test_practice_priority_labels_cover_mixedcase_and_punctuation_twins(self) -> None:
        """Practice samples should target the audited exact-recognition blockers."""

        for label in ["q", "g", "F", "f", "U", "u", "T", "t", "7", ":", ";", "!", "+"]:
            self.assertIn(label, main.PRACTICE_PRIORITY_LABELS)
        self.assertEqual(len(main.PRACTICE_PRIORITY_LABELS), len(set(main.PRACTICE_PRIORITY_LABELS)))

    def test_practice_priority_labels_start_with_verified_worst_labels(self) -> None:
        """Practice collection should attack the measured exact-recognition gaps first."""

        self.assertEqual(main.PRACTICE_PRIORITY_LABELS[:10], ["s", "O", "V", "1", "c", "I", "F", "o", "m", "0"])

    def test_correction_readiness_report_exposes_training_gates(self) -> None:
        """The app should expose machine-readable correction readiness."""

        report = main.correction_readiness_report()

        self.assertTrue(report["ok"])
        self.assertIn("readiness", report["character"])
        self.assertIn("next_needed", report["character"])
        self.assertIn("readiness", report["folded_alnum"])
        self.assertIn("readiness", report["mixedcase"])


if __name__ == "__main__":
    unittest.main()
