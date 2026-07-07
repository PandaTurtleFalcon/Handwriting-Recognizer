"""Small stdlib web app for uploading and recognizing handwriting images.

The server intentionally avoids a web framework so the project is easy to run:
`python3 main.py` starts an upload form, predicts every uploaded image, and
renders the original image with numbered bounding boxes that match the cards.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import html
import io
import json
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps, UnidentifiedImageError

from alnum_model import METRICS_PATH as ALNUM_METRICS_PATH
from alnum_model import MIXEDCASE_METRICS_PATH, MIXEDCASE_WEIGHTS_PATH
from alnum_model import WEIGHTS_PATH as ALNUM_WEIGHTS_PATH
from alnum_model import load_mixedcase_model
from character_model import METRICS_PATH as CHARACTER_METRICS_PATH
from character_model import WEIGHTS_PATH as CHARACTER_WEIGHTS_PATH
from character_model import load_alnum_model, load_character_model, load_letter_model, predict_characters
from context_rules import cleanup_context
from emnist_experiment import METRICS_PATH as LETTER_METRICS_PATH
from emnist_experiment import WEIGHTS_PATH as LETTER_WEIGHTS_PATH
from mnist_model import METRICS_PATH, WEIGHTS_PATH, get_device, load_model, predict_digits


HOST = "127.0.0.1"
PORT = 8000
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
MAX_FILES = 20
CORRECTIONS_PATH = Path("data") / "corrections" / "corrections.jsonl"
LOW_CONFIDENCE_THRESHOLD = 0.80
CLOSE_GUESS_MARGIN = 0.12
TOP_GUESS_LIMIT = 3
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS * 2


PAGE_CSS = """
:root {
  color-scheme: light;
  --ink: #172033;
  --muted: #5f6d7e;
  --line: #d9e0ea;
  --paper: #f8fafc;
  --panel: #ffffff;
  --accent: #2563eb;
  --accent-dark: #1e40af;
  --ok: #15803d;
  --warn: #b45309;
  --warn-bg: #fffbeb;
  --warn-line: #f59e0b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--paper);
  color: var(--ink);
}
main {
  width: min(1040px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 36px 0 48px;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 28px;
}
h1 {
  margin: 0 0 6px;
  font-size: 34px;
  line-height: 1.08;
}
p {
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
}
.badge {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 12px;
  background: var(--panel);
  color: var(--ok);
  font-weight: 700;
  white-space: nowrap;
}
.workspace {
  display: grid;
  grid-template-columns: minmax(0, 0.9fr) minmax(320px, 1.1fr);
  gap: 20px;
}
.upload-panel,
.result-panel,
.empty-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 20px;
}
.upload-zone {
  display: grid;
  place-items: center;
  min-height: 240px;
  border: 2px dashed #a9b8ca;
  border-radius: 8px;
  background: #f4f7fb;
  text-align: center;
  padding: 22px;
}
input[type="file"] {
  width: 100%;
  max-width: 320px;
  margin-top: 16px;
}
button {
  display: inline-flex;
  justify-content: center;
  align-items: center;
  min-height: 44px;
  margin-top: 18px;
  width: 100%;
  border: 0;
  border-radius: 8px;
  background: var(--accent);
  color: white;
  font-size: 16px;
  font-weight: 800;
  cursor: pointer;
}
button:hover { background: var(--accent-dark); }
button:focus-visible,
input[type="file"]:focus-visible {
  outline: 3px solid #93c5fd;
  outline-offset: 3px;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
.hint {
  margin-top: 14px;
  font-size: 14px;
}
.result-panel + .result-panel { margin-top: 14px; }
.result-panel { overflow: hidden; }
.result-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 12px;
}
.filename {
  font-size: 16px;
  font-weight: 800;
  overflow-wrap: anywhere;
}
.sequence {
  font-size: 32px;
  line-height: 1;
  font-weight: 900;
  color: var(--accent-dark);
  letter-spacing: 0;
  max-width: 100%;
  overflow-wrap: anywhere;
}
.row-output {
  margin-top: 8px;
  color: var(--muted);
  font-weight: 700;
}
.row-output code {
  display: inline-block;
  margin: 4px 8px 0 0;
  padding: 4px 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #f8fafc;
  color: var(--ink);
  max-width: 100%;
  overflow-wrap: anywhere;
}
.digits {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(92px, 1fr));
  gap: 10px;
}
.preview-wrap {
  position: relative;
  margin: 14px 0;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f4f7fb;
}
.preview-wrap img {
  display: block;
  width: 100%;
  height: auto;
}
.digit-box {
  position: absolute;
  border: 3px solid #dc2626;
  border-radius: 5px;
  box-shadow: 0 0 0 2px rgb(255 255 255 / 0.9);
}
.digit-box.uncertain {
  border-color: var(--warn-line);
}
.digit-box span {
  position: absolute;
  top: 0;
  left: 0;
  min-width: 26px;
  height: 26px;
  display: grid;
  place-items: center;
  border-radius: 5px;
  background: #dc2626;
  color: #fff;
  font-size: 13px;
  font-weight: 900;
}
.digit-box.uncertain span {
  background: var(--warn-line);
}
.digit-box:hover,
.digit-box:focus-visible {
  border-color: #991b1b;
  outline: 3px solid rgb(220 38 38 / 0.25);
}
.digit-box.uncertain:hover,
.digit-box.uncertain:focus-visible {
  border-color: var(--warn);
  outline: 3px solid rgb(245 158 11 / 0.28);
}
.digit-index {
  color: #dc2626;
  font-weight: 900;
}
.digit {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: #fbfdff;
}
.digit.uncertain {
  border-color: var(--warn-line);
  background: var(--warn-bg);
}
.digit strong {
  display: block;
  font-size: 26px;
}
.digit span {
  color: var(--muted);
  font-size: 13px;
}
.alternatives {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.35;
  color: var(--muted);
}
.alternatives b {
  color: var(--ink);
}
.uncertain-note {
  display: inline-block;
  margin-top: 7px;
  color: var(--warn);
  font-size: 12px;
  font-weight: 800;
}
.full-correction {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  margin: 12px 0 14px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f8fafc;
}
.full-correction label {
  grid-column: 1 / -1;
  color: var(--muted);
  font-size: 13px;
  font-weight: 800;
}
.full-correction input[type="text"] {
  min-width: 0;
  height: 38px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0 10px;
  font: inherit;
}
.full-correction button {
  width: auto;
  min-height: 38px;
  margin: 0;
  padding: 0 12px;
  font-size: 13px;
}
.correction-form {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  margin-top: 10px;
}
.correction-form input[type="text"] {
  min-width: 0;
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0 9px;
  font: inherit;
}
.correction-form button {
  width: auto;
  min-height: 34px;
  margin: 0;
  padding: 0 10px;
  font-size: 13px;
}
.correction-form button:disabled,
.full-correction button:disabled {
  background: #94a3b8;
  cursor: wait;
}
.correction-status {
  grid-column: 1 / -1;
  min-height: 16px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}
.error {
  border-color: #fecaca;
  background: #fff7f7;
}
.notice {
  border-color: #bbf7d0;
  background: #f0fdf4;
}
code {
  color: var(--accent-dark);
  font-weight: 700;
}
@media (max-width: 760px) {
  .topbar,
  .workspace { display: block; }
  .badge { display: inline-block; margin-top: 14px; }
  .upload-panel { margin-bottom: 18px; }
  main { width: min(100vw - 20px, 1040px); padding-top: 20px; }
  h1 { font-size: 28px; }
}
"""


PAGE_SCRIPT = """
<script>
(() => {
  const forms = document.querySelectorAll("[data-correction-form]");
  forms.forEach((form) => {
    const input = form.querySelector('input[name="corrected_label"]');
    const button = form.querySelector('button[type="submit"]');
    const status = form.querySelector("[data-correction-status]");
    if (!input || !button || !status) {
      return;
    }
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const value = input.value.trim();
      if (!value) {
        status.textContent = "Type the right character first.";
        input.focus();
        return;
      }
      button.disabled = true;
      status.textContent = "Saving...";
      try {
        const body = new URLSearchParams(new FormData(form)).toString();
        const response = await fetch(form.action, {
          method: "POST",
          headers: {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
          body,
        });
        if (!response.ok) {
          throw new Error("Correction save failed.");
        }
        input.dataset.savedValue = value;
        status.textContent = `Saved "${value}". You can edit it again.`;
        input.select();
      } catch (error) {
        status.textContent = "Could not save. Try again.";
      } finally {
        button.disabled = false;
      }
    });
  });
})();
</script>
"""


class MnistWebHandler(BaseHTTPRequestHandler):
    """HTTP handler that owns loaded model state and request routing."""

    model = None
    device = None
    labels = None
    letter_model = None
    letter_labels = None
    alnum_model = None
    alnum_labels = None
    recognizer_kind = "digits"

    def log_message(self, format: str, *args: object) -> None:
        """Keep default server logging but route it through stdout."""

        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        """Serve the upload page, health check, or a 404."""

        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_page())
            return
        if parsed.path == "/health":
            self._send_json({"ok": True, "model_loaded": self.model is not None, "recognizer": self.recognizer_kind})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Accept prediction uploads or correction submissions."""

        parsed = urlparse(self.path)
        if parsed.path == "/correct":
            self._handle_correction_post()
            return
        if parsed.path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_html(render_page(error="Upload request is malformed."), HTTPStatus.BAD_REQUEST)
            return

        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._send_html(render_page(error="Upload one or more image files under 8 MB total."), HTTPStatus.BAD_REQUEST)
            return

        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)
        try:
            files = parse_multipart_files(content_type, body)
            results = classify_files(files, self.model, self.device)
            self._send_html(render_page(results=results))
        except ValueError as exc:
            self._send_html(render_page(error=str(exc)), HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            print(f"Prediction failed: {exc!r}")
            self._send_html(
                render_page(error="Prediction failed. Check that the upload is a valid handwriting image and try again."),
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_correction_post(self) -> None:
        """Persist one user correction from a result card."""

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_html(render_page(error="Correction request is malformed."), HTTPStatus.BAD_REQUEST)
            return
        if length <= 0 or length > 16_384:
            self._send_html(render_page(error="Correction request is malformed."), HTTPStatus.BAD_REQUEST)
            return
        body = self.rfile.read(length)
        try:
            form = parse_correction_form(body)
            record = build_correction_record(form)
            save_correction(record)
        except ValueError as exc:
            self._send_html(render_page(error=str(exc)), HTTPStatus.BAD_REQUEST)
            return
        except OSError as exc:
            print(f"Correction save failed: {exc!r}")
            self._send_html(render_page(error="Could not save that correction."), HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_html(render_page(notice="Correction saved. Thanks, this can be used for retraining."))

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Send a UTF-8 HTML response."""

        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, object]) -> None:
        """Send a compact JSON response."""

        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_multipart_files(content_type: str, body: bytes) -> list[tuple[str, bytes]]:
    """Extract uploaded image files from a multipart/form-data request."""

    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("Use the upload form to send image files.")
    message_bytes = (
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=policy.default).parsebytes(message_bytes)
    files: list[tuple[str, bytes]] = []
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        filename = part.get_filename()
        if not filename:
            continue
        payload = part.get_payload(decode=True) or b""
        if payload:
            files.append((Path(filename).name, payload))
    if not files:
        raise ValueError("Choose at least one PNG, JPG, or WEBP image.")
    if len(files) > MAX_FILES:
        raise ValueError(f"Upload {MAX_FILES} or fewer files at a time.")
    return files


def classify_files(files: list[tuple[str, bytes]], model, device) -> list[dict[str, object]]:
    """Decode images, run the active recognizer, and package render data."""

    results: list[dict[str, object]] = []
    for filename, payload in files:
        try:
            image = Image.open(io.BytesIO(payload))
            if image.width * image.height > MAX_IMAGE_PIXELS:
                results.append({"filename": filename, "error": "Image is too large. Use an image under 4 megapixels."})
                continue
            image.load()
        except (UnidentifiedImageError, OSError, ValueError):
            results.append({"filename": filename, "error": "Could not read this as an image."})
            continue

        image = ImageOps.exif_transpose(image).convert("RGB")
        if MnistWebHandler.recognizer_kind == "characters" and MnistWebHandler.labels is not None:
            predictions = predict_characters(
                model,
                MnistWebHandler.labels,
                image,
                device,
                letter_model=MnistWebHandler.letter_model,
                letter_labels=MnistWebHandler.letter_labels,
                alnum_model=MnistWebHandler.alnum_model,
                alnum_labels=MnistWebHandler.alnum_labels,
            )
        else:
            predictions = predict_digits(model, image, device)
        if not predictions:
            results.append(
                {
                    "filename": filename,
                    "error": "No handwriting-like marks were detected.",
                    "preview": image_to_data_url(image),
                    "image_width": image.width,
                    "image_height": image.height,
                }
            )
            continue

        raw_sequence = "".join(prediction_value(item) for item in predictions)
        raw_row_sequences = build_row_sequences(predictions)
        context = cleanup_context(raw_sequence, raw_row_sequences)
        results.append(
            {
                "filename": filename,
                "sequence": context.display,
                "row_sequences": context.rows,
                "context_notes": context.notes,
                "predictions": predictions,
                "preview": image_to_data_url(image),
                "image_width": image.width,
                "image_height": image.height,
            }
        )
    return results


def build_row_sequences(predictions: list[dict[str, object]]) -> list[str]:
    """Build one predicted text string per detected row."""

    rows: dict[int, list[dict[str, object]]] = {}
    for prediction in predictions:
        row = int(prediction.get("row", 1))
        rows.setdefault(row, []).append(prediction)
    return [
        "".join(prediction_value(item) for item in sorted(items, key=lambda item: float(item.get("x", 0))))
        for _, items in sorted(rows.items())
    ]


def prediction_value(prediction: dict[str, object]) -> str:
    """Read either a character label or legacy digit value from a prediction."""

    return str(prediction.get("label", prediction.get("digit", "")))


def top_guesses(prediction: dict[str, object]) -> list[dict[str, object]]:
    """Return up to three model alternatives sorted by confidence."""

    alternatives = prediction.get("alternatives", [])
    if not isinstance(alternatives, list):
        return []
    guesses = [item for item in alternatives if isinstance(item, dict) and item.get("label", "") != ""]
    return sorted(guesses, key=lambda item: float(item.get("confidence", 0)), reverse=True)[:TOP_GUESS_LIMIT]


def is_prediction_uncertain(prediction: dict[str, object]) -> bool:
    """Flag predictions whose confidence or alternatives make the result shaky."""

    confidence = float(prediction.get("confidence", 0))
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return True
    guesses = top_guesses(prediction)
    if not guesses:
        return False
    displayed = prediction_value(prediction)
    top_confidence = float(guesses[0].get("confidence", 0))
    if str(guesses[0].get("label", "")) != displayed and top_confidence >= confidence:
        return True
    if len(guesses) < 2:
        return False
    second_confidence = float(guesses[1].get("confidence", 0))
    return top_confidence - second_confidence <= CLOSE_GUESS_MARGIN


def image_to_data_url(image: Image.Image) -> str:
    """Convert a preview image into an inline browser-safe data URL."""

    display_image = image.copy()
    display_image.thumbnail((1200, 900), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    display_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_correction_form(body: bytes) -> dict[str, str]:
    """Parse a URL-encoded correction form into single string values."""

    fields = parse_qs(body.decode("utf-8"), keep_blank_values=True, strict_parsing=True)
    return {key: values[-1] if values else "" for key, values in fields.items()}


def build_correction_record(form: dict[str, str]) -> dict[str, object]:
    """Validate form fields and shape them for the correction JSONL log."""

    correction_kind = form.get("correction_kind", "character").strip() or "character"
    corrected_label = form.get("corrected_label", "").strip()
    if not corrected_label:
        raise ValueError("Type the correct character before saving.")
    max_length = 255 if correction_kind == "sequence" else 16
    if len(corrected_label) > max_length:
        raise ValueError("Correction is too long.")
    try:
        prediction_index = int(form.get("prediction_index", "0"))
        confidence = float(form.get("confidence", "0"))
        bbox = json.loads(form.get("bbox", "{}"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Correction request is malformed.") from exc
    if correction_kind not in {"character", "sequence"} or not isinstance(bbox, dict):
        raise ValueError("Correction request is malformed.")
    if correction_kind == "character" and prediction_index < 1:
        raise ValueError("Correction request is malformed.")
    return {
        "correction_kind": correction_kind,
        "filename": form.get("filename", "")[:255],
        "sequence": form.get("sequence", "")[:255],
        "prediction_index": prediction_index,
        "original_label": form.get("original_label", "")[:16],
        "corrected_label": corrected_label,
        "confidence": confidence,
        "bbox": {
            "x": float(bbox.get("x", 0)),
            "y": float(bbox.get("y", 0)),
            "width": float(bbox.get("width", 0)),
            "height": float(bbox.get("height", 0)),
            "row": int(bbox.get("row", 1)),
        },
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    }


def save_correction(record: dict[str, object], path: Path = CORRECTIONS_PATH) -> None:
    """Append one validated correction record to the training feedback log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def render_page(
    results: list[dict[str, object]] | None = None,
    error: str | None = None,
    notice: str | None = None,
) -> str:
    """Render the complete upload/results page."""

    metrics = read_metrics()
    metrics_text = "Model not trained yet"
    if metrics:
        best = max(metrics, key=lambda item: item.get("test_accuracy", 0))
        metrics_text = f"Best test accuracy: {best['test_accuracy']:.2f}%"
    if ALNUM_WEIGHTS_PATH.exists():
        alnum_metrics = read_metrics(ALNUM_METRICS_PATH)
        if isinstance(alnum_metrics, dict):
            history = alnum_metrics.get("history", [])
            if history:
                best = max(history, key=lambda item: item.get("test_accuracy", 0))
                metrics_text = (
                    f"Combined test accuracy: {best['test_accuracy']:.2f}% "
                    f"(digits {best.get('digit_test_accuracy', 0):.2f}%, "
                    f"letters {best.get('letter_test_accuracy', 0):.2f}%)"
                )
    if MIXEDCASE_WEIGHTS_PATH.exists():
        mixedcase_metrics = read_metrics(MIXEDCASE_METRICS_PATH)
        if isinstance(mixedcase_metrics, dict):
            history = mixedcase_metrics.get("history", [])
            if history:
                best = max(history, key=lambda item: item.get("test_accuracy", 0))
                metrics_text = (
                    f"Mixed-case test accuracy: {best['test_accuracy']:.2f}% "
                    f"(digits {best.get('digit_test_accuracy', 0):.2f}%, "
                    f"upper {best.get('upper_test_accuracy', 0):.2f}%, "
                    f"lower {best.get('lower_test_accuracy', 0):.2f}%)"
                )
    if LETTER_WEIGHTS_PATH.exists():
        letter_metrics = read_metrics(LETTER_METRICS_PATH)
        if isinstance(letter_metrics, dict):
            history = letter_metrics.get("history", [])
            if history:
                best = max(history, key=lambda item: item.get("test_accuracy", 0))
                if not ALNUM_WEIGHTS_PATH.exists():
                    metrics_text = f"Alphabet test accuracy: {best['test_accuracy']:.2f}%"
    if CHARACTER_WEIGHTS_PATH.exists():
        character_metrics = read_metrics(CHARACTER_METRICS_PATH)
        if character_metrics and not LETTER_WEIGHTS_PATH.exists():
            best = max(character_metrics, key=lambda item: item.get("validation_accuracy", 0))
            metrics_text = f"Character validation accuracy: {best['validation_accuracy']:.2f}%"
        elif character_metrics:
            metrics_text = f"{metrics_text} + punctuation"

    result_html = ""
    if error:
        result_html = f'<section class="empty-panel error"><p>{html.escape(error)}</p></section>'
    elif notice:
        result_html = f'<section class="empty-panel notice"><p>{html.escape(notice)}</p></section>'
    elif results:
        result_html = "\n".join(render_result(item) for item in results)
    else:
        result_html = (
            '<section class="empty-panel">'
            "<p>Upload image files containing handwritten characters. A single image can contain multiple separated letters, numbers, or punctuation, "
            "and the app will read them in row order.</p>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Handwriting Recognizer</title>
  <style>{PAGE_CSS}</style>
</head>
<body>
  <main>
    <div class="topbar">
      <div>
        <h1>Handwriting Recognizer</h1>
        <p>PyTorch CNN predictions for uploaded handwritten characters.</p>
      </div>
      <div class="badge">{html.escape(metrics_text)}</div>
    </div>
    <div class="workspace">
      <section class="upload-panel">
        <form action="/predict" method="post" enctype="multipart/form-data">
          <div class="upload-zone">
            <div>
              <label for="images"><strong>Choose handwriting images</strong></label>
              <p id="upload-help" class="hint">PNG, JPG, or WEBP. Multiple files are okay.</p>
              <input id="images" name="images" type="file" accept="image/png,image/jpeg,image/webp" aria-describedby="upload-help" multiple required>
            </div>
          </div>
          <button type="submit">Recognize handwriting</button>
        </form>
        <p class="hint">The expanded recognizer reads digits, English letters, and common punctuation when <code>character_cnn.pt</code> is trained.</p>
      </section>
      <section aria-live="polite">
        {result_html}
      </section>
    </div>
  </main>
  {PAGE_SCRIPT}
</body>
</html>"""


def render_result(result: dict[str, object]) -> str:
    """Render one uploaded file's prediction panel."""

    filename = html.escape(str(result["filename"]))
    if "error" in result:
        overlay_html = render_overlays(result, [])
        return (
            f'<article class="result-panel error"><div class="filename">{filename}</div>'
            f'{overlay_html}<p>{html.escape(str(result["error"]))}</p></article>'
        )

    predictions = result.get("predictions", [])
    digit_cards = []
    for index, prediction in enumerate(predictions, start=1):
        if not isinstance(prediction, dict):
            continue
        digit = html.escape(prediction_value(prediction))
        confidence = float(prediction["confidence"]) * 100
        uncertain = is_prediction_uncertain(prediction)
        alternatives_html = ""
        guesses = top_guesses(prediction)
        if guesses:
            items = []
            for alternative in guesses:
                label = html.escape(str(alternative.get("label", "")))
                alt_confidence = 100.0 * float(alternative.get("confidence", 0))
                items.append(f"<b>{label}</b> {alt_confidence:.1f}%")
            if items:
                alternatives_html = f'<div class="alternatives">top guesses: {" / ".join(items)}</div>'
        uncertain_html = '<span class="uncertain-note">uncertain</span>' if uncertain else ""
        card_class = "digit uncertain" if uncertain else "digit"
        correction_html = render_correction_form(result, prediction, index)
        digit_cards.append(
            f'<div class="{card_class}"><strong><span class="digit-index">#{index}</span> {digit}</strong>'
            f'<span>confidence {confidence:.1f}%</span>{uncertain_html}{alternatives_html}{correction_html}</div>'
    )
    overlay_html = render_overlays(result, predictions)
    row_html = render_row_sequences(result.get("row_sequences", []))
    context_html = render_context_notes(result.get("context_notes", []))
    full_correction_html = render_full_correction_form(result)
    return f"""
<article class="result-panel">
  <div class="result-head">
    <div class="filename">{filename}</div>
    <div class="sequence">{html.escape(str(result.get("sequence", "")))}</div>
  </div>
  {row_html}
  {context_html}
  {full_correction_html}
  {overlay_html}
  <div class="digits">{''.join(digit_cards)}</div>
</article>"""


def render_full_correction_form(result: dict[str, object]) -> str:
    """Render one whole-result correction field for fixing every character."""

    sequence = str(result.get("sequence", ""))
    hidden_fields = {
        "correction_kind": "sequence",
        "filename": result.get("filename", ""),
        "sequence": sequence,
        "prediction_index": 0,
        "original_label": sequence,
        "confidence": 0,
        "bbox": "{}",
    }
    inputs = "".join(
        f'<input type="hidden" name="{html.escape(str(name), quote=True)}" '
        f'value="{html.escape(str(value), quote=True)}">'
        for name, value in hidden_fields.items()
    )
    label_seed = json.dumps(hidden_fields, ensure_ascii=True, sort_keys=True)
    label_token = hashlib.sha1(label_seed.encode("utf-8")).hexdigest()[:10]
    label_id = f"sequence-correction-{label_token}"
    return (
        '<form class="full-correction" action="/correct" method="post" data-correction-form>'
        f"{inputs}"
        f'<label for="{label_id}">Fix the whole result</label>'
        f'<input id="{label_id}" name="corrected_label" type="text" maxlength="255" '
        f'value="{html.escape(sequence, quote=True)}" autocomplete="off">'
        '<button type="submit">Save all</button>'
        '<span class="correction-status" data-correction-status></span>'
        "</form>"
    )


def render_correction_form(result: dict[str, object], prediction: dict[str, object], index: int) -> str:
    """Render a tiny per-character correction form for later retraining."""

    bbox = {
        "x": prediction.get("x", 0),
        "y": prediction.get("y", 0),
        "width": prediction.get("width", 0),
        "height": prediction.get("height", 0),
        "row": prediction.get("row", 1),
    }
    hidden_fields = {
        "correction_kind": "character",
        "filename": result.get("filename", ""),
        "sequence": result.get("sequence", ""),
        "prediction_index": index,
        "original_label": prediction_value(prediction),
        "confidence": prediction.get("confidence", 0),
        "bbox": json.dumps(bbox, separators=(",", ":")),
    }
    inputs = "".join(
        f'<input type="hidden" name="{html.escape(str(name), quote=True)}" '
        f'value="{html.escape(str(value), quote=True)}">'
        for name, value in hidden_fields.items()
    )
    label_seed = json.dumps(hidden_fields, ensure_ascii=True, sort_keys=True)
    label_token = hashlib.sha1(label_seed.encode("utf-8")).hexdigest()[:10]
    label_id = f"correction-{index}-{label_token}"
    return (
        '<form class="correction-form" action="/correct" method="post" data-correction-form>'
        f"{inputs}"
        f'<label class="sr-only" for="{label_id}">Correct prediction #{index}</label>'
        f'<input id="{label_id}" name="corrected_label" type="text" maxlength="16" '
        f'placeholder="fix #{index}" autocomplete="off">'
        '<button type="submit">Save</button>'
        '<span class="correction-status" data-correction-status></span>'
        "</form>"
    )


def render_row_sequences(row_sequences: object) -> str:
    """Render per-row outputs when the upload contains multiple lines."""

    if not isinstance(row_sequences, list) or len(row_sequences) <= 1:
        return ""
    rows = "".join(
        f"<code>row {index}: {html.escape(str(sequence))}</code>"
        for index, sequence in enumerate(row_sequences, start=1)
    )
    return f'<div class="row-output">{rows}</div>'


def render_context_notes(notes: object) -> str:
    """Render short context-cleanup notes under the prediction rows."""

    if not isinstance(notes, list) or not notes:
        return ""
    items = "".join(f"<code>{html.escape(str(note))}</code>" for note in notes)
    return f'<div class="row-output">{items}</div>'


def render_overlays(result: dict[str, object], predictions: object) -> str:
    """Render absolute-positioned prediction boxes over the preview image."""

    preview = html.escape(str(result.get("preview", "")), quote=True)
    if not preview:
        return ""
    prediction_kind = "character" if MnistWebHandler.recognizer_kind == "characters" else "digit"
    image_width = float(result.get("image_width", 1))
    image_height = float(result.get("image_height", 1))
    boxes = []
    if not isinstance(predictions, list):
        predictions = []
    for index, prediction in enumerate(predictions, start=1):
        if not isinstance(prediction, dict):
            continue
        left = 100.0 * float(prediction.get("x", 0)) / image_width
        top = 100.0 * float(prediction.get("y", 0)) / image_height
        width = 100.0 * float(prediction.get("width", 0)) / image_width
        height = 100.0 * float(prediction.get("height", 0)) / image_height
        digit = html.escape(prediction_value(prediction))
        confidence = 100.0 * float(prediction.get("confidence", 0))
        uncertain = is_prediction_uncertain(prediction)
        box_class = "digit-box uncertain" if uncertain else "digit-box"
        uncertainty_label = ", uncertain" if uncertain else ""
        boxes.append(
            f'<div class="{box_class}" '
            f'tabindex="0" title="Prediction #{index}: {digit}, confidence {confidence:.1f}%" '
            f'aria-label="Prediction {index}: {prediction_kind} {digit}, confidence {confidence:.1f} percent{uncertainty_label}" '
            f'style="left:{left:.3f}%;top:{top:.3f}%;width:{width:.3f}%;height:{height:.3f}%;">'
            f'<span aria-hidden="true">#{index}</span></div>'
        )
    return (
        '<div class="preview-wrap">'
        f'<img src="{preview}" alt="Uploaded handwriting image with prediction boxes">'
        f"{''.join(boxes)}"
        "</div>"
    )


def read_metrics(path=METRICS_PATH):
    """Read a metrics JSON file, returning an empty list on bad/missing data."""

    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def run(host: str = HOST, port: int = PORT) -> None:
    """Load the best available recognizer and start the local HTTP server."""

    if CHARACTER_WEIGHTS_PATH.exists():
        MnistWebHandler.device = get_device()
        MnistWebHandler.model, MnistWebHandler.labels = load_character_model(device=MnistWebHandler.device)
        MnistWebHandler.letter_model, MnistWebHandler.letter_labels = load_letter_model(device=MnistWebHandler.device)
        MnistWebHandler.alnum_model, MnistWebHandler.alnum_labels = load_mixedcase_model(
            device=MnistWebHandler.device
        )
        if MnistWebHandler.alnum_model is None:
            MnistWebHandler.alnum_model, MnistWebHandler.alnum_labels = load_alnum_model(device=MnistWebHandler.device)
        MnistWebHandler.recognizer_kind = "characters"
    elif WEIGHTS_PATH.exists():
        MnistWebHandler.device = get_device()
        MnistWebHandler.model = load_model(device=MnistWebHandler.device)
        MnistWebHandler.labels = None
        MnistWebHandler.letter_model = None
        MnistWebHandler.letter_labels = None
        MnistWebHandler.alnum_model = None
        MnistWebHandler.alnum_labels = None
        MnistWebHandler.recognizer_kind = "digits"
    else:
        raise SystemExit(
            f"Missing model weights. Train first with: python3 character_model.py or python3 mnist_model.py"
        )
    server = ThreadingHTTPServer((host, port), MnistWebHandler)
    print(f"Handwriting Recognizer ({MnistWebHandler.recognizer_kind}) running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
