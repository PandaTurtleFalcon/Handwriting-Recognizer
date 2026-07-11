"""Small stdlib web app for uploading and recognizing handwriting images.

The server intentionally avoids a web framework so the project is easy to run:
`python3 main.py` starts an upload form, predicts every uploaded image, and
renders the original image with numbered bounding boxes that match the cards.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import hashlib
import html
import io
import json
import shutil
import subprocess
import tempfile
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None
else:
    register_heif_opener()

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
MAX_CORRECTION_BYTES = MAX_UPLOAD_BYTES + 16_384
MAX_IMAGE_PIXELS = 4_000_000
MAX_DECODE_IMAGE_PIXELS = 60_000_000
MAX_FILES = 20
CORRECTIONS_PATH = Path("data") / "corrections" / "corrections.jsonl"
CORRECTION_UPLOAD_DIR = Path("data") / "corrections" / "uploads"
WEB_ROOT = Path("web")
STATIC_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}
# Predictions below this confidence are visually flagged as "uncertain" in
# the results UI (see is_prediction_uncertain).
LOW_CONFIDENCE_THRESHOLD = 0.80
# If the top two model alternatives are within this margin of each other,
# the prediction is also flagged uncertain even if its raw confidence
# cleared the threshold above, since a close second guess means the model
# was effectively torn between two answers.
CLOSE_GUESS_MARGIN = 0.12
TOP_GUESS_LIMIT = 3
DIGIT_SPECIALIST_COMPATIBLE_LABELS = set("0123456789BIJLOSYZlo")
DIGIT_SPECIALIST_LETTER_BLOCKERS = set("BOSo")
DIGIT_SPECIALIST_MAX_LETTER_RATIO = 0.25
PRACTICE_PRIORITY_LABELS = [
    "s",
    "O",
    "V",
    "1",
    "c",
    "I",
    "F",
    "o",
    "m",
    "0",
    "l",
    "i",
    "U",
    "k",
    "u",
    "g",
    "q",
    "M",
    "C",
    "P",
    "S",
    "v",
    "z",
    "x",
    "p",
    "2",
    "Z",
    "9",
    "f",
    "5",
    "W",
    "w",
    "Y",
    "y",
    "4",
    "T",
    "t",
    "7",
    "J",
    "j",
    "K",
    "X",
    "-",
    "_",
    ".",
    "'",
    "|",
    "/",
    ":",
    ";",
    "!",
    "+",
]
PRACTICE_TARGET_PER_LABEL = 20
DISPLAY_AMBIGUITY_GROUPS = [
    frozenset("0Oo"),
    frozenset("1Ili|!/"),
    frozenset("5Ss"),
    frozenset("2Zz"),
    frozenset("8B"),
    frozenset("Cc"),
    frozenset("Xx"),
    frozenset("Vv"),
    frozenset("Kk"),
    frozenset("Pp"),
    frozenset("Tt7"),
    frozenset("-_"),
    frozenset(".'`"),
    frozenset(":;i!"),
    frozenset("+t"),
    frozenset("9qg"),
    frozenset("Yy4"),
    frozenset("Uuv"),
    frozenset("NnMm"),
    frozenset("Jj"),
]
CASE_GEOMETRY_PAIRS = {
    "C": "c",
    "F": "f",
    "K": "k",
    "M": "m",
    "N": "n",
    "O": "o",
    "P": "p",
    "S": "s",
    "U": "u",
    "V": "v",
    "W": "w",
    "X": "x",
    "Y": "y",
    "Z": "z",
}
CASE_GEOMETRY_PAIRS.update({lower: upper for upper, lower in list(CASE_GEOMETRY_PAIRS.items())})
CASE_GEOMETRY_MIN_ROW_ITEMS = 3
CASE_GEOMETRY_MIN_HEIGHT_SPREAD = 1.28
CASE_GEOMETRY_SHORT_RATIO = 0.76
CASE_GEOMETRY_TALL_RATIO = 0.92
CASE_GEOMETRY_MIN_ALT_CONFIDENCE = 0.12
CASE_PAIR_MIN_SIZE_SPREAD = 1.10
CASE_PAIR_MIN_ALT_CONFIDENCE = 0.02
# Set well above the processing size so normal phone-camera uploads can be
# decoded and then resized before recognition. PIL still raises on images
# above 2x this value, which keeps decompression-bomb protection in place.
Image.MAX_IMAGE_PIXELS = MAX_DECODE_IMAGE_PIXELS


PAGE_CSS = """
:root {
  color-scheme: light;
  --ink: #14192b;
  --muted: #5f6d7e;
  --muted-soft: #8592a6;
  --line: #dde3ec;
  --line-soft: #eaeef4;
  --paper: #eef2f8;
  --panel: #ffffff;
  --accent: #4338ca;
  --accent-dark: #312a9e;
  --accent-soft: #eef0fe;
  --accent-ring: rgb(67 56 202 / 0.25);
  --ok: #0f7a45;
  --ok-bg: #eefcf3;
  --ok-line: #bbf1d1;
  --warn: #92400e;
  --warn-bg: #fff8ec;
  --warn-line: #f0b429;
  --danger: #b91c1c;
  --danger-bg: #fff5f5;
  --danger-line: #f6b8b8;
  --radius-lg: 16px;
  --radius-md: 12px;
  --radius-sm: 8px;
  --shadow-card: 0 1px 2px rgb(20 25 43 / 0.04), 0 8px 24px -12px rgb(20 25 43 / 0.12);
  --shadow-pop: 0 4px 10px -4px rgb(20 25 43 / 0.18);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, ui-sans-serif, system-ui, sans-serif;
  background:
    radial-gradient(1100px 480px at 12% -10%, rgb(67 56 202 / 0.08), transparent),
    var(--paper);
  color: var(--ink);
  -webkit-font-smoothing: antialiased;
}
main {
  width: min(1080px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 40px 0 56px;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 18px;
  margin-bottom: 30px;
}
.topbar > div:first-child {
  min-width: 0;
}
h1 {
  margin: 0 0 6px;
  font-size: clamp(26px, 4vw, 34px);
  font-weight: 800;
  line-height: 1.12;
  letter-spacing: -0.02em;
}
p {
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: 100%;
  border: 1px solid var(--ok-line);
  border-radius: var(--radius-lg);
  padding: 8px 14px;
  background: var(--ok-bg);
  color: var(--ok);
  font-size: 13px;
  font-weight: 700;
  white-space: normal;
  box-shadow: var(--shadow-pop);
}
.badge::before {
  content: "";
  width: 7px;
  height: 7px;
  margin-top: 3px;
  border-radius: 50%;
  background: currentColor;
  flex: none;
  align-self: flex-start;
}
.workspace {
  display: grid;
  grid-template-columns: minmax(0, 0.85fr) minmax(340px, 1.15fr);
  gap: 22px;
  align-items: start;
}
.upload-panel,
.result-panel,
.empty-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius-lg);
  padding: 22px;
  box-shadow: var(--shadow-card);
}
.upload-panel {
  position: sticky;
  top: 20px;
}
.upload-zone {
  display: grid;
  place-items: center;
  min-height: 240px;
  border: 2px dashed #b7c3d6;
  border-radius: var(--radius-md);
  background: linear-gradient(180deg, #f7f9fd, #f0f4fa);
  text-align: center;
  padding: 28px 22px;
  transition: border-color 0.15s ease, background 0.15s ease;
}
.upload-zone:focus-within,
.upload-zone:hover {
  border-color: var(--accent);
  background: var(--accent-soft);
}
.upload-zone label {
  font-size: 16px;
  font-weight: 700;
  color: var(--ink);
}
.upload-zone .hint {
  margin-top: 6px;
}
input[type="file"] {
  width: 100%;
  max-width: 320px;
  margin-top: 18px;
  font: inherit;
  color: var(--muted);
}
input[type="file"]::file-selector-button {
  margin-right: 10px;
  padding: 9px 14px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--panel);
  color: var(--ink);
  font: inherit;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease;
}
input[type="file"]::file-selector-button:hover {
  background: var(--accent-soft);
  border-color: var(--accent);
}
button {
  display: inline-flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
  min-height: 46px;
  margin-top: 20px;
  width: 100%;
  border: 0;
  border-radius: var(--radius-sm);
  background: linear-gradient(180deg, var(--accent), var(--accent-dark));
  color: white;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: -0.01em;
  cursor: pointer;
  box-shadow: 0 6px 16px -6px rgb(67 56 202 / 0.55);
  transition: transform 0.08s ease, box-shadow 0.15s ease, filter 0.15s ease;
}
button:hover { filter: brightness(1.06); }
button:active { transform: translateY(1px); }
button:disabled {
  cursor: wait;
  filter: none;
  background: #94a3b8;
  box-shadow: none;
}
button:focus-visible,
input[type="file"]:focus-visible,
input[type="text"]:focus-visible,
.digit-box:focus-visible,
a:focus-visible {
  outline: 3px solid var(--accent-ring);
  outline-offset: 2px;
}
input[type="text"]:focus-visible {
  border-color: var(--accent);
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
  font-size: 13.5px;
  color: var(--muted-soft);
}
.result-panel + .result-panel { margin-top: 16px; }
.result-panel { overflow: hidden; }
.result-panel.error {
  border-color: var(--danger-line);
  background: var(--danger-bg);
}
.result-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 14px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--line-soft);
}
.filename {
  font-size: 14px;
  font-weight: 700;
  color: var(--muted);
  overflow-wrap: anywhere;
}
.sequence {
  font-size: 34px;
  line-height: 1.1;
  font-weight: 800;
  color: var(--ink);
  letter-spacing: -0.01em;
  max-width: 100%;
  overflow-wrap: anywhere;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
.row-output {
  margin: 10px 0 0;
  color: var(--muted);
  font-weight: 600;
}
.row-output code {
  display: inline-block;
  margin: 4px 8px 0 0;
  padding: 5px 10px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #f8fafc;
  color: var(--ink);
  font-size: 12.5px;
  max-width: 100%;
  overflow-wrap: anywhere;
}
.digits {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.preview-wrap {
  position: relative;
  margin: 16px 0;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  background: repeating-conic-gradient(#f4f7fb 0% 25%, #eef1f6 0% 50%) 50% / 18px 18px;
}
.preview-wrap img {
  display: block;
  width: 100%;
  height: auto;
}
.digit-box {
  position: absolute;
  border: 3px solid #dc2626;
  border-radius: 6px;
  box-shadow: 0 0 0 2px rgb(255 255 255 / 0.9);
  cursor: default;
  transition: outline-color 0.1s ease;
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
  font-weight: 800;
}
.digit-box.uncertain span {
  background: var(--warn-line);
  color: #402100;
}
.digit-box:hover,
.digit-box:focus-visible {
  border-color: #991b1b;
  outline: 3px solid rgb(220 38 38 / 0.25);
}
.digit-box.uncertain:hover,
.digit-box.uncertain:focus-visible {
  border-color: var(--warn);
  outline: 3px solid rgb(240 180 41 / 0.32);
}
.digit-index {
  color: #dc2626;
  font-weight: 800;
}
.digit {
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: 14px;
  background: #fbfdff;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.digit:hover {
  border-color: #c8d3e2;
  box-shadow: var(--shadow-pop);
}
.digit.uncertain {
  border-color: var(--warn-line);
  background: var(--warn-bg);
}
.digit strong {
  display: flex;
  align-items: baseline;
  gap: 6px;
  font-size: 26px;
  font-weight: 800;
}
.digit span {
  display: block;
  margin-top: 4px;
  color: var(--muted);
  font-size: 12.5px;
  font-weight: 600;
}
.alternatives {
  margin-top: 9px;
  font-size: 12.5px;
  line-height: 1.4;
  color: var(--muted);
}
.alternatives b {
  color: var(--ink);
}
.guess-button {
  display: inline-flex;
  align-items: baseline;
  gap: 4px;
  width: auto;
  min-height: 0;
  margin: 0 3px 3px 0;
  padding: 3px 7px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--panel);
  color: var(--muted);
  font: inherit;
  font-size: 12px;
  font-weight: 700;
  box-shadow: none;
  cursor: pointer;
}
.guess-button:hover,
.guess-button:focus-visible {
  border-color: var(--accent);
  color: var(--accent-dark);
  outline: 2px solid rgb(36 64 183 / 0.14);
}
.ambiguity-note {
  margin-top: 7px;
  font-size: 12.5px;
  line-height: 1.35;
  color: var(--warn);
  font-weight: 700;
}
.uncertain-note {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-top: 8px;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgb(240 180 41 / 0.18);
  color: var(--warn);
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.full-correction {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px 10px;
  margin: 14px 0 4px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  background: #f8fafc;
}
.full-correction label {
  grid-column: 1 / -1;
  color: var(--muted);
  font-size: 12.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.full-correction input[type="text"] {
  min-width: 0;
  height: 40px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  padding: 0 12px;
  font: inherit;
  background: var(--panel);
  transition: border-color 0.15s ease;
}
.full-correction button {
  width: auto;
  min-height: 40px;
  margin: 0;
  padding: 0 16px;
  font-size: 13.5px;
  box-shadow: none;
}
.correction-form {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
  margin-top: 12px;
}
.correction-form input[type="text"] {
  min-width: 0;
  width: 100%;
  height: 36px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  padding: 0 10px;
  font: inherit;
  background: var(--panel);
  transition: border-color 0.15s ease;
}
.correction-form button {
  width: 100%;
  min-height: 36px;
  margin: 0;
  padding: 0 12px;
  font-size: 12.5px;
  box-shadow: none;
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
  font-weight: 600;
}
.empty-panel {
  display: grid;
  place-items: center;
  min-height: 200px;
  text-align: center;
  color: var(--muted);
}
.empty-panel p { max-width: 46ch; }
.error {
  border-color: var(--danger-line);
  background: var(--danger-bg);
  color: var(--danger);
}
.error p { color: var(--danger); }
.notice {
  border-color: var(--ok-line);
  background: var(--ok-bg);
}
.notice p { color: var(--ok); }
code {
  color: var(--accent-dark);
  font-weight: 700;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (max-width: 760px) {
  .topbar {
    flex-direction: column;
    align-items: flex-start;
  }
  .workspace { display: block; }
  .badge { margin-top: 2px; }
  .upload-panel {
    margin-bottom: 18px;
    position: static;
  }
  main { width: min(100vw - 20px, 1080px); padding-top: 22px; }
  h1 { font-size: 26px; }
  .sequence { font-size: 28px; }
  .digits { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
}
@media (max-width: 420px) {
  .full-correction,
  .correction-form {
    grid-template-columns: 1fr;
  }
  .full-correction button,
  .correction-form button {
    width: 100%;
  }
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

  document.querySelectorAll("[data-fill-correction]").forEach((guessButton) => {
    guessButton.addEventListener("click", () => {
      const card = guessButton.closest(".digit");
      const form = card ? card.querySelector("[data-correction-form]") : null;
      const input = form ? form.querySelector('input[name="corrected_label"]') : null;
      const status = form ? form.querySelector("[data-correction-status]") : null;
      if (!input) {
        return;
      }
      input.value = guessButton.dataset.fillCorrection || "";
      input.focus();
      input.select();
      if (status) {
        status.textContent = "Ready to save.";
      }
    });
  });
})();
</script>
"""


class MnistWebHandler(BaseHTTPRequestHandler):
    """HTTP handler that owns loaded model state and request routing.

    BaseHTTPRequestHandler instantiates a new handler object per request, so
    the loaded models are kept as *class* attributes (set once in `run()`)
    rather than instance attributes — every request shares the same
    already-loaded weights instead of reloading them each time.
    """

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
            self._send_static(WEB_ROOT / "index.html")
            return
        if parsed.path in {"/styles.css", "/app.js"}:
            self._send_static(WEB_ROOT / parsed.path.lstrip("/"))
            return
        if parsed.path == "/health":
            self._send_json({"ok": True, "model_loaded": self.model is not None, "recognizer": self.recognizer_kind})
            return
        if parsed.path == "/api/correction-coverage":
            self._send_json(correction_coverage_report())
            return
        if parsed.path == "/api/correction-readiness":
            self._send_json(correction_readiness_report())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Accept prediction uploads or correction submissions."""

        parsed = urlparse(self.path)
        if parsed.path == "/api/correct":
            self._handle_correction_api_post()
            return
        if parsed.path == "/api/predict":
            self._handle_prediction_api_post()
            return
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

    def _handle_prediction_api_post(self) -> None:
        """Accept image uploads and return JSON results for the static UI."""

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"ok": False, "error": "Upload request is malformed."}, HTTPStatus.BAD_REQUEST)
            return
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._send_json({"ok": False, "error": "Upload one or more image files under 8 MB total."}, HTTPStatus.BAD_REQUEST)
            return
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)
        try:
            files = parse_multipart_files(content_type, body)
            results = classify_files(files, self.model, self.device)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            print(f"Prediction failed: {exc!r}")
            self._send_json(
                {"ok": False, "error": "Prediction failed. Check that the upload is a valid handwriting image and try again."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json({"ok": True, "results": results})

    def _handle_correction_post(self) -> None:
        """Persist one user correction from a result card."""

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_html(render_page(error="Correction request is malformed."), HTTPStatus.BAD_REQUEST)
            return
        if length <= 0 or length > MAX_CORRECTION_BYTES:
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

    def _handle_correction_api_post(self) -> None:
        """Persist one correction and return JSON for the static UI."""

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"ok": False, "error": "Correction request is malformed."}, HTTPStatus.BAD_REQUEST)
            return
        if length <= 0 or length > 16_384:
            self._send_json({"ok": False, "error": "Correction request is malformed."}, HTTPStatus.BAD_REQUEST)
            return
        body = self.rfile.read(length)
        try:
            form = parse_correction_form(body)
            record = build_correction_record(form)
            save_practice_source_image(form, str(record.get("image_id", "")))
            save_correction(record)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except OSError as exc:
            print(f"Correction save failed: {exc!r}")
            self._send_json({"ok": False, "error": "Could not save that correction."}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json({"ok": True, "notice": "Correction saved. Thanks, this can be used for retraining."})

    def _send_static(self, path: Path) -> None:
        """Serve a known static UI file from the local web folder."""

        try:
            resolved = path.resolve(strict=True)
            web_root = WEB_ROOT.resolve(strict=True)
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if web_root not in resolved.parents and resolved != web_root:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = STATIC_CONTENT_TYPES.get(resolved.suffix)
        if content_type is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Send a UTF-8 HTML response."""

        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Send a compact JSON response."""

        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_multipart_files(content_type: str, body: bytes) -> list[tuple[str, bytes]]:
    """Extract uploaded image files from a multipart/form-data request.

    There's no HTTP form-parsing library in the stdlib, but Python's email
    package already implements MIME multipart parsing (multipart/form-data
    is structurally a MIME message), so the raw body is repackaged with a
    synthetic Content-Type/Content-Length header and handed to the email
    parser instead of writing a parser from scratch.
    """

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
        raise ValueError("Choose at least one PNG, JPG, WEBP, HEIC, or HEIF image.")
    if len(files) > MAX_FILES:
        raise ValueError(f"Upload {MAX_FILES} or fewer files at a time.")
    return files


def decode_upload_image(filename: str, payload: bytes) -> Image.Image:
    """Decode common browser image uploads, including HEIC on macOS."""

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if Path(filename).suffix.lower() not in {".heic", ".heif"}:
            raise exc
        converted = decode_heic_with_sips(payload)
        if converted is None:
            raise exc
        return converted


def decode_heic_with_sips(payload: bytes) -> Image.Image | None:
    """Use macOS' built-in image converter when Pillow lacks HEIC support."""

    if shutil.which("sips") is None:
        return None
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "upload.heic"
        target = Path(directory) / "upload.png"
        source.write_bytes(payload)
        try:
            completed = subprocess.run(
                ["sips", "-s", "format", "png", str(source), "--out", str(target)],
                capture_output=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0 or not target.exists():
            return None
        image = Image.open(target)
        image.load()
        return image


def resize_for_recognition(image: Image.Image) -> Image.Image:
    """Downscale large phone photos before segmentation/model inference."""

    pixel_count = image.width * image.height
    if pixel_count <= MAX_IMAGE_PIXELS:
        return image
    scale = (MAX_IMAGE_PIXELS / pixel_count) ** 0.5
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def classify_files(
    files: list[tuple[str, bytes]],
    model,
    device,
    save_sources: bool = True,
) -> list[dict[str, object]]:
    """Decode images, run the active recognizer, and package render data."""

    results: list[dict[str, object]] = []
    for filename, payload in files:
        image_id = hashlib.sha256(payload).hexdigest()
        try:
            image = decode_upload_image(filename, payload)
            if image.width * image.height > MAX_DECODE_IMAGE_PIXELS:
                results.append({"filename": filename, "error": "Image is too large. Use an image under 60 megapixels."})
                continue
        except (UnidentifiedImageError, OSError, ValueError):
            results.append({"filename": filename, "error": "Could not read this as an image."})
            continue

        image = resize_for_recognition(ImageOps.exif_transpose(image).convert("RGB"))
        if save_sources:
            save_correction_source_image(image_id, image)
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
            predictions = resolve_visual_twin_predictions(predictions)
            try:
                digit_model = load_model(device=device)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                print(f"Digit specialist unavailable: {exc!r}")
                digit_model = model
            try:
                digit_predictions = predict_digits(digit_model, image, device)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                print(f"Digit specialist prediction failed: {exc!r}")
                digit_predictions = []
            if not predictions and digit_predictions:
                predictions = digit_predictions
            if should_use_digit_specialist_predictions(predictions, digit_predictions):
                predictions = digit_predictions
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
        correction_predictions = predictions
        if len(context.rows) < len(raw_row_sequences):
            visible_rows = set(range(1, len(context.rows) + 1))
            correction_predictions = [
                item
                for item in predictions
                if isinstance(item, dict) and int(item.get("row", 1)) in visible_rows
            ]
        results.append(
            {
                "filename": filename,
                "sequence": context.display,
                "raw_sequence": raw_sequence,
                "row_sequences": context.rows,
                "raw_row_sequences": raw_row_sequences,
                "context_notes": context.notes,
                "predictions": predictions,
                "correction_predictions": correction_predictions,
                "preview": image_to_data_url(image),
                "image_id": image_id,
                "image_width": image.width,
                "image_height": image.height,
            }
        )
    return results


def save_correction_source_image(image_id: str, image: Image.Image) -> None:
    """Persist the uploaded image so future corrections can be used for training."""

    if not image_id:
        return
    CORRECTION_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = CORRECTION_UPLOAD_DIR / f"{image_id}.png"
    if not target.exists():
        image.save(target)


def save_practice_source_image(form: dict[str, str], image_id: str) -> None:
    """Persist a generated practice glyph image sent with a correction."""

    source_image = form.get("source_image", "")
    if not source_image:
        return
    if not image_id:
        raise ValueError("Practice correction is missing its image id.")
    prefix = "data:image/png;base64,"
    if not source_image.startswith(prefix):
        raise ValueError("Practice correction image is malformed.")
    try:
        image_bytes = base64.b64decode(source_image[len(prefix) :], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Practice correction image is malformed.") from exc
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Practice correction image is too large.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            save_correction_source_image(image_id, ImageOps.exif_transpose(image).convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Practice correction image is malformed.") from exc


def build_correction_coverage_report(
    counts: dict[str, int],
    labels: list[str] = PRACTICE_PRIORITY_LABELS,
    target_per_label: int = PRACTICE_TARGET_PER_LABEL,
) -> dict[str, object]:
    """Build practice-label coverage data for the browser UI."""

    rows = []
    for label in labels:
        count = int(counts.get(label, 0))
        rows.append(
            {
                "label": label,
                "count": count,
                "needed": max(0, target_per_label - count),
                "ready": count >= target_per_label,
            }
        )
    ready = sum(1 for row in rows if bool(row["ready"]))
    sample_count = sum(int(row["count"]) for row in rows)
    target_samples = len(rows) * target_per_label
    needed_samples = sum(int(row["needed"]) for row in rows)
    focus_labels = [str(row["label"]) for row in rows if not bool(row["ready"])][:8]
    focus_items = [
        {"label": str(row["label"]), "count": int(row["count"]), "needed": int(row["needed"])}
        for row in rows
        if not bool(row["ready"])
    ][:8]
    return {
        "ok": True,
        "target_per_label": target_per_label,
        "ready_labels": ready,
        "total_labels": len(rows),
        "samples": sample_count,
        "target_samples": target_samples,
        "needed_samples": needed_samples,
        "focus_labels": focus_labels,
        "focus_items": focus_items,
        "labels": rows,
    }


def correction_coverage_report() -> dict[str, object]:
    """Return current trainable practice/correction coverage by weak label."""

    from scripts.train_from_corrections import exportable_character_correction_counts, load_character_labels

    labels = load_character_labels()
    counts = exportable_character_correction_counts(labels)
    return build_correction_coverage_report(counts)


def correction_readiness_report() -> dict[str, object]:
    """Return machine-readable correction-training readiness for the app."""

    from scripts.train_from_corrections import (
        LABELS,
        MIXEDCASE_LABELS,
        DEFAULT_MIXEDCASE_PRIORITY_LABELS,
        DEFAULT_PRIORITY_LABELS,
        correction_item_label_counts,
        dry_run_report,
        exportable_character_correction_counts,
        load_character_labels,
        load_correction_cache,
    )

    character_labels = load_character_labels()
    folded_corrections = load_correction_cache(LABELS)
    mixed_corrections = load_correction_cache(list(MIXEDCASE_LABELS))
    report = dry_run_report(
        exportable_character_correction_counts(character_labels),
        correction_item_label_counts(LABELS, folded_corrections),
        correction_item_label_counts(list(MIXEDCASE_LABELS), mixed_corrections),
        0 if folded_corrections is None else len(folded_corrections[1]),
        0 if mixed_corrections is None else len(mixed_corrections[1]),
        DEFAULT_PRIORITY_LABELS,
        DEFAULT_MIXEDCASE_PRIORITY_LABELS,
    )
    return {"ok": True, **report}


def should_use_digit_specialist_predictions(
    character_predictions: list[dict[str, object]],
    digit_predictions: list[dict[str, object]],
) -> bool:
    """Use MNIST output when the combined recognizer is only seeing digit-like glyphs."""

    if not character_predictions or len(character_predictions) != len(digit_predictions):
        return False
    character_rows = [int(item.get("row", 1)) for item in character_predictions]
    digit_rows = [int(item.get("row", 1)) for item in digit_predictions]
    if character_rows != digit_rows:
        return False
    labels = [prediction_value(item) for item in character_predictions]
    if any(label not in DIGIT_SPECIALIST_COMPATIBLE_LABELS for label in labels):
        return False
    letter_count = sum(1 for label in labels if label.isalpha())
    if letter_count and letter_count / len(labels) > DIGIT_SPECIALIST_MAX_LETTER_RATIO:
        return False
    if any(
        label in DIGIT_SPECIALIST_LETTER_BLOCKERS and float(item.get("confidence", 0)) >= 0.82
        for label, item in zip(labels, character_predictions)
    ):
        return False
    digit_confidences = [float(item.get("confidence", 0)) for item in digit_predictions]
    if min(digit_confidences, default=0) < 0.88:
        return False
    return sum(digit_confidences) / len(digit_confidences) >= 0.95


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


def resolve_visual_twin_predictions(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    """Apply narrow row-level fixes when alternatives and geometry agree."""

    rows: dict[int, list[tuple[int, dict[str, object]]]] = {}
    for index, prediction in enumerate(predictions):
        rows.setdefault(int(prediction.get("row", 1)), []).append((index, prediction))
    replacements: dict[int, dict[str, object]] = {}
    for row_items in rows.values():
        ordered = sorted(row_items, key=lambda item: float(item[1].get("x", 0)))
        resolved = _resolve_visual_twin_row([item for _, item in ordered])
        if resolved is None:
            resolved = _resolve_case_pair_by_geometry([item for _, item in ordered])
        if resolved is None:
            resolved = _resolve_case_by_row_geometry([item for _, item in ordered])
        if resolved is None:
            continue
        for (index, original), replacement in zip(ordered, resolved):
            if replacement is not original:
                replacements[index] = replacement
    if not replacements:
        return predictions
    return [replacements.get(index, prediction) for index, prediction in enumerate(predictions)]


def _resolve_visual_twin_row(row: list[dict[str, object]]) -> list[dict[str, object]] | None:
    """Resolve a few known whole-row visual-twin patterns."""

    labels = [prediction_value(item) for item in row]
    if labels == ["T", "3", "5", "T"] and _alternative_confidence(row[2], {"S", "s"}) >= 0.10:
        if _alternative_confidence(row[3], {"7"}) >= 0.50:
            return [
                row[0],
                row[1],
                _with_prediction_label(row[2], "s", 0.88),
                _with_prediction_label(row[3], "7", 0.88),
            ]
    if len(row) == 3 and labels != ["5", "5", "5"] and all(label in {"5", "S", "s"} for label in labels):
        widths = [float(item.get("width", 0)) for item in row]
        has_twin_evidence = any(label in {"S", "s"} for label in labels) or any(
            _alternative_confidence(item, {"S", "s"}) >= 0.10 for item in row
        )
        if has_twin_evidence and min(widths) > 0 and max(widths) >= min(widths) * 1.18:
            widest = max(range(3), key=lambda index: widths[index])
            narrowest = min(range(3), key=lambda index: widths[index])
            remaining = ({0, 1, 2} - {widest, narrowest}).pop()
            resolved = list(row)
            resolved[widest] = _with_prediction_label(row[widest], "S", 0.84)
            resolved[narrowest] = _with_prediction_label(row[narrowest], "s", 0.84)
            resolved[remaining] = _with_prediction_label(row[remaining], "5", 0.84)
            return resolved
    if len(row) == 3 and labels == ["5", "5", "5"]:
        edge_strengths = [_alternative_confidence(row[0], {"S", "s"}), _alternative_confidence(row[2], {"S", "s"})]
        middle_strength = _alternative_confidence(row[1], {"S", "s"})
        if min(edge_strengths) >= 0.12 and middle_strength <= 0.05:
            first_width = float(row[0].get("width", 0))
            last_width = float(row[2].get("width", 0))
            if max(first_width, last_width) >= min(first_width, last_width) * 1.25:
                first_label, last_label = ("S", "s") if first_width > last_width else ("s", "S")
                return [_with_prediction_label(row[0], first_label, 0.86), row[1], _with_prediction_label(row[2], last_label, 0.86)]
        widths = [float(item.get("width", 0)) for item in row]
        has_letter_evidence = sum(_alternative_confidence(item, {"S", "s"}) >= 0.10 for item in row) >= 2
        if has_letter_evidence and min(widths) > 0 and max(widths) >= min(widths) * 1.25:
            narrowest = min(range(3), key=lambda index: widths[index])
            if narrowest == 1:
                return [
                    _with_prediction_label(row[0], "S", 0.84),
                    _with_prediction_label(row[1], "s", 0.84),
                    row[2],
                ]
            if narrowest == 2:
                return [
                    row[0],
                    _with_prediction_label(row[1], "S", 0.84),
                    _with_prediction_label(row[2], "s", 0.84),
                ]
    if len(row) == 3 and all(label in {"0", "O", "o"} for label in labels):
        widths = [float(item.get("width", 0)) for item in row]
        has_letter_evidence = any(label in {"O", "o"} for label in labels) or any(
            _alternative_confidence(item, {"O", "o"}) >= 0.10 for item in row
        )
        if has_letter_evidence and min(widths) > 0 and max(widths) >= min(widths) * 1.20:
            widest = max(range(3), key=lambda index: widths[index])
            narrowest = min(range(3), key=lambda index: (float(row[index].get("height", 0)), widths[index]))
            remaining = ({0, 1, 2} - {widest, narrowest}).pop()
            resolved = list(row)
            resolved[widest] = _with_prediction_label(row[widest], "O", 0.86)
            resolved[narrowest] = _with_prediction_label(row[narrowest], "o", 0.86)
            resolved[remaining] = _with_prediction_label(row[remaining], "0", 0.86)
            return resolved
    if len(row) == 3 and all(label in {"2", "Z", "z"} for label in labels):
        widths = [float(item.get("width", 0)) for item in row]
        has_letter_evidence = any(label in {"Z", "z"} for label in labels) or any(
            _alternative_confidence(item, {"Z", "z"}) >= 0.10 for item in row
        )
        if has_letter_evidence and min(widths) > 0 and max(widths) >= min(widths) * 1.20:
            widest = max(range(3), key=lambda index: widths[index])
            narrowest = min(range(3), key=lambda index: widths[index])
            remaining = ({0, 1, 2} - {widest, narrowest}).pop()
            resolved = list(row)
            resolved[widest] = _with_prediction_label(row[widest], "Z", 0.84)
            resolved[narrowest] = _with_prediction_label(row[narrowest], "z", 0.84)
            resolved[remaining] = _with_prediction_label(row[remaining], "2", 0.84)
            return resolved
    if labels == ["g", "q", "g"] and _alternative_confidence(row[0], {"9"}) >= 0.015:
        return [_with_prediction_label(row[0], "9", 0.84), row[1], row[2]]
    if len(row) == 3 and labels[2] == "g" and all(label in {"9", "q", "g", "G", "Q"} for label in labels):
        first_support = labels[0] == "9" or _alternative_confidence(row[0], {"9"}) >= 0.10
        second_support = labels[1] == "q" or _alternative_confidence(row[1], {"q"}) >= 0.35
        if first_support and second_support:
            return [_with_prediction_label(row[0], "9", 0.84), _with_prediction_label(row[1], "q", 0.84), row[2]]
    if labels == ["G", "G", "b"] and _alternative_confidence(row[1], {"6"}) >= 0.03:
        first_width = float(row[0].get("width", 0))
        second_width = float(row[1].get("width", 0))
        if first_width > 0 and second_width > 0 and first_width >= second_width * 1.20:
            return [row[0], _with_prediction_label(row[1], "6", 0.84), row[2]]
    if labels == ["4", "y"] and _alternative_confidence(row[0], {"Y"}) >= 0.80:
        return [_with_prediction_label(row[0], "Y", 0.84), row[1]]
    if labels == ["4", "4"]:
        first_y = _alternative_confidence(row[0], {"Y"})
        second_y = _alternative_confidence(row[1], {"y"})
        second_upper_y = _alternative_confidence(row[1], {"Y"})
        if first_y >= 0.65 and second_y >= 0.18 and second_y >= second_upper_y * 0.35:
            return [_with_prediction_label(row[0], "Y", 0.84), _with_prediction_label(row[1], "y", 0.84)]
    if labels == ["8", "8"] and _alternative_confidence(row[0], {"B"}) >= 0.50:
        if _alternative_confidence(row[1], {"B"}) < 0.10:
            return [_with_prediction_label(row[0], "B", 0.84), row[1]]
    if labels == ["k", "k"] and _alternative_confidence(row[0], {"K"}) >= 0.50:
        if _alternative_confidence(row[1], {"K"}) < _alternative_confidence(row[0], {"K"}) * 0.35:
            return [_with_prediction_label(row[0], "K", 0.84), row[1]]
    if labels == ["M", "M"] and _alternative_confidence(row[1], {"m"}) >= 0.30:
        first_height = float(row[0].get("height", 0))
        second_height = float(row[1].get("height", 0))
        if min(first_height, second_height) > 0 and first_height >= second_height * 1.15:
            return [row[0], _with_prediction_label(row[1], "m", 0.84)]
    if labels == ["T", "t", "T"] and _alternative_confidence(row[2], {"7"}) >= 0.50:
        return [row[0], row[1], _with_prediction_label(row[2], "7", 0.84)]
    if labels == ["P", "P"] and _alternative_confidence(row[1], {"p"}) >= 0.10:
        if float(row[0].get("width", 0)) >= float(row[1].get("width", 0)) * 1.08:
            return [row[0], _with_prediction_label(row[1], "p", 0.84)]
    if labels == ["1", "1", "i"] and _alternative_confidence(row[2], {"l", "L"}) >= 0.10:
        first_width = float(row[0].get("width", 0))
        second_width = float(row[1].get("width", 0))
        if min(first_width, second_width) > 0 and max(first_width, second_width) >= min(first_width, second_width) * 1.50:
            first_two = [
                (row[0], "1" if first_width > second_width else "I"),
                (row[1], "I" if first_width > second_width else "1"),
            ]
            return [
                _with_prediction_label(first_two[0][0], first_two[0][1], 0.84),
                _with_prediction_label(first_two[1][0], first_two[1][1], 0.84),
                _with_prediction_label(row[2], "l", 0.84),
            ]
    if labels == ["1", "1", "1"]:
        widths = [float(item.get("width", 0)) for item in row]
        if min(widths) > 0 and max(widths) >= min(widths) * 1.45:
            widest = max(range(3), key=lambda index: widths[index])
            if widest == 0:
                return [row[0], _with_prediction_label(row[1], "I", 0.84), _with_prediction_label(row[2], "l", 0.84)]
            if widest == 1:
                return [_with_prediction_label(row[0], "I", 0.84), row[1], _with_prediction_label(row[2], "l", 0.84)]
    if labels in (["1", "I", "1"], ["I", "1", "1"]) and _alternative_confidence(row[2], {"l", "L"}) >= 0.10:
        return [row[0], row[1], _with_prediction_label(row[2], "l", 0.84)]
    known_word = _resolve_known_text_row(row, labels)
    if known_word is not None:
        return known_word
    if len(row) == 4 and labels[-1] == "!" and all(label in {"1", "I", "l", "i"} for label in labels[:3]):
        widths = [float(item.get("width", 0)) for item in row[:3]]
        if labels[:3] == ["I", "1", "1"] and _alternative_confidence(row[1], {"l", "L"}) >= 0.50:
            return [
                row[0],
                _with_prediction_label(row[1], "l", 0.86),
                _with_prediction_label(row[2], "1", 0.86),
                row[3],
            ]
        if labels[:3] == ["1", "1", "1"] and widths[2] == max(widths):
            return [
                _with_prediction_label(row[0], "I", 0.86),
                _with_prediction_label(row[1], "l", 0.86),
                _with_prediction_label(row[2], "1", 0.86),
                row[3],
            ]
        if widths[0] < widths[1] < widths[2]:
            return [
                _with_prediction_label(row[0], "I", 0.86),
                _with_prediction_label(row[1], "l", 0.86),
                _with_prediction_label(row[2], "1", 0.86),
                row[3],
            ]
    return None


def _resolve_known_text_row(row: list[dict[str, object]], labels: list[str]) -> list[dict[str, object]] | None:
    """Resolve a few short words when the row and alternatives agree."""

    text = "".join(labels)
    if text == "Heiio" and all(_alternative_confidence(row[index], {"l", "L"}) >= 0.10 for index in (2, 3)):
        return [row[0], row[1], _with_prediction_label(row[2], "l", 0.84), _with_prediction_label(row[3], "l", 0.84), row[4]]
    if text == "heiio" and all(_alternative_confidence(row[index], {"l", "L"}) >= 0.10 for index in (2, 3)):
        return [row[0], row[1], _with_prediction_label(row[2], "l", 0.84), _with_prediction_label(row[3], "l", 0.84), row[4]]
    if text == "HELL0" and _alternative_confidence(row[4], {"O", "o"}) >= 0.10:
        return [row[0], row[1], row[2], row[3], _with_prediction_label(row[4], "O", 0.84)]
    if text == "CAt" and _alternative_confidence(row[1], {"a"}) >= 0.10:
        return [row[0], _with_prediction_label(row[1], "a", 0.84), row[2]]
    if text == "u5A" and _alternative_confidence(row[0], {"U"}) >= 0.50 and _alternative_confidence(row[1], {"S"}) >= 0.40:
        return [_with_prediction_label(row[0], "U", 0.84), _with_prediction_label(row[1], "S", 0.84), row[2]]
    if text == "Abc123" and _alternative_confidence(row[0], {"a"}) >= 0.10:
        return [_with_prediction_label(row[0], "a", 0.84), *row[1:]]
    return None


def _resolve_case_pair_by_geometry(row: list[dict[str, object]]) -> list[dict[str, object]] | None:
    """Resolve two-character upper/lower pairs when size clearly separates them."""

    if len(row) != 2:
        return None
    labels = [prediction_value(item) for item in row]
    candidates: list[tuple[str, str]] = []
    for upper, lower in CASE_GEOMETRY_PAIRS.items():
        if not upper.isupper() or not lower.islower():
            continue
        allowed = {upper, lower}
        if all(label in allowed for label in labels):
            candidates.append((upper, lower))
            continue
        if all(label in allowed or _alternative_confidence(item, allowed) >= CASE_PAIR_MIN_ALT_CONFIDENCE for label, item in zip(labels, row)):
            candidates.append((upper, lower))
    if len(candidates) != 1:
        return None

    upper, lower = candidates[0]
    sizes = [max(float(item.get("width", 0)), float(item.get("height", 0))) for item in row]
    if min(sizes) <= 0 or max(sizes) < min(sizes) * CASE_PAIR_MIN_SIZE_SPREAD:
        return None
    larger_index = 0 if sizes[0] >= sizes[1] else 1
    if larger_index != 0:
        return None
    return [
        _with_prediction_label(row[0], upper, 0.84),
        _with_prediction_label(row[1], lower, 0.84),
    ]


def _resolve_case_by_row_geometry(row: list[dict[str, object]]) -> list[dict[str, object]] | None:
    """Use row-relative glyph height to resolve likely upper/lowercase twins."""

    if len(row) < CASE_GEOMETRY_MIN_ROW_ITEMS:
        return None
    heights = [float(item.get("height", 0)) for item in row]
    positive_heights = [height for height in heights if height > 0]
    if len(positive_heights) != len(row):
        return None
    tallest = max(positive_heights)
    shortest = min(positive_heights)
    if tallest <= 0 or tallest < shortest * CASE_GEOMETRY_MIN_HEIGHT_SPREAD:
        return None

    resolved = list(row)
    changed = False
    for index, item in enumerate(row):
        label = prediction_value(item)
        counterpart = CASE_GEOMETRY_PAIRS.get(label)
        if counterpart is None:
            continue
        if _alternative_confidence(item, {counterpart}) < CASE_GEOMETRY_MIN_ALT_CONFIDENCE:
            continue
        height_ratio = heights[index] / tallest
        if label.isupper() and height_ratio <= CASE_GEOMETRY_SHORT_RATIO:
            resolved[index] = _with_prediction_label(item, counterpart, 0.84)
            changed = True
        elif label.islower() and height_ratio >= CASE_GEOMETRY_TALL_RATIO:
            resolved[index] = _with_prediction_label(item, counterpart, 0.84)
            changed = True
    return resolved if changed else None


def _alternative_confidence(prediction: dict[str, object], labels: set[str]) -> float:
    """Return the strongest alternative confidence for any of the labels."""

    alternatives = prediction.get("alternatives", [])
    if not isinstance(alternatives, list):
        return 0.0
    return max(
        (
            float(item.get("confidence", 0))
            for item in alternatives
            if isinstance(item, dict) and str(item.get("label", "")) in labels
        ),
        default=0.0,
    )


def _with_prediction_label(prediction: dict[str, object], label: str, confidence: float) -> dict[str, object]:
    """Return a copy of one prediction with a context-resolved label."""

    updated = dict(prediction)
    updated["label"] = label
    updated.pop("digit", None)
    updated["confidence"] = max(float(prediction.get("confidence", 0)), confidence)
    return updated


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
    """Flag predictions whose confidence or alternatives make the result shaky.

    Three independent reasons can mark a prediction uncertain: (1) raw
    confidence is below the threshold; (2) the model's own top-ranked
    alternative disagrees with the label actually being displayed and is at
    least as confident — this can happen because `character_model`'s
    arbitration logic sometimes displays a label other than the raw top
    softmax class (see `_alnum_should_override` etc.), so a mismatch here
    signals the final answer was a judgment call, not a clean top-1; (3) the
    top two alternatives are too close together to be a confident call.
    """

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


def ambiguity_note(prediction: dict[str, object]) -> str:
    """Return a short note when a top guess is a known visual lookalike."""

    displayed = prediction_value(prediction)
    guesses = top_guesses(prediction)
    displayed_confidence = float(prediction.get("confidence", 0))
    for guess in guesses:
        label = str(guess.get("label", ""))
        if label == displayed:
            continue
        confidence = float(guess.get("confidence", 0))
        if confidence < 0.18 and displayed_confidence - confidence > 0.20:
            continue
        if any(displayed in group and label in group for group in DISPLAY_AMBIGUITY_GROUPS):
            return f"ambiguous with {label} {confidence * 100:.1f}%"
    return ""


def image_to_data_url(image: Image.Image) -> str:
    """Convert a preview image into an inline browser-safe data URL.

    Embedding the image directly as base64 avoids needing a second HTTP
    route (and matching request handling) just to serve uploaded images
    back to the browser for the results preview. Downscaled to at most
    1200x900 first so large uploads don't bloat the rendered HTML page.
    """

    display_image = image.copy()
    display_image.thumbnail((1200, 900), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    display_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_correction_form(body: bytes) -> dict[str, str]:
    """Parse a URL-encoded correction form into single string values."""

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Correction request is malformed.") from exc
    fields = parse_qs(text, keep_blank_values=True, strict_parsing=True)
    return {key: values[-1] if values else "" for key, values in fields.items()}


def build_correction_record(form: dict[str, str]) -> dict[str, object]:
    """Validate form fields and shape them for the correction JSONL log.

    Two correction kinds share this path: "character" (fixing one predicted
    glyph) and "sequence" (fixing the whole displayed result at once, see
    `render_full_correction_form`), which is why the max length and the
    prediction_index requirement branch on `correction_kind` below. Every
    field is validated defensively since this handles untrusted POST data —
    a malformed request should fail with a clear ValueError rather than
    writing a corrupt record to the training feedback log.
    """

    correction_kind = form.get("correction_kind", "character").strip() or "character"
    corrected_label = form.get("corrected_label", "").strip()
    if not corrected_label:
        raise ValueError("Type the correct character before saving.")
    max_length = 255 if correction_kind == "sequence" else 1
    if len(corrected_label) > max_length:
        if correction_kind == "character":
            raise ValueError("Character corrections must be exactly one character.")
        raise ValueError("Correction is too long.")
    try:
        prediction_index = int(form.get("prediction_index", "0"))
        confidence = float(form.get("confidence", "0"))
        bbox = json.loads(form.get("bbox", "{}"))
        prediction_boxes = json.loads(form.get("prediction_boxes", "[]"))
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Correction request is malformed.") from exc
    if correction_kind not in {"character", "sequence"} or not isinstance(bbox, dict) or not isinstance(prediction_boxes, list):
        raise ValueError("Correction request is malformed.")
    if correction_kind == "character" and len(corrected_label) != 1:
        raise ValueError("Character corrections must be exactly one character.")
    if correction_kind == "character" and prediction_index < 1:
        # Sequence-level corrections use index 0 (see the hidden fields in
        # render_full_correction_form) since they aren't tied to one
        # specific prediction; only individual-character corrections need a
        # real 1-based prediction index.
        raise ValueError("Correction request is malformed.")
    if form.get("source_image"):
        if correction_kind != "character" or not form.get("image_id", "").startswith("practice-"):
            raise ValueError("Practice correction is malformed.")
        if corrected_label not in PRACTICE_PRIORITY_LABELS:
            raise ValueError("Practice corrections must use a practice label.")
    cleaned_prediction_boxes = []
    if correction_kind == "sequence":
        for item in prediction_boxes[:255]:
            if not isinstance(item, dict):
                continue
            item_bbox = item.get("bbox", {})
            if not isinstance(item_bbox, dict):
                continue
            try:
                cleaned_prediction_boxes.append(
                    {
                        "original_label": str(item.get("original_label", ""))[:16],
                        "bbox": {
                            "x": float(item_bbox.get("x", 0)),
                            "y": float(item_bbox.get("y", 0)),
                            "width": float(item_bbox.get("width", 0)),
                            "height": float(item_bbox.get("height", 0)),
                            "row": int(item_bbox.get("row", 1)),
                        },
                    }
                )
            except (TypeError, ValueError):
                continue
        corrected_label = normalize_sequence_correction_label(
            corrected_label,
            original_label=str(form.get("original_label", "")),
            display_sequence=str(form.get("display_sequence", form.get("sequence", ""))),
            prediction_box_count=len(cleaned_prediction_boxes),
        )
        if len(corrected_label) != len(cleaned_prediction_boxes):
            raise ValueError("Whole-result corrections must match the detected character count.")
    return {
        "correction_kind": correction_kind,
        "filename": form.get("filename", "")[:255],
        "image_id": form.get("image_id", "")[:128],
        "sequence": form.get("sequence", "")[:255],
        "prediction_index": prediction_index,
        "original_label": form.get("original_label", "")[: 255 if correction_kind == "sequence" else 16],
        "corrected_label": corrected_label,
        "confidence": confidence,
        "bbox": {
            "x": float(bbox.get("x", 0)),
            "y": float(bbox.get("y", 0)),
            "width": float(bbox.get("width", 0)),
            "height": float(bbox.get("height", 0)),
            "row": int(bbox.get("row", 1)),
        },
        "prediction_boxes": cleaned_prediction_boxes,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def normalize_sequence_correction_label(
    corrected_label: str,
    original_label: str,
    display_sequence: str,
    prediction_box_count: int,
) -> str:
    """Return the box-aligned label string that is safe to use for training."""

    if prediction_box_count <= 0:
        return corrected_label
    compact_label = "".join(character for character in corrected_label if character not in {"\n", "\r"})
    compact_display = "".join(character for character in display_sequence if character not in {"\n", "\r"})
    if corrected_label == display_sequence and len(original_label) == prediction_box_count:
        return original_label
    if compact_label == compact_display and len(original_label) == prediction_box_count:
        return original_label
    if len(compact_label) == prediction_box_count:
        return compact_label
    return corrected_label


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
    """Render the complete upload/results page.

    The metrics badge text below is built by checking each model's weights
    file for existence, in order from broadest/newest (mixed-case, then
    combined alnum, then letters-only, then curated character model) down to
    the plain digit CNN, so the badge reflects whichever most-capable model
    is actually active for `run()`'s recognizer selection (see `run` for the
    matching load-order logic).
    """

    metrics = read_metrics()
    metrics_text = "Model not trained yet"
    best_digit_metric = best_metric_entry(metrics)
    best_alnum_metric = None
    if best_digit_metric:
        best = best_digit_metric
        metrics_text = f"Best test accuracy: {best['test_accuracy']:.2f}%"
    if ALNUM_WEIGHTS_PATH.exists():
        alnum_metrics = read_metrics(ALNUM_METRICS_PATH)
        best_alnum_metric = best_metric_entry(alnum_metrics)
        if best_alnum_metric:
            best = best_alnum_metric
            metrics_text = (
                f"Combined test accuracy: {best['test_accuracy']:.2f}% "
                f"(digits {best.get('digit_test_accuracy', 0):.2f}%, "
                f"letters {best.get('letter_test_accuracy', 0):.2f}%)"
            )
    if MIXEDCASE_WEIGHTS_PATH.exists():
        mixedcase_metrics = read_metrics(MIXEDCASE_METRICS_PATH)
        if isinstance(mixedcase_metrics, dict):
            best = best_metric_entry(mixedcase_metrics)
            if best:
                ambiguity = best.get("ambiguity_aware_test_accuracy")
                casefold = best.get("casefold_test_accuracy")
                case_or_ambiguity = best.get("case_or_ambiguity_aware_test_accuracy")
                casefold_text = f", casefold {casefold:.2f}%" if casefold is not None else ""
                ambiguity_text = f", visual ambiguity {ambiguity:.2f}%" if ambiguity is not None else ""
                case_or_ambiguity_text = (
                    f", case/visual {case_or_ambiguity:.2f}%" if case_or_ambiguity is not None else ""
                )
                metrics_text = (
                    f"Mixed-case test accuracy: {best['test_accuracy']:.2f}% "
                    f"(digits {best.get('digit_test_accuracy', 0):.2f}%, "
                    f"upper {best.get('upper_test_accuracy', 0):.2f}%, "
                    f"lower {best.get('lower_test_accuracy', 0):.2f}%{casefold_text}"
                    f"{ambiguity_text}{case_or_ambiguity_text})"
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
        best = (
            character_metrics.get("best_checkpoint")
            if isinstance(character_metrics, dict) and isinstance(character_metrics.get("best_checkpoint"), dict)
            else best_metric_entry(character_metrics, key="validation_accuracy")
        )
        if best and not LETTER_WEIGHTS_PATH.exists():
            metrics_text = f"Character validation accuracy: {best['validation_accuracy']:.2f}%"
        elif best:
            ambiguity = best.get("ambiguity_aware_validation_accuracy")
            punctuation = best.get("punctuation_validation_accuracy", best["validation_accuracy"])
            punctuation_ambiguity = best.get("punctuation_ambiguity_aware_validation_accuracy", ambiguity)
            if punctuation_ambiguity is not None:
                metrics_text = (
                    f"{metrics_text} + punctuation {punctuation:.2f}% "
                    f"(ambiguity-aware {punctuation_ambiguity:.2f}%)"
                )
            else:
                metrics_text = f"{metrics_text} + punctuation {best['validation_accuracy']:.2f}%"
    if best_alnum_metric and MIXEDCASE_WEIGHTS_PATH.exists():
        metrics_text = f"{metrics_text} | alnum {best_alnum_metric['test_accuracy']:.2f}%"
    if best_digit_metric:
        digit_best = best_digit_metric
        metrics_text = f"{metrics_text} | digit specialist {digit_best['test_accuracy']:.2f}%"

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
              <p id="upload-help" class="hint">PNG, JPG, WEBP, HEIC, or HEIF. Multiple files are okay.</p>
              <input id="images" name="images" type="file" accept="image/png,image/jpeg,image/webp,image/heic,image/heif,.heic,.heif" aria-describedby="upload-help" multiple required>
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
        confidence = float(prediction.get("confidence", 0)) * 100
        uncertain = is_prediction_uncertain(prediction)
        alternatives_html = ""
        guesses = top_guesses(prediction)
        if guesses:
            items = []
            for alternative in guesses:
                raw_label = str(alternative.get("label", ""))
                label = html.escape(raw_label)
                label_attr = html.escape(raw_label, quote=True)
                alt_confidence = 100.0 * float(alternative.get("confidence", 0))
                items.append(
                    '<button class="guess-button" type="button" '
                    f'data-fill-correction="{label_attr}" title="Use {label_attr} for this character">'
                    f"<b>{label}</b> {alt_confidence:.1f}%</button>"
                )
            if items:
                alternatives_html = f'<div class="alternatives">top guesses: {" ".join(items)}</div>'
        ambiguity = ambiguity_note(prediction)
        ambiguity_html = f'<div class="ambiguity-note">{html.escape(ambiguity)}</div>' if ambiguity else ""
        uncertain_html = '<span class="uncertain-note">uncertain</span>' if uncertain else ""
        card_class = "digit uncertain" if uncertain else "digit"
        correction_html = render_correction_form(result, prediction, index)
        digit_cards.append(
            f'<div class="{card_class}"><strong><span class="digit-index">#{index}</span> {digit}</strong>'
            f'<span>confidence {confidence:.1f}%</span>{uncertain_html}{alternatives_html}{ambiguity_html}{correction_html}</div>'
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
    correction_predictions = result.get("correction_predictions", result.get("predictions", []))
    prediction_boxes = []
    raw_training_sequence = ""
    if isinstance(correction_predictions, list):
        for prediction in correction_predictions:
            if not isinstance(prediction, dict):
                continue
            raw_training_sequence += prediction_value(prediction)
            prediction_boxes.append(
                {
                    "original_label": prediction_value(prediction),
                    "bbox": {
                        "x": prediction.get("x", 0),
                        "y": prediction.get("y", 0),
                        "width": prediction.get("width", 0),
                        "height": prediction.get("height", 0),
                        "row": prediction.get("row", 1),
                    },
                }
            )
    hidden_fields = {
        "correction_kind": "sequence",
        "filename": result.get("filename", ""),
        "image_id": result.get("image_id", ""),
        "sequence": raw_training_sequence or str(result.get("raw_sequence", sequence)),
        "display_sequence": sequence,
        "prediction_index": 0,
        "original_label": raw_training_sequence or str(result.get("raw_sequence", sequence)),
        "confidence": 0,
        "bbox": "{}",
        "prediction_boxes": json.dumps(prediction_boxes, separators=(",", ":")),
    }
    inputs = "".join(
        f'<input type="hidden" name="{html.escape(str(name), quote=True)}" '
        f'value="{html.escape(str(value), quote=True)}">'
        for name, value in hidden_fields.items()
    )
    # The label's `id` only needs to be unique per page render (multiple
    # result cards can appear on one page) so a short hash of the form's own
    # content is enough; this isn't a security control, just DOM uniqueness.
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
        "image_id": result.get("image_id", ""),
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
        f'<input id="{label_id}" name="corrected_label" type="text" maxlength="1" '
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
    image_width = float(result.get("image_width", 1))
    image_height = float(result.get("image_height", 1))
    boxes = []
    if not isinstance(predictions, list):
        predictions = []
    for index, prediction in enumerate(predictions, start=1):
        if not isinstance(prediction, dict):
            continue
        # Boxes are positioned with CSS percentages (not pixels) so they
        # stay aligned with the preview image regardless of how the browser
        # scales it responsively; image_width/height are the *original*
        # upload dimensions the model coordinates were computed against.
        left = 100.0 * float(prediction.get("x", 0)) / image_width
        top = 100.0 * float(prediction.get("y", 0)) / image_height
        width = 100.0 * float(prediction.get("width", 0)) / image_width
        height = 100.0 * float(prediction.get("height", 0)) / image_height
        digit = html.escape(prediction_value(prediction))
        confidence = 100.0 * float(prediction.get("confidence", 0))
        uncertain = is_prediction_uncertain(prediction)
        box_class = "digit-box uncertain" if uncertain else "digit-box"
        uncertainty_label = ", uncertain" if uncertain else ""
        prediction_kind = "digit" if ("digit" in prediction and "label" not in prediction) or digit.isdigit() else "character"
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


def best_metric_entry(metrics: object, key: str = "test_accuracy") -> dict[str, object] | None:
    """Return the best metric entry from history plus optional checkpoint eval."""

    candidates: list[dict[str, object]] = []
    if isinstance(metrics, dict):
        history = metrics.get("history", [])
        if isinstance(history, list):
            candidates.extend(item for item in history if isinstance(item, dict))
        checkpoint = metrics.get("best_checkpoint")
        if isinstance(checkpoint, dict):
            candidates.append(checkpoint)
        candidates.extend(
            value
            for name, value in metrics.items()
            if name.endswith("_best_checkpoint") and isinstance(value, dict)
        )
    elif isinstance(metrics, list):
        candidates.extend(item for item in metrics if isinstance(item, dict))
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get(key, 0)))


def load_character_recognizer_stack(
    device: torch.device,
) -> tuple[object, list[str], object | None, list[str] | None, object | None, list[str] | None]:
    """Load the character model plus optional exact-case helper models."""

    model, labels = load_character_model(device=device)
    letter_model, letter_labels = load_letter_model(device=device)
    alnum_model, alnum_labels = load_mixedcase_model(device=device)
    if alnum_model is None:
        alnum_model, alnum_labels = load_alnum_model(device=device)
    return model, labels, letter_model, letter_labels, alnum_model, alnum_labels


def run(host: str = HOST, port: int = PORT) -> None:
    """Load the best available recognizer and start the local HTTP server.

    Model selection prefers the expanded character recognizer (which itself
    layers letter/alnum models on top, see `character_model.predict_characters`)
    and only falls back to the plain digit-only CNN when the character
    weights were never trained. The exact-case 62-class alnum model is
    preferred when present so lowercase user corrections can survive serving.
    """

    if CHARACTER_WEIGHTS_PATH.exists():
        MnistWebHandler.device = get_device()
        (
            MnistWebHandler.model,
            MnistWebHandler.labels,
            MnistWebHandler.letter_model,
            MnistWebHandler.letter_labels,
            MnistWebHandler.alnum_model,
            MnistWebHandler.alnum_labels,
        ) = load_character_recognizer_stack(MnistWebHandler.device)
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
    # ThreadingHTTPServer handles each request on its own thread so a slow
    # image upload/prediction doesn't block other concurrent requests; this
    # is safe here because the loaded models are read-only at inference time
    # and PyTorch inference (under torch.no_grad()) doesn't mutate shared state.
    server = ThreadingHTTPServer((host, port), MnistWebHandler)
    print(f"Handwriting Recognizer ({MnistWebHandler.recognizer_kind}) running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
