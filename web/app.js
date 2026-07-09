const uploadForm = document.querySelector("#upload-form");
const imageInput = document.querySelector("#image-input");
const resultsEl = document.querySelector("#results");
const statusEl = document.querySelector("#status-line");
const serverPill = document.querySelector("#server-pill");
const modelNote = document.querySelector("#model-note");

const lowConfidenceThreshold = 0.8;
const closeGuessMargin = 0.12;

function text(value) {
  return value === undefined || value === null ? "" : String(value);
}

function predictionLabel(prediction) {
  return text(prediction.label ?? prediction.digit);
}

function predictionConfidence(prediction) {
  return Number(prediction.confidence ?? 0);
}

function topGuesses(prediction) {
  const alternatives = Array.isArray(prediction.alternatives) ? prediction.alternatives : [];
  return alternatives
    .filter((item) => item && text(item.label) !== "")
    .slice()
    .sort((a, b) => Number(b.confidence ?? 0) - Number(a.confidence ?? 0))
    .slice(0, 3);
}

function isUncertain(prediction) {
  const confidence = predictionConfidence(prediction);
  if (confidence < lowConfidenceThreshold) {
    return true;
  }
  const guesses = topGuesses(prediction);
  if (guesses.length === 0) {
    return false;
  }
  const displayed = predictionLabel(prediction);
  const topConfidence = Number(guesses[0].confidence ?? 0);
  if (text(guesses[0].label) !== displayed && topConfidence >= confidence) {
    return true;
  }
  if (guesses.length < 2) {
    return false;
  }
  return topConfidence - Number(guesses[1].confidence ?? 0) <= closeGuessMargin;
}

function setStatus(message, className = "") {
  statusEl.textContent = message;
  statusEl.className = `status-line ${className}`.trim();
}

function makeElement(tagName, className = "", content = "") {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (content !== "") {
    element.textContent = content;
  }
  return element;
}

function hiddenInput(name, value) {
  const input = document.createElement("input");
  input.type = "hidden";
  input.name = name;
  input.value = text(value);
  return input;
}

async function postCorrection(form, statusTarget) {
  statusTarget.textContent = "Saving...";
  const button = form.querySelector("button[type='submit']");
  if (button) {
    button.disabled = true;
  }
  try {
    const response = await fetch("/api/correct", {
      method: "POST",
      headers: {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
      body: new URLSearchParams(new FormData(form)),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Could not save that correction.");
    }
    statusTarget.textContent = "Saved. You can edit it again.";
  } catch (error) {
    statusTarget.textContent = error.message;
  } finally {
    if (button) {
      button.disabled = false;
    }
  }
}

function bindCorrectionForm(form) {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const statusTarget = form.querySelector("[data-correction-status]");
    postCorrection(form, statusTarget);
  });
}

function predictionBoxData(prediction) {
  return {
    x: Number(prediction.x ?? 0),
    y: Number(prediction.y ?? 0),
    width: Number(prediction.width ?? 0),
    height: Number(prediction.height ?? 0),
    row: Number(prediction.row ?? 1),
  };
}

function renderRows(result, panel) {
  const rows = Array.isArray(result.row_sequences) ? result.row_sequences : [];
  if (rows.length > 1) {
    const rowWrap = makeElement("div", "row-output");
    rows.forEach((row, index) => {
      rowWrap.append(makeElement("code", "row-chip", `row ${index + 1}: ${row}`));
    });
    panel.append(rowWrap);
  }
  const notes = Array.isArray(result.context_notes) ? result.context_notes : [];
  if (notes.length > 0) {
    const noteWrap = makeElement("div", "context-output");
    notes.forEach((note) => noteWrap.append(makeElement("code", "row-chip", note)));
    panel.append(noteWrap);
  }
}

function renderFullCorrection(result, panel) {
  const predictions = Array.isArray(result.correction_predictions)
    ? result.correction_predictions
    : Array.isArray(result.predictions)
      ? result.predictions
      : [];
  if (predictions.length === 0) {
    return;
  }
  const predictionBoxes = predictions.map((prediction) => ({
    original_label: predictionLabel(prediction),
    bbox: predictionBoxData(prediction),
  }));
  const rawSequence = predictionBoxes.map((item) => item.original_label).join("");
  const form = makeElement("form", "full-correction");
  form.append(hiddenInput("correction_kind", "sequence"));
  form.append(hiddenInput("filename", result.filename));
  form.append(hiddenInput("image_id", result.image_id));
  form.append(hiddenInput("sequence", rawSequence || result.raw_sequence || result.sequence));
  form.append(hiddenInput("display_sequence", result.sequence));
  form.append(hiddenInput("prediction_index", "0"));
  form.append(hiddenInput("original_label", rawSequence || result.raw_sequence || result.sequence));
  form.append(hiddenInput("confidence", "0"));
  form.append(hiddenInput("bbox", "{}"));
  form.append(hiddenInput("prediction_boxes", JSON.stringify(predictionBoxes)));

  const label = makeElement("label", "", "Fix the whole result");
  const input = makeElement("input", "text-input");
  input.name = "corrected_label";
  input.type = "text";
  input.maxLength = 255;
  input.autocomplete = "off";
  input.value = text(result.sequence);
  const button = makeElement("button", "save-button", "Save all");
  button.type = "submit";
  const status = makeElement("span", "correction-status");
  status.dataset.correctionStatus = "";

  form.append(label, input, button, status);
  bindCorrectionForm(form);
  panel.append(form);
}

function renderPreview(result, predictions, panel) {
  if (!result.preview) {
    return;
  }
  const wrap = makeElement("div", "preview-wrap");
  const image = document.createElement("img");
  image.src = result.preview;
  image.alt = "Uploaded handwriting image with prediction boxes";
  wrap.append(image);

  const imageWidth = Number(result.image_width || 1);
  const imageHeight = Number(result.image_height || 1);
  predictions.forEach((prediction, index) => {
    const boxData = predictionBoxData(prediction);
    const box = makeElement("div", isUncertain(prediction) ? "prediction-box uncertain" : "prediction-box");
    box.style.left = `${(100 * boxData.x) / imageWidth}%`;
    box.style.top = `${(100 * boxData.y) / imageHeight}%`;
    box.style.width = `${(100 * boxData.width) / imageWidth}%`;
    box.style.height = `${(100 * boxData.height) / imageHeight}%`;
    box.title = `#${index + 1}: ${predictionLabel(prediction)} ${(predictionConfidence(prediction) * 100).toFixed(1)}%`;
    box.append(makeElement("span", "", `#${index + 1}`));
    wrap.append(box);
  });
  panel.append(wrap);
}

function renderGuessButtons(prediction, correctionInput) {
  const guesses = topGuesses(prediction);
  if (guesses.length === 0) {
    return null;
  }
  const wrap = makeElement("div", "top-guesses");
  wrap.append(makeElement("div", "", "top guesses:"));
  const row = makeElement("div", "guess-row");
  guesses.forEach((guess) => {
    const button = makeElement(
      "button",
      "guess-button",
      `${text(guess.label)} ${(Number(guess.confidence ?? 0) * 100).toFixed(1)}%`,
    );
    button.type = "button";
    button.addEventListener("click", () => {
      correctionInput.value = text(guess.label);
      correctionInput.focus();
    });
    row.append(button);
  });
  wrap.append(row);
  return wrap;
}

function renderPredictionCard(result, prediction, index) {
  const uncertain = isUncertain(prediction);
  const card = makeElement("article", uncertain ? "prediction-card uncertain" : "prediction-card");

  const label = makeElement("div", "card-label");
  label.append(makeElement("span", "", `#${index + 1}`));
  label.append(document.createTextNode(predictionLabel(prediction)));
  card.append(label);
  card.append(makeElement("div", "card-meta", `confidence ${(predictionConfidence(prediction) * 100).toFixed(1)}%`));
  if (uncertain) {
    card.append(makeElement("div", "uncertain-badge", "UNCERTAIN"));
  }

  const form = makeElement("form", "correction-form");
  form.append(hiddenInput("correction_kind", "character"));
  form.append(hiddenInput("filename", result.filename));
  form.append(hiddenInput("image_id", result.image_id));
  form.append(hiddenInput("sequence", result.sequence));
  form.append(hiddenInput("prediction_index", String(index + 1)));
  form.append(hiddenInput("original_label", predictionLabel(prediction)));
  form.append(hiddenInput("confidence", String(predictionConfidence(prediction))));
  form.append(hiddenInput("bbox", JSON.stringify(predictionBoxData(prediction))));

  const input = makeElement("input", "text-input");
  input.name = "corrected_label";
  input.type = "text";
  input.maxLength = 1;
  input.autocomplete = "off";
  input.placeholder = "fix";

  const guesses = renderGuessButtons(prediction, input);
  if (guesses) {
    card.append(guesses);
  }

  const save = makeElement("button", "save-button", "Save");
  save.type = "submit";
  const status = makeElement("span", "correction-status");
  status.dataset.correctionStatus = "";
  form.append(input, save, status);
  bindCorrectionForm(form);
  card.append(form);
  return card;
}

function renderResult(result) {
  const panel = makeElement("article", "result-panel");
  if (result.error) {
    panel.append(makeElement("div", "filename", result.filename || "Upload"));
    panel.append(makeElement("div", "error", result.error));
    return panel;
  }

  const head = makeElement("div", "result-head");
  head.append(makeElement("div", "filename", result.filename));
  head.append(makeElement("div", "sequence", result.sequence));
  panel.append(head);

  renderRows(result, panel);
  renderFullCorrection(result, panel);

  const predictions = Array.isArray(result.predictions) ? result.predictions : [];
  renderPreview(result, predictions, panel);

  const cards = makeElement("div", "cards");
  predictions.forEach((prediction, index) => cards.append(renderPredictionCard(result, prediction, index)));
  panel.append(cards);
  return panel;
}

function renderResults(results) {
  resultsEl.replaceChildren();
  if (!Array.isArray(results) || results.length === 0) {
    resultsEl.append(makeElement("div", "empty-state", "No recognizable handwriting was returned."));
    return;
  }
  results.forEach((result) => resultsEl.append(renderResult(result)));
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!imageInput.files || imageInput.files.length === 0) {
    setStatus("Choose at least one image first.", "error");
    return;
  }
  const submitButton = uploadForm.querySelector("button[type='submit']");
  submitButton.disabled = true;
  setStatus(`Recognizing ${imageInput.files.length} file${imageInput.files.length === 1 ? "" : "s"}...`);
  try {
    const formData = new FormData(uploadForm);
    const response = await fetch("/api/predict", {method: "POST", body: formData});
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Prediction failed.");
    }
    renderResults(payload.results);
    setStatus("Done.", "notice");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
});

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error("offline");
    }
    serverPill.textContent = `${payload.recognizer || "model"} live`;
    serverPill.classList.add("live");
    modelNote.textContent = "Upload handwriting images and correct any result for retraining.";
  } catch {
    serverPill.textContent = "offline";
    serverPill.classList.remove("live");
  }
}

checkHealth();
