const uploadForm = document.querySelector("#upload-form");
const imageInput = document.querySelector("#image-input");
const resultsEl = document.querySelector("#results");
const statusEl = document.querySelector("#status-line");
const serverPill = document.querySelector("#server-pill");
const modelNote = document.querySelector("#model-note");
const practiceCanvas = document.querySelector("#practice-canvas");
const practiceForm = document.querySelector("#practice-form");
const practiceLabelsEl = document.querySelector("#practice-labels");
const practiceCoverageEl = document.querySelector("#practice-coverage");
const practiceReadinessEl = document.querySelector("#practice-readiness");
const practiceLabelInput = document.querySelector("#practice-label-input");
const practiceTargetEl = document.querySelector("#practice-target");
const practiceTargetProgressEl = document.querySelector("#practice-target-progress");
const practiceAutoNextInput = document.querySelector("#practice-auto-next");
const practiceClearButton = document.querySelector("#practice-clear");
const practiceNextButton = document.querySelector("#practice-next-needed");
const practiceStatus = document.querySelector("#practice-status");

const lowConfidenceThreshold = 0.8;
const closeGuessMargin = 0.12;
let practiceLabels = ["0"];
let practicePointerDown = false;
let practiceHasInk = false;
let latestPracticeCoverage = null;

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

function practiceContext() {
  return practiceCanvas.getContext("2d");
}

function clearPracticeCanvas(clearStatus = true) {
  const context = practiceContext();
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, practiceCanvas.width, practiceCanvas.height);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "#111111";
  context.lineWidth = 9;
  practiceHasInk = false;
  if (clearStatus) {
    practiceStatus.textContent = "";
  }
}

function setPracticeLabel(label) {
  practiceLabelInput.value = label;
  practiceTargetEl.textContent = label;
  practiceLabelsEl.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("selected", button.dataset.label === label);
  });
  renderSelectedPracticeProgress();
}

function practiceLabelValuesFromCoverage(payload) {
  if (!payload || !Array.isArray(payload.labels)) {
    return [];
  }
  return payload.labels.map((item) => text(item.label)).filter((label) => label.length === 1);
}

function renderPracticeLabelButtons(labels) {
  const nextLabels = labels.length > 0 ? labels : ["0"];
  const selectedLabel = text(practiceLabelInput.value);
  practiceLabels = nextLabels;
  practiceLabelsEl.replaceChildren();
  practiceLabels.forEach((label) => {
    const button = makeElement("button", "practice-label-button", label);
    button.type = "button";
    button.dataset.label = label;
    button.addEventListener("click", () => setPracticeLabel(label));
    practiceLabelsEl.append(button);
  });
  setPracticeLabel(practiceLabels.includes(selectedLabel) ? selectedLabel : practiceLabels[0]);
}

function selectedPracticeCoverage(label) {
  if (!latestPracticeCoverage || !Array.isArray(latestPracticeCoverage.labels)) {
    return null;
  }
  return latestPracticeCoverage.labels.find((item) => text(item.label) === label) || null;
}

function renderSelectedPracticeProgress() {
  if (!practiceTargetProgressEl) {
    return;
  }
  const label = text(practiceLabelInput.value);
  const coverage = selectedPracticeCoverage(label);
  const target = Number(latestPracticeCoverage?.target_per_label || 20);
  const count = Number(coverage?.count || 0);
  const needed = Number(coverage?.needed ?? Math.max(0, target - count));
  practiceTargetProgressEl.textContent = needed > 0 ? `${count}/${target} saved, ${needed} needed` : `${count}/${target} saved, ready`;
  practiceTargetProgressEl.classList.toggle("ready", needed <= 0);
}

function repeatPracticeStatus(label) {
  const coverage = selectedPracticeCoverage(label);
  if (!coverage) {
    return `Saved ${label}.`;
  }
  const needed = Number(coverage.needed || 0);
  return needed > 0 ? `Saved ${label}. ${needed} more ${label} needed.` : `Saved ${label}. ${label} is ready.`;
}

function nextNeededPracticeLabel() {
  if (!latestPracticeCoverage || !Array.isArray(latestPracticeCoverage.labels)) {
    return practiceLabels[0];
  }
  const labelRanks = new Map(practiceLabels.map((label, index) => [label, index]));
  const needyLabels = latestPracticeCoverage.labels
    .filter((item) => Number(item.needed || 0) > 0)
    .slice()
    .sort((left, right) => {
      const neededDelta = Number(right.needed || 0) - Number(left.needed || 0);
      if (neededDelta !== 0) {
        return neededDelta;
      }
      const countDelta = Number(left.count || 0) - Number(right.count || 0);
      if (countDelta !== 0) {
        return countDelta;
      }
      return (labelRanks.get(text(left.label)) ?? 999) - (labelRanks.get(text(right.label)) ?? 999);
    });
  return text(needyLabels[0]?.label || practiceLabels[0]);
}

function selectNextNeededPracticeLabel(showStatus = true) {
  const label = nextNeededPracticeLabel();
  setPracticeLabel(label);
  if (showStatus) {
    practiceStatus.textContent = `Next needed: ${label}`;
  }
}

function renderPracticeCoverage(payload) {
  if (!practiceCoverageEl || !payload || !Array.isArray(payload.labels)) {
    return;
  }
  latestPracticeCoverage = payload;
  renderPracticeLabelButtons(practiceLabelValuesFromCoverage(payload));
  practiceCoverageEl.replaceChildren();
  const summary = makeElement(
    "div",
    "practice-coverage-summary",
    `${payload.ready_labels || 0}/${payload.total_labels || practiceLabels.length} labels ready, target ${payload.target_per_label || 20} each`,
  );
  practiceCoverageEl.append(summary);
  const grid = makeElement("div", "practice-coverage-grid");
  payload.labels.forEach((item) => {
    const count = Number(item.count || 0);
    const needed = Number(item.needed || 0);
    const chip = makeElement("button", item.ready ? "coverage-chip ready" : "coverage-chip", `${text(item.label)} ${count}`);
    chip.type = "button";
    chip.title = item.ready ? "Ready for correction training" : `${needed} more needed`;
    chip.addEventListener("click", () => setPracticeLabel(text(item.label)));
    grid.append(chip);
  });
  practiceCoverageEl.append(grid);
}

async function refreshPracticeCoverage(selectNext = false) {
  if (!practiceCoverageEl) {
    return null;
  }
  try {
    const response = await fetch("/api/correction-coverage");
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error("coverage unavailable");
    }
    renderPracticeCoverage(payload);
    if (selectNext) {
      selectNextNeededPracticeLabel(false);
    }
    return payload;
  } catch {
    practiceCoverageEl.replaceChildren(makeElement("div", "practice-coverage-summary", "Coverage unavailable"));
    return null;
  }
}

function renderReadinessCard(name, report) {
  const readiness = report?.readiness || {};
  const nextNeeded = Array.isArray(report?.next_needed) ? report.next_needed.slice(0, 4) : [];
  const samples = Number(readiness.samples || 0);
  const targetSamples = Number(readiness.target_samples || 0);
  const percent = targetSamples > 0 ? Math.min(100, Math.max(0, (100 * samples) / targetSamples)) : 0;
  const card = makeElement("div", readiness.ready ? "readiness-card ready" : "readiness-card");
  card.append(makeElement("strong", "", name));
  const meter = makeElement("div", "readiness-meter");
  const fill = makeElement("div", "readiness-meter-fill");
  fill.style.width = `${percent.toFixed(1)}%`;
  meter.append(fill);
  card.append(meter);
  card.append(
    makeElement(
      "span",
      "",
      `${Number(readiness.ready_labels || 0)}/${Number(readiness.total_labels || 0)} labels, ${samples}/${targetSamples} samples`,
    ),
  );
  card.append(makeElement("span", "", readiness.ready ? "ready" : `${Number(readiness.needed_samples || 0)} needed`));
  if (nextNeeded.length > 0) {
    const nextWrap = makeElement("div", "readiness-next");
    nextWrap.append(makeElement("span", "", "next"));
    nextNeeded.forEach((item) => {
      const label = text(item.label);
      const button = makeElement("button", "readiness-next-button", `${label}:${Number(item.needed || 0)}`);
      button.type = "button";
      button.addEventListener("click", () => setPracticeLabel(label));
      nextWrap.append(button);
    });
    card.append(nextWrap);
  }
  return card;
}

function renderCorrectionReadiness(payload) {
  if (!practiceReadinessEl || !payload || !payload.ok) {
    return;
  }
  practiceReadinessEl.replaceChildren();
  practiceReadinessEl.append(makeElement("div", "practice-coverage-summary", "Training readiness"));
  const grid = makeElement("div", "readiness-grid");
  grid.append(renderReadinessCard("Character", payload.character || {}));
  grid.append(renderReadinessCard("Folded", payload.folded_alnum || {}));
  grid.append(renderReadinessCard("Mixed case", payload.mixedcase || {}));
  practiceReadinessEl.append(grid);
}

async function refreshCorrectionReadiness() {
  if (!practiceReadinessEl) {
    return;
  }
  try {
    const response = await fetch("/api/correction-readiness");
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error("readiness unavailable");
    }
    renderCorrectionReadiness(payload);
  } catch {
    practiceReadinessEl.replaceChildren(makeElement("div", "practice-coverage-summary", "Readiness unavailable"));
  }
}

function practicePoint(event) {
  const rect = practiceCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * practiceCanvas.width) / rect.width,
    y: ((event.clientY - rect.top) * practiceCanvas.height) / rect.height,
  };
}

function beginPracticeStroke(event) {
  event.preventDefault();
  practicePointerDown = true;
  practiceCanvas.setPointerCapture(event.pointerId);
  const point = practicePoint(event);
  const context = practiceContext();
  context.beginPath();
  context.moveTo(point.x, point.y);
}

function drawPracticeStroke(event) {
  if (!practicePointerDown) {
    return;
  }
  event.preventDefault();
  const point = practicePoint(event);
  const context = practiceContext();
  context.lineTo(point.x, point.y);
  context.stroke();
  practiceHasInk = true;
}

function endPracticeStroke(event) {
  if (!practicePointerDown) {
    return;
  }
  event.preventDefault();
  practicePointerDown = false;
  try {
    practiceCanvas.releasePointerCapture(event.pointerId);
  } catch {
    // Some browsers release capture automatically when the pointer leaves.
  }
}

function practiceImageId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return `practice-${window.crypto.randomUUID()}`;
  }
  return `practice-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function savePracticeSample(event) {
  event.preventDefault();
  await submitPracticeSample();
}

async function submitPracticeSample() {
  const label = text(practiceLabelInput.value).trim();
  if (label.length !== 1) {
    practiceStatus.textContent = "Pick one label.";
    practiceLabelInput.focus();
    return;
  }
  if (!practiceHasInk) {
    practiceStatus.textContent = "Draw the sample first.";
    return;
  }
  const imageId = practiceImageId();
  const payload = new URLSearchParams({
    correction_kind: "character",
    filename: `${imageId}.png`,
    image_id: imageId,
    sequence: label,
    prediction_index: "1",
    original_label: label,
    corrected_label: label,
    confidence: "1",
    bbox: JSON.stringify({x: 0, y: 0, width: practiceCanvas.width, height: practiceCanvas.height, row: 1}),
    prediction_boxes: "[]",
    source_image: practiceCanvas.toDataURL("image/png"),
  });
  practiceStatus.textContent = "Saving...";
  const button = practiceForm.querySelector("button[type='submit']");
  button.disabled = true;
  try {
    const response = await fetch("/api/correct", {
      method: "POST",
      headers: {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
      body: payload,
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.error || "Could not save that sample.");
    }
    const autoNext = Boolean(practiceAutoNextInput?.checked);
    clearPracticeCanvas(false);
    await refreshPracticeCoverage(autoNext);
    await refreshCorrectionReadiness();
    practiceStatus.textContent = autoNext ? `Saved ${label}.` : repeatPracticeStatus(label);
  } catch (error) {
    practiceStatus.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function eventIsInsidePractice(event) {
  return Boolean(event.target && practiceForm.closest(".practice-panel")?.contains(event.target));
}

function handlePracticeShortcut(event) {
  if (!eventIsInsidePractice(event)) {
    return;
  }
  if ((event.metaKey || event.ctrlKey) && (event.key === "Enter" || event.key.toLowerCase() === "s")) {
    event.preventDefault();
    submitPracticeSample();
    return;
  }
  if (event.key === "Escape") {
    event.preventDefault();
    clearPracticeCanvas();
    return;
  }
  if (event.altKey && event.key.toLowerCase() === "n") {
    event.preventDefault();
    selectNextNeededPracticeLabel();
  }
}

function setupPracticeMode() {
  if (!practiceCanvas || !practiceForm) {
    return;
  }
  renderPracticeLabelButtons(practiceLabels);
  practiceCanvas.addEventListener("pointerdown", beginPracticeStroke);
  practiceCanvas.addEventListener("pointermove", drawPracticeStroke);
  practiceCanvas.addEventListener("pointerup", endPracticeStroke);
  practiceCanvas.addEventListener("pointercancel", endPracticeStroke);
  practiceClearButton.addEventListener("click", clearPracticeCanvas);
  practiceNextButton.addEventListener("click", () => selectNextNeededPracticeLabel());
  practiceLabelInput.addEventListener("input", () => setPracticeLabel(text(practiceLabelInput.value).slice(0, 1)));
  practiceForm.addEventListener("submit", savePracticeSample);
  document.addEventListener("keydown", handlePracticeShortcut);
  setPracticeLabel(practiceLabels[0]);
  clearPracticeCanvas();
  refreshPracticeCoverage(true);
  refreshCorrectionReadiness();
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
setupPracticeMode();
