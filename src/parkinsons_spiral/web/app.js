const canvas = document.getElementById("spiralCanvas");
const context = canvas.getContext("2d");

const state = {
  sessionId: null,
  steps: [],
  currentStep: 0,
  points: [],
  stroke: 0,
  drawing: false,
  firstTimestamp: null,
  pressureMin: 1,
  pressureMax: 0,
  templateVisible: true,
  blinkTimer: null,
  spiralResult: null,
  voiceRepetition: 1,
};

const $ = (id) => document.getElementById(id);

function showMessage(element, text, type = "error") {
  element.textContent = text;
  element.classList.toggle("success", type === "success");
  element.hidden = false;
}

function hideMessage(element) {
  element.hidden = true;
  element.textContent = "";
  element.classList.remove("success");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    } catch (_) {
      // Keep the HTTP fallback message.
    }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function buildSteps(handedness) {
  const first = handedness === "left" ? "left" : "right";
  return [
    { hand: first, mode: "static", repetition: 1 },
    { hand: first, mode: "dynamic", repetition: 1 },
  ];
}

function titleCase(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function renderProtocol() {
  $("protocolList").innerHTML = state.steps
    .map((step, index) => {
      const status = index < state.currentStep ? "done" : index === state.currentStep ? "active" : "";
      return `<li class="${status}">${titleCase(step.hand)} hand · ${titleCase(step.mode)}</li>`;
    })
    .join("");
}

function drawTemplate() {
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const maxRadius = canvas.width * 0.37;
  const turns = 3;
  const maxTheta = turns * Math.PI * 2;

  context.save();
  context.lineWidth = 3;
  context.strokeStyle = "#c7cec9";
  context.setLineDash([6, 5]);
  context.beginPath();
  for (let index = 0; index <= 720; index += 1) {
    const theta = (index / 720) * maxTheta;
    const radius = 8 + (theta / maxTheta) * maxRadius;
    const x = centerX + radius * Math.cos(theta);
    const y = centerY + radius * Math.sin(theta);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = "#79ad71";
  context.beginPath();
  context.arc(centerX + 8, centerY, 8, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function drawCapturedPath() {
  if (!state.points.length) return;
  context.save();
  context.strokeStyle = "#1e5a50";
  context.lineWidth = 4;
  context.lineCap = "round";
  context.lineJoin = "round";
  let previousStroke = null;
  for (const point of state.points) {
    if (point.stroke !== previousStroke) {
      if (previousStroke !== null) context.stroke();
      context.beginPath();
      context.moveTo(point.x, point.y);
      previousStroke = point.stroke;
    } else {
      context.lineTo(point.x, point.y);
    }
  }
  context.stroke();
  context.restore();
}

function renderCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const step = state.steps[state.currentStep];
  if (!step || step.mode === "static" || state.templateVisible) drawTemplate();
  drawCapturedPath();
}

function resetTrial() {
  state.points = [];
  state.stroke = 0;
  state.drawing = false;
  state.firstTimestamp = null;
  state.pressureMin = 1;
  state.pressureMax = 0;
  $("pointCount").textContent = "0";
  $("elapsedTime").textContent = "0.0";
  $("sampleRate").textContent = "0";
  $("pressureRange").textContent = "—";
  $("inputType").textContent = "—";
  $("canvasHint").classList.remove("hidden");
  hideMessage($("trialMessage"));
  renderCanvas();
}

function setBlinking(enabled) {
  if (state.blinkTimer) window.clearInterval(state.blinkTimer);
  state.blinkTimer = null;
  state.templateVisible = true;
  if (enabled) {
    state.blinkTimer = window.setInterval(() => {
      state.templateVisible = !state.templateVisible;
      renderCanvas();
    }, 900);
  }
}

function showCurrentStep() {
  const step = state.steps[state.currentStep];
  renderProtocol();
  $("trialEyebrow").textContent = `Trial ${state.currentStep + 1} of ${state.steps.length}`;
  $("trialTitle").textContent = `${titleCase(step.hand)} hand · ${titleCase(step.mode)} spiral`;
  const dynamic = step.mode === "dynamic";
  $("modeBadge").textContent = dynamic ? "Guide blinking" : "Guide visible";
  $("trialInstruction").textContent = dynamic
    ? "The guide will blink. Continue tracing naturally from the center even while it is hidden."
    : "Keep the pen on the surface and follow the visible guide from the center outward.";
  setBlinking(dynamic);
  resetTrial();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function recordEvent(event) {
  if (state.firstTimestamp === null) state.firstTimestamp = event.timeStamp;
  const point = canvasPoint(event);
  const pointerType = ["pen", "touch", "mouse"].includes(event.pointerType)
    ? event.pointerType
    : "unknown";
  state.points.push({
    ...point,
    t: Math.max(0, event.timeStamp - state.firstTimestamp),
    pressure: Math.max(0, Math.min(1, event.pressure || 0)),
    tilt_x: event.tiltX || 0,
    tilt_y: event.tiltY || 0,
    pointer_type: pointerType,
    stroke: state.stroke,
  });
  state.pressureMin = Math.min(state.pressureMin, state.points.at(-1).pressure);
  state.pressureMax = Math.max(state.pressureMax, state.points.at(-1).pressure);
  $("pointCount").textContent = state.points.length.toLocaleString();
  $("elapsedTime").textContent = (state.points.at(-1).t / 1000).toFixed(1);
  const elapsedSeconds = state.points.at(-1).t / 1000;
  $("sampleRate").textContent = elapsedSeconds > 0
    ? Math.round(state.points.length / elapsedSeconds).toString()
    : "0";
  $("pressureRange").textContent = `${state.pressureMin.toFixed(2)}–${state.pressureMax.toFixed(2)}`;
  $("inputType").textContent = pointerType;
  $("canvasHint").classList.add("hidden");
  if (pointerType === "pen") {
    $("deviceStatus").classList.add("pen-ready");
    $("deviceStatus").lastChild.textContent = " Pen detected";
  }
}

canvas.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  state.stroke += 1;
  state.drawing = true;
  canvas.setPointerCapture(event.pointerId);
  recordEvent(event);
  renderCanvas();
});

canvas.addEventListener("pointermove", (event) => {
  if (!state.drawing) return;
  event.preventDefault();
  const events = typeof event.getCoalescedEvents === "function" ? event.getCoalescedEvents() : [event];
  for (const sample of events) recordEvent(sample);
  renderCanvas();
});

function finishStroke(event) {
  if (!state.drawing) return;
  event.preventDefault();
  recordEvent(event);
  state.drawing = false;
  renderCanvas();
}

canvas.addEventListener("pointerup", finishStroke);
canvas.addEventListener("pointercancel", finishStroke);

$("sessionForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  hideMessage($("setupMessage"));
  const submit = event.submitter;
  submit.disabled = true;
  try {
    const handedness = $("handedness").value;
    const session = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        participant_code: $("participantCode").value,
        handedness,
        age_band: $("ageBand").value,
        medication_state: $("medicationState").value,
        consent_research: $("consent").checked,
      }),
    });
    state.sessionId = session.id;
    state.steps = buildSteps(handedness);
    state.currentStep = 0;
    $("setupCard").hidden = true;
    $("captureArea").hidden = false;
    showCurrentStep();
    $("captureArea").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showMessage($("setupMessage"), error.message);
  } finally {
    submit.disabled = false;
  }
});

$("clearTrial").addEventListener("click", resetTrial);

$("saveTrial").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  hideMessage($("trialMessage"));
  if (state.points.length < 20) {
    showMessage($("trialMessage"), "Draw the complete spiral before saving this trial.");
    return;
  }
  button.disabled = true;
  const step = state.steps[state.currentStep];
  try {
    const trial = await api(`/api/sessions/${state.sessionId}/trials`, {
      method: "POST",
      body: JSON.stringify({
        ...step,
        canvas_width: canvas.width,
        canvas_height: canvas.height,
        points: state.points,
      }),
    });
    if (!trial.quality.valid) {
      showMessage($("trialMessage"), trial.quality.errors.join(" "));
      return;
    }
    if (trial.quality.warnings.length) {
      showMessage($("trialMessage"), trial.quality.warnings.join(" "));
    }
    state.currentStep += 1;
    if (state.currentStep < state.steps.length) {
      showCurrentStep();
      return;
    }
    setBlinking(false);
    await finishSpiralProtocol();
  } catch (error) {
    showMessage($("trialMessage"), error.message);
  } finally {
    button.disabled = false;
  }
});

async function finishSpiralProtocol() {
  try {
    state.spiralResult = await api(`/api/sessions/${state.sessionId}/score`, { method: "POST" });
    $("captureArea").hidden = true;
    $("voiceCard").hidden = false;
    $("voiceCard").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showMessage($("trialMessage"), error.message);
  }
}

async function uploadVoice(blob, repetition) {
  const response = await fetch(
    `/api/sessions/${state.sessionId}/voice?repetition=${repetition}`,
    {
      method: "POST",
      headers: { "Content-Type": blob.type || "application/octet-stream" },
      body: blob,
    },
  );
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    } catch (_) {
      // Keep fallback message.
    }
    throw new Error(detail);
  }
  return response.json();
}

function showResults(voiceResult = null) {
  const result = state.spiralResult;
  const displayScore = Math.round(result.experimental_screening_score * 100);
  $("scoreValue").textContent = displayScore;
  $("scoreFill").style.width = `${displayScore}%`;
  $("signalLabel").textContent = `${titleCase(result.pattern_signal)} pattern signal`;

  if (
    voiceResult
    && (voiceResult.status === "unscorable" || voiceResult.experimental_voice_score == null)
  ) {
    $("voiceScoreValue").textContent = "—";
    $("voiceScoreFill").style.width = "0%";
    $("voiceSignalLabel").textContent = "Unable to score safely";
    $("agreementLabel").textContent = "Spiral result only — voice unavailable";
    $("resultText").textContent = result.pattern_signal === "elevated"
      ? "The dominant-hand spiral crossed the 75-point research boundary. The voice recordings were outside the training domain and were not assigned a number."
      : "The dominant-hand spiral remained below the 75-point research boundary. The voice recordings were outside the training domain and were not assigned a number.";
    showMessage(
      $("resultMessage"),
      "The voice recordings were not comparable enough to the training audio. No voice score was produced; this is safer than displaying an extrapolated result.",
    );
  } else if (voiceResult) {
    const voiceScore = Math.round(voiceResult.experimental_voice_score * 100);
    $("voiceScoreValue").textContent = voiceScore;
    $("voiceScoreFill").style.width = `${voiceScore}%`;
    $("voiceSignalLabel").textContent = `${titleCase(voiceResult.pattern_signal)} pattern signal`;
    if (voiceResult.pattern_signal === result.pattern_signal) {
      $("agreementLabel").textContent = "The two experimental signals agree";
      $("resultText").textContent = result.pattern_signal === "elevated"
        ? "Both captures are more similar to Parkinson’s-labeled patterns in their separate research cohorts. This warrants clinical discussion if there are symptoms, but it is not a diagnosis."
        : "Both captures are less similar to Parkinson’s-labeled patterns in their separate research cohorts. This cannot rule out Parkinson’s or another condition.";
    } else {
      $("agreementLabel").textContent = "The experimental signals are mixed";
      $("resultText").textContent =
        "The movement and voice models do not agree. Do not average them into a diagnosis; recording conditions, normal variation, or other conditions can affect either test.";
    }
  } else {
    $("voiceScoreValue").textContent = "—";
    $("voiceSignalLabel").textContent = "Not completed";
    $("agreementLabel").textContent = "Spiral result only";
    $("resultText").textContent = result.pattern_signal === "elevated"
      ? "The movement crossed the 75-point research decision boundary. This is a reason to discuss symptoms with a clinician, not a diagnosis or confidence percentage."
      : "The movement remained below the 75-point research decision boundary. This cannot rule out Parkinson’s or another movement disorder.";
  }

  $("downloadReport").href = `/api/sessions/${state.sessionId}/report.pdf`;
  $("voiceCard").hidden = true;
  $("resultCard").hidden = false;
  $("resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

$("recordVoice").addEventListener("click", async () => {
  const button = $("recordVoice");
  hideMessage($("voiceMessage"));
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    showMessage($("voiceMessage"), "This browser does not support local microphone recording.");
    return;
  }
  button.disabled = true;
  $("skipVoice").disabled = true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    const preferred = "audio/webm;codecs=opus";
    const options = MediaRecorder.isTypeSupported(preferred) ? { mimeType: preferred } : {};
    const recorder = new MediaRecorder(stream, options);
    const chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    const stopped = new Promise((resolve, reject) => {
      recorder.addEventListener("stop", resolve, { once: true });
      recorder.addEventListener("error", () => reject(new Error("Microphone recording failed.")), { once: true });
    });
    recorder.start(250);
    $("micOrb").classList.add("recording");
    const started = performance.now();
    const timer = window.setInterval(() => {
      const remaining = Math.max(0, 6 - (performance.now() - started) / 1000);
      $("recordingTime").textContent = `${remaining.toFixed(1)} seconds`;
    }, 100);
    await new Promise((resolve) => window.setTimeout(resolve, 6000));
    recorder.stop();
    await stopped;
    window.clearInterval(timer);
    $("micOrb").classList.remove("recording");
    $("recordingTime").textContent = "Analyzing…";
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    const trial = await uploadVoice(blob, state.voiceRepetition);
    if (!trial.quality.valid) {
      showMessage($("voiceMessage"), trial.quality.errors.join(" "));
      $("recordingTime").textContent = "Retry this recording";
      return;
    }
    const warning = trial.quality.warnings.join(" ");
    state.voiceRepetition += 1;
    if (state.voiceRepetition <= 3) {
      $("voiceProgress").textContent = `Recording ${state.voiceRepetition} of 3`;
      $("recordingTime").textContent = "Ready";
      showMessage(
        $("voiceMessage"),
        warning || "Recording accepted. Take a normal breath before the next one.",
        warning ? "error" : "success",
      );
      return;
    }
    const voiceResult = await api(`/api/sessions/${state.sessionId}/voice-score`, { method: "POST" });
    showResults(voiceResult);
  } catch (error) {
    showMessage($("voiceMessage"), error.message);
    $("recordingTime").textContent = "Ready to retry";
    $("micOrb").classList.remove("recording");
  } finally {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    button.disabled = false;
    $("skipVoice").disabled = false;
  }
});

$("skipVoice").addEventListener("click", () => showResults());

$("retryVoice").addEventListener("click", () => {
  state.voiceRepetition = 1;
  $("voiceProgress").textContent = "Recording 1 of 3";
  $("recordingTime").textContent = "Ready";
  hideMessage($("voiceMessage"));
  hideMessage($("resultMessage"));
  $("resultCard").hidden = true;
  $("voiceCard").hidden = false;
  $("voiceCard").scrollIntoView({ behavior: "smooth", block: "start" });
});

$("deleteSession").addEventListener("click", async () => {
  if (!window.confirm("Permanently delete this local session and all raw drawing points?")) return;
  try {
    await api(`/api/sessions/${state.sessionId}`, { method: "DELETE" });
    window.location.reload();
  } catch (error) {
    showMessage($("resultMessage"), error.message);
  }
});

$("newSession").addEventListener("click", () => window.location.reload());

async function initialize() {
  renderCanvas();
  try {
    const health = await api("/api/health");
    $("modelState").textContent = health.model_ready && health.voice_model_ready
      ? "Motor + voice models ready · data stays local"
      : "One or more models are missing";
  } catch (error) {
    $("modelState").textContent = "Local service unavailable";
  }
}

initialize();
