// App wiring: upload → process → plots/metrics/player; parameter changes → debounced reprocess.

import { Controls } from "./controls.js";
import { renderMetrics } from "./metrics.js";
import { ABPlayer } from "./player.js";
import { Plots } from "./plots.js";
import { $, $$, api, debounce, fmt, postJSON, toast } from "./util.js";
import { decodeAudio, downmix, encodeWav, envelope } from "./wav.js";

const REPROCESS_DEBOUNCE_MS = 250;
const SERVER_FORMATS = /\.(wav|flac|ogg|oga|mp3)$/i;

const state = {
  job: null,           // last server response
  health: null,
  busy: 0,
  reqSeq: 0,
  abort: null,
};

// ---------------------------------------------------------------------------- status bar
function setBusy(on, text = "processing") {
  state.busy += on ? 1 : -1;
  $("#st-busy").hidden = state.busy <= 0;
  $("#st-busy-text").textContent = text;
}

function setClipInfo({ name, sr, duration }) {
  $("#st-file").textContent = name ?? "no audio";
  $("#st-file").title = name ?? "";
  $("#st-sr").textContent = sr ? `${sr} Hz` : "— Hz";
  $("#st-dur").textContent = duration ? fmt(duration, 2, "s") : "—.— s";
  $("#t-total").textContent = duration ? duration.toFixed(3) : "0.000";
}

// ---------------------------------------------------------------------------- plots + player
const plots = new Plots({
  wave: $("#plot-wave"),
  specs: { noisy: $("#plot-spec-noisy"), cleaned: $("#plot-spec-cleaned"), removed: $("#plot-spec-removed") },
  onSeek: (t) => player.seek(t),
  onRegion: (a, b) => {
    if (b - a < 0.05) return;
    controls.set({ region_start: +a.toFixed(3), region_end: +b.toFixed(3) });
  },
});

const player = new ABPlayer({
  onTime: (t) => {
    $("#t-now").textContent = t.toFixed(3);
    plots.setPlayhead(t);
  },
  onState: ({ playing, source, ready }) => {
    $("#player").classList.toggle("playing", playing);
    $("#btn-play").disabled = !ready;
    for (const b of $$(".ab button")) {
      b.disabled = !ready;
      b.setAttribute("aria-checked", String(b.dataset.src === source));
    }
  },
});

$("#btn-play").addEventListener("click", () => player.toggle());
for (const b of $$(".ab button")) b.addEventListener("click", () => player.setSource(b.dataset.src));

// ---------------------------------------------------------------------------- results
function showResult(res, { newJob }) {
  state.job = res;
  $("#empty-state").hidden = true;
  setClipInfo({ name: res.name, sr: res.sr, duration: res.duration });
  controls.setContext({ sr: res.sr, duration: res.duration });
  plots.update(res, { newJob });
  updateRegionOverlay();
  renderMetrics($("#metrics"), res.metrics);
  const urls = newJob ? res.audio : { cleaned: res.audio.cleaned, noise: res.audio.noise };
  if (newJob) player.reset();
  player.load(urls).catch((e) => toast(e.message, "error"));
}

function updateRegionOverlay() {
  const v = controls?.get();
  if (!v || !state.job) return;
  const regionMode = v.noise_mode === "region";
  $("#region-hint").hidden = !regionMode;
  if (v.noise_mode === "initial") plots.setRegion({ x0: 0, x1: v.noise_seconds, editable: false });
  else if (regionMode) plots.setRegion({ x0: v.region_start, x1: v.region_end, editable: true });
  else plots.setRegion(null);
}

// ---------------------------------------------------------------------------- upload
async function handleFile(file) {
  if (!file) return;
  const maxMb = state.health?.limits?.max_upload_mb ?? 25;
  if (file.size > maxMb * 1024 * 1024) {
    toast(`${file.name} is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${maxMb} MB`, "error");
    return;
  }
  setBusy(true, "decoding");
  let decoded = null;
  try {
    // Instant waveform preview (and the fallback path for formats the server cannot read).
    decoded = await decodeAudio(await file.arrayBuffer());
    const mono = downmix(decoded);
    $("#empty-state").hidden = true;
    setClipInfo({ name: file.name, sr: decoded.sampleRate, duration: decoded.duration });
    plots.preview(envelope(mono, decoded.sampleRate), decoded.duration);
    decoded = { mono, sr: decoded.sampleRate };
  } catch {
    decoded = null;   // the browser cannot decode it; the server may still be able to
  } finally {
    setBusy(false);
  }

  // Upload the original bytes when the server can read the format (keeps the true sample rate);
  // otherwise upload a 16-bit PCM WAV encoded in the browser.
  let blob = file, name = file.name;
  if (!SERVER_FORMATS.test(file.name)) {
    if (!decoded) { toast(`Cannot decode ${file.name} in this browser or on the server.`, "error"); return; }
    blob = encodeWav(decoded.mono, decoded.sr);
    name = file.name.replace(/\.[^.]+$/, "") + ".wav";
  }
  try {
    await processUpload(blob, name);
  } catch (e) {
    if (e.status === 415 && decoded && blob === file) {
      toast("Server could not read this file; retrying with a browser-decoded WAV…");
      await processUpload(encodeWav(decoded.mono, decoded.sr), file.name.replace(/\.[^.]+$/, "") + ".wav")
        .catch((e2) => toast(e2.message, "error"));
    } else {
      toast(e.message, "error");
    }
  }
}

export async function processUpload(blob, name) {
  const fd = new FormData();
  fd.append("audio", blob, name);
  fd.append("params", JSON.stringify(controls.request()));
  setBusy(true, "processing");
  try {
    const res = await api("/api/process", { method: "POST", body: fd });
    showResult(res, { newJob: true });
    return res;
  } finally {
    setBusy(false);
  }
}

async function runDemo(options) {
  setBusy(true, "generating demo");
  try {
    const res = await postJSON("/api/demo", { ...options, params: controls.request() });
    showResult(res, { newJob: true });
  } catch (e) {
    toast(e.message, "error");
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------- reprocess
const reprocess = debounce(async () => {
  if (!state.job) return;
  const seq = ++state.reqSeq;
  state.abort?.abort();
  state.abort = new AbortController();
  setBusy(true, "reprocessing");
  try {
    const res = await postJSON(`/api/reprocess/${state.job.job_id}`, { params: controls.request() }, state.abort.signal);
    if (seq === state.reqSeq) showResult(res, { newJob: false });
  } catch (e) {
    if (e.name !== "AbortError") toast(e.message, "error");
  } finally {
    setBusy(false);
  }
}, REPROCESS_DEBOUNCE_MS);

// ---------------------------------------------------------------------------- drag & drop
function wireDropzone() {
  const dz = $("#dropzone"), input = $("#file-input");
  input.addEventListener("change", () => { handleFile(input.files[0]); input.value = ""; });
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  for (const ev of ["dragenter", "dragover"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) dz.addEventListener(ev, () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); });
  // Dropping anywhere on the page should not navigate away.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => { if (!dz.contains(e.target)) { e.preventDefault(); handleFile(e.dataTransfer.files[0]); } });
}

// ---------------------------------------------------------------------------- keyboard
function wireKeys() {
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = e.target.tagName;
    if (tag === "INPUT" && e.target.type !== "range" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (e.code === "Space") { e.preventDefault(); player.toggle(); }
    else if (e.key === "a" || e.key === "A") player.cycle();
  });
}

// ---------------------------------------------------------------------------- boot
let controls;

async function boot() {
  wireDropzone();
  wireKeys();
  try {
    const [health, presets] = await Promise.all([api("/api/health"), api("/api/presets")]);
    state.health = health;
    $("#formats-hint").textContent = `${health.formats.join(" · ")} · ≤\u00a0${health.limits.max_upload_mb}\u00a0MB · ≤\u00a0${health.limits.max_duration_s}\u00a0s`;
    controls = new Controls({
      container: $("#param-groups"), presetSelect: $("#preset"), presetDesc: $("#preset-desc"),
      schema: presets.schema, presets: presets.presets,
      onChange: () => { updateRegionOverlay(); reprocess(); },
    });
  } catch (e) {
    $("#param-groups").textContent = "";
    toast(`Could not load configuration: ${e.message}`, "error", 10000);
    return;
  }
  $("#btn-reset").addEventListener("click", () => controls.reset());
  $("#btn-simulink").addEventListener("click", () => {
    controls.applyPreset("match_simulink_basic");
    toast("Using the exact parameters of the basic Simulink model.", "ok");
  });
  $("#btn-demo-quick").addEventListener("click", () => runDemo({ noise_type: "white", snr_db: 5 }));
}

boot();
