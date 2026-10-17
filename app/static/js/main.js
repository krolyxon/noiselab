/* global Plotly */
// App wiring: upload → process → plots/metrics/player; parameter changes → debounced reprocess.

import { Controls } from "./controls.js";
import { renderMetrics } from "./metrics.js";
import { ABPlayer } from "./player.js";
import { colors, drawInspector, drawSpectra, Plots } from "./plots.js";
import { drawLive, micSupport, Recorder } from "./recorder.js";
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
  // dim the plots (no layout change) while a newer result is being computed
  document.querySelector(".col-centre").classList.toggle("busy-plots", state.busy > 0 && !!state.job);
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
  gain: $("#plot-gain"),
  energy: $("#plot-energy"),
  onSeek: (t) => { player.seek(t); inspector.inspect(t); },
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
  onState: ({ playing, source, ready, available }) => {
    $("#player").classList.toggle("playing", playing);
    $("#btn-play").disabled = !ready;
    for (const b of $$(".ab button")) {
      b.disabled = !ready || !available.includes(b.dataset.src);
      if (b.dataset.src === "clean") b.hidden = !available.includes("clean");
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
  drawSpectra($("#plot-spectra"), res.viz.spectra);
  $("#noise-mode-label").textContent = res.params.noise_mode;
  $("#demo-info").textContent = res.demo
    ? `${res.demo.source} · ${res.demo.noise_type} · ${res.demo.snr_db} dB · ${res.demo.duration} s · seed ${res.demo.seed}`
    : "";
  inspector.refresh(res, { newJob });
  $("#btn-export").disabled = false;
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

async function runDemo(options, cleanFile = null) {
  setBusy(true, "generating demo");
  try {
    let res;
    if (cleanFile) {
      const fd = new FormData();
      fd.append("clean", cleanFile, cleanFile.name);
      fd.append("options", JSON.stringify({ ...options, params: controls.request() }));
      res = await api("/api/demo", { method: "POST", body: fd });
    } else {
      res = await postJSON("/api/demo", { ...options, params: controls.request() });
    }
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
    if (document.querySelector("dialog[open]")) return;     // Esc is handled by <dialog> itself
    const tag = e.target.tagName;
    if (tag === "INPUT" && e.target.type !== "range" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (e.code === "Space") { e.preventDefault(); player.toggle(); }
    else if (e.key === "a" || e.key === "A") player.cycle();
    else if (e.key === "r" || e.key === "R") toggleRecord();
    else if (e.key === "?") openDialog("#dlg-help");
    else if (e.key === ",") inspector.step(-1);
    else if (e.key === ".") inspector.step(1);
  });
}

// ---------------------------------------------------------------------------- frame inspector
const inspector = {
  t: null, seq: 0, hopS: null, frame: null,

  refresh(res, { newJob }) {
    this.hopS = (res.params.n_fft * (1 - res.params.overlap)) / res.sr;
    this.duration = res.duration;
    // A new clip starts on a frame inside the speech (first VAD-speech frame + 0.25 s), else the middle.
    if (newJob || this.t === null) {
      const en = res.viz.energy, i = en.speech.findIndex((v) => v > 0);
      this.t = i >= 0 ? Math.min(en.t[i] + 0.25, res.duration) : res.duration / 2;
    }
    this.inspect(this.t);
  },

  inspect: debounce(async function (t) {
    if (!state.job) return;
    inspector.t = t;
    const seq = ++inspector.seq;
    try {
      const fr = await api(`/api/frame/${state.job.job_id}?t=${t.toFixed(4)}`);
      if (seq !== inspector.seq) return;
      inspector.frame = fr;
      inspector.t = fr.t;
      $("#insp-label").textContent = `frame ${fr.frame + 1}/${fr.n_frames} · t = ${fr.t.toFixed(3)} s`;
      $("#insp-prev").disabled = fr.frame <= 0;
      $("#insp-next").disabled = fr.frame >= fr.n_frames - 1;
      drawInspector($("#plot-insp"), fr);
    } catch (e) {
      if (seq === inspector.seq) toast(`Frame inspector: ${e.message}`, "error");
    }
  }, 60),

  step(dir) {
    if (this.frame === null || this.hopS === null) return;
    const t = Math.max(0, Math.min(this.duration, this.t + dir * this.hopS));
    player.seek(t);
    this.inspect(t);
  },
};
$("#insp-prev").addEventListener("click", () => inspector.step(-1));
$("#insp-next").addEventListener("click", () => inspector.step(1));

// ---------------------------------------------------------------------------- demo panel
let demoClean = null;
function wireDemo() {
  const snr = $("#demo-snr"), dur = $("#demo-dur");
  const fill = (inp) => inp.style.setProperty("--fill", `${((inp.value - inp.min) / (inp.max - inp.min)) * 100}%`);
  const sync = () => {
    $("#demo-snr-out").textContent = `${(+snr.value).toFixed(0).replace("-", "−")} dB`;
    $("#demo-dur-out").textContent = `${dur.value} s`;
    fill(snr); fill(dur);
  };
  snr.addEventListener("input", sync);
  dur.addEventListener("input", sync);
  sync();
  $("#demo-clean").addEventListener("change", (e) => {
    demoClean = e.target.files[0] ?? null;
    e.target.value = "";
    $("#demo-clean-name").textContent = demoClean ? `Clean: ${demoClean.name}` : "Synthetic voiced signal (or choose a clean WAV…)";
    $("#demo-clean-clear").hidden = !demoClean;
    dur.disabled = !!demoClean;     // a user clean file sets its own duration
  });
  $("#demo-clean-clear").addEventListener("click", () => {
    demoClean = null;
    $("#demo-clean-name").textContent = "Synthetic voiced signal (or choose a clean WAV…)";
    $("#demo-clean-clear").hidden = true;
    dur.disabled = false;
  });
  $("#btn-demo").addEventListener("click", () => {
    const seed = Math.max(0, Math.floor(+$("#demo-seed").value || 0));
    const opts = { noise_type: $("#demo-noise").value, snr_db: +snr.value, seed };
    if (!demoClean) opts.duration = +dur.value;
    runDemo(opts, demoClean);
  });
}

// ---------------------------------------------------------------------------- dialogs, theme, export
function openDialog(sel) {
  const d = $(sel);
  if (d.open) return;
  if (sel === "#dlg-method") renderEquations(d);
  d.showModal();
}
for (const d of $$("dialog")) {
  d.addEventListener("click", (e) => { if (e.target === d || e.target.closest("[data-close]")) d.close(); });
}

let eqRendered = false;
function renderEquations(root) {
  if (eqRendered) return;
  if (!window.katex) {   // vendored script missing: show the TeX source rather than nothing
    for (const n of $$("[data-tex]", root)) n.textContent = n.dataset.tex;
    return;
  }
  for (const n of $$("[data-tex]", root)) {
    window.katex.render(n.dataset.tex, n, { displayMode: n.classList.contains("eq"), throwOnError: false });
  }
  eqRendered = true;
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("noiselab-theme", next); } catch { /* storage unavailable: theme lasts this visit */ }
  if (state.job) {                     // plot colours are read from CSS at draw time
    plots.update(state.job, { newJob: false });
    drawSpectra($("#plot-spectra"), state.job.viz.spectra);
    if (inspector.frame) drawInspector($("#plot-insp"), inspector.frame);
  }
}

// name → [selector, title for the PNG, show a legend in the PNG?]
const EXPORT_PLOTS = {
  waveform: ["#plot-wave", "Waveform: noisy (amber) vs cleaned (cyan)", false],
  spectrogram_noisy: ["#plot-spec-noisy", "Spectrogram, noisy |Y| (dBFS)", false],
  spectrogram_cleaned: ["#plot-spec-cleaned", "Spectrogram, cleaned |Ŝ| (dBFS)", false],
  spectrogram_removed: ["#plot-spec-removed", "Spectrogram, removed |Y − Ŝ| (dBFS)", false],
  frame_energy_vad: ["#plot-energy", "Frame energy with energy-VAD speech frames (Eq. 20)", true],
  gain_mask: ["#plot-gain", "Gain mask G(k, m), Eq. 8", false],
  frame_inspector: ["#plot-insp", "STFT frame inspector", true],
  noise_spectra: ["#plot-spectra", "Noise estimate |N̂| and mean spectra", true],
};

async function exportZip() {
  if (!state.job) return;
  setBusy(true, "exporting");
  const bg = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim();
  const plotsPng = {};
  try {
    const fg = getComputedStyle(document.documentElement).getPropertyValue("--text").trim();
    for (const [name, [sel, title, legend]] of Object.entries(EXPORT_PLOTS)) {
      const div = $(sel);
      if (!div.data) continue;
      let titleText = title;
      if (name === "frame_inspector" && inspector.frame) titleText += ` (frame ${inspector.frame.frame + 1}, t = ${inspector.frame.t.toFixed(3)} s)`;
      // Opaque background, a title and (for multi-curve plots) a legend, so each PNG stands alone in a report.
      const layout = {
        ...div.layout, paper_bgcolor: bg, plot_bgcolor: bg,
        title: { text: titleText, font: { size: 13, color: fg }, x: 0.01, xanchor: "left" },
        margin: { ...div.layout.margin, t: legend ? 64 : 36 },
        showlegend: legend,
        legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: 10, color: fg } },
      };
      plotsPng[name] = await Plotly.toImage({ data: div.data, layout },
        { format: "png", width: Math.max(div.clientWidth, 720), height: Math.max(div.clientHeight, 260) + (legend ? 60 : 32), scale: 2 });
    }
  } catch (e) {
    toast(`Plot images skipped (${e.message}); exporting audio and data only.`, "error");
  }
  try {
    const res = await fetch(`/api/export/${state.job.job_id}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ plots: plotsPng }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => null))?.error || `${res.status}`);
    const blob = await res.blob();
    const name = /filename="?([^";]+)"?/.exec(res.headers.get("Content-Disposition") || "")?.[1] || "noiselab-export.zip";
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    toast(`Exported ${name}`, "ok");
  } catch (e) {
    toast(`Export failed: ${e.message}`, "error");
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------- recording
const PROFILE_SECONDS = 2;
const recorder = new Recorder({
  onLevel: drawLevel,
  onTick: (t) => { $("#rec-time").textContent = clock(t); },
});

const clock = (t) => {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
};
const dbToPct = (db) => Math.max(0, Math.min(100, ((db + 60) / 60) * 100));   // meter spans −60…0 dBFS

let clipUntil = 0;
function drawLevel({ rmsDb, peakDb, clip, spectrum, scope, sr }) {
  $("#meter-bar").style.width = `${100 - dbToPct(rmsDb)}%`;
  $("#meter-peak").style.left = `calc(${dbToPct(peakDb)}% - 2px)`;
  $("#meter").setAttribute("aria-valuenow", rmsDb.toFixed(1));
  const now = performance.now();
  if (clip) clipUntil = now + 1500;                  // hold the CLIP light for 1.5 s
  $("#meter").classList.toggle("clip", now < clipUntil);
  drawLive($("#live"), { spectrum, scope, sr }, colors());
}

function setRecUI(on, mode = null) {
  const clipMode = on && mode === "clip", profMode = on && mode === "profile";
  $("#rec").classList.toggle("recording", on);
  $("#btn-rec").classList.toggle("on", clipMode);
  $("#btn-rec-label").textContent = clipMode ? "Stop" : "Record";
  $("#btn-rec").disabled = profMode;
  $("#btn-profile").disabled = clipMode || profMode;
  $("#btn-profile").classList.toggle("on", profMode);
  $("#btn-profile").textContent = profMode ? "Listening — stay quiet…" : `Capture noise profile (${PROFILE_SECONDS} s)`;
  $("#rec-hint").textContent = profMode
    ? "Recording room noise only. Don’t speak."
    : clipMode ? "Recording… press Stop (or R) to process."
      : "Speak after ~1 s of silence: the first 0.5 s is used as the noise estimate.";
  if (!on) {
    $("#meter-bar").style.width = "100%";
    $("#meter-peak").style.left = "0";
    $("#meter").classList.remove("clip");
  }
}

async function toggleRecord() {
  if (recorder.state === "recording") { recorder.stop().catch(() => {}); return; }
  if (recorder.state === "idle") await beginRecording("clip");
}

async function beginRecording(mode) {
  const unsupported = micSupport();
  if (unsupported) { toast(unsupported, "error"); return; }
  player.pause();
  try {
    await recorder.start(mode === "profile" ? { seconds: PROFILE_SECONDS } : {});
  } catch (e) {
    toast(e.message, "error");
    return;
  }
  setRecUI(true, mode);
  recorder.finished
    .then((clip) => onRecorded(clip, mode))
    .catch((e) => toast(`Recording failed: ${e.message}`, "error"))
    .finally(() => setRecUI(false));
}

async function onRecorded(clip, mode) {
  if (!clip) return;
  const minS = mode === "profile" ? 0.5 : 0.3;
  if (clip.duration < minS) { toast(`Recording too short (${clip.duration.toFixed(2)} s).`, "error"); return; }
  const wav = encodeWav(clip.mono, clip.sr);           // 16-bit PCM WAV, encoded in the browser
  if (mode === "profile") {
    const fd = new FormData();
    fd.append("audio", wav, "noise-profile.wav");
    setBusy(true, "uploading profile");
    try {
      const r = await api("/api/noise-profile", { method: "POST", body: fd });
      setProfile(r);
      controls.set({ noise_mode: "profile" });          // reprocesses the current clip, if any
      toast(`Noise profile captured (${fmt(r.duration, 2, "s")}). Noise mode set to “Noise profile clip”.`, "ok");
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setBusy(false);
    }
    return;
  }
  const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
  $("#empty-state").hidden = true;
  plots.preview(envelope(clip.mono, clip.sr), clip.duration);
  setClipInfo({ name: "recording", sr: clip.sr, duration: clip.duration });
  await processUpload(wav, `recording-${stamp}.wav`).catch((e) => toast(e.message, "error"));
}

function setProfile(p) {
  controls.setProfile(p?.profile_id ?? null);
  $("#profile-status").hidden = !p;
  $("#profile-info").textContent = p ? `Profile: ${fmt(p.duration, 2, "s")} @ ${p.sr} Hz` : "";
}

// ---------------------------------------------------------------------------- boot
let controls;

async function boot() {
  wireDropzone();
  wireKeys();
  wireDemo();
  $("#btn-method").addEventListener("click", () => openDialog("#dlg-method"));
  $("#btn-help").addEventListener("click", () => openDialog("#dlg-help"));
  $("#btn-theme").addEventListener("click", toggleTheme);
  $("#btn-export").addEventListener("click", exportZip);
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

  recorder.maxSeconds = state.health.limits.max_duration_s;
  const unsupported = micSupport();
  if (unsupported) {
    $("#btn-rec").disabled = $("#btn-profile").disabled = true;
    $("#mic-unsupported").textContent = unsupported;
    $("#mic-unsupported").hidden = false;
  }
  $("#btn-rec").addEventListener("click", toggleRecord);
  $("#btn-profile").addEventListener("click", () => {
    if (recorder.state === "idle") beginRecording("profile");
  });
  $("#btn-profile-clear").addEventListener("click", () => {
    setProfile(null);
    if (controls.get().noise_mode === "profile") controls.set({ noise_mode: "initial" });
  });
}

boot();
