// Microphone capture: MediaRecorder for the recording, an AnalyserNode for the live
// level meter / spectrum / scope. On stop the clip is decoded in the browser
// (decodeAudioData) and returned as mono samples, ready for 16-bit WAV encoding.

import { decodeAudio, downmix } from "./wav.js";

// Ask the browser NOT to process the signal: its own noise suppression would denoise
// the recording before our algorithm sees it, and AGC would pump the noise floor.
export const CONSTRAINTS = {
  audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
};

export function micSupport() {
  if (!window.isSecureContext) {
    return "Microphone needs a secure context: open the app via http://localhost:5000 or HTTPS (not a LAN IP).";
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    return "This browser does not support microphone recording (getUserMedia / MediaRecorder).";
  }
  return null;
}

export class Recorder {
  constructor({ onLevel, onTick, maxSeconds = 120 }) {
    this.onLevel = onLevel;        // ({rmsDb, peakDb, clip, spectrum: Float32Array, scope: Float32Array, sr})
    this.onTick = onTick;          // (elapsedSeconds)
    this.maxSeconds = maxSeconds;
    this.state = "idle";           // idle | recording | stopping
  }

  /** Start recording. `seconds` (optional) auto-stops after that long and resolves stop(). */
  async start({ seconds = null } = {}) {
    if (this.state !== "idle") throw new Error("already recording");
    this.state = "recording";
    try {
      this.stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
    } catch (e) {
      this.state = "idle";
      throw new Error(e.name === "NotAllowedError" ? "microphone permission was denied"
        : e.name === "NotFoundError" ? "no microphone found" : `microphone error: ${e.message}`);
    }
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.6;
    src.connect(this.analyser);   // analysis only — not connected to the speakers (no feedback)

    this.chunks = [];
    this.rec = new MediaRecorder(this.stream);
    this.rec.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
    this._stopped = new Promise((resolve) => { this.rec.onstop = resolve; });
    this.rec.start(250);
    this.t0 = performance.now();
    this.limit = Math.min(seconds ?? this.maxSeconds, this.maxSeconds);
    this._stopPromise = null;
    // Resolves with the decoded clip however recording ends (button, key or time limit).
    this.finished = new Promise((resolve, reject) => { this._finish = { resolve, reject }; });
    this._loop();
    return this;
  }

  get elapsed() { return this.t0 ? (performance.now() - this.t0) / 1000 : 0; }

  /** Stop (idempotent) and return {mono: Float32Array, sr, duration}. */
  stop() {
    if (this.state === "idle" && !this._stopPromise) return Promise.resolve(null);
    this._stopPromise ??= this._doStop().then(
      (r) => { this._finish.resolve(r); return r; },
      (e) => { this._finish.reject(e); throw e; });
    return this._stopPromise;
  }

  async _doStop() {
    this.state = "stopping";
    cancelAnimationFrame(this._raf);
    this.rec.stop();
    await this._stopped;
    this.stream.getTracks().forEach((t) => t.stop());
    this.ctx.close();
    const blob = new Blob(this.chunks, { type: this.rec.mimeType });
    this.state = "idle";
    this.t0 = null;
    if (!blob.size) throw new Error("the recording is empty");
    const buf = await decodeAudio(await blob.arrayBuffer());
    return { mono: downmix(buf), sr: buf.sampleRate, duration: buf.duration };
  }

  _loop() {
    const a = this.analyser;
    const scope = new Float32Array(a.fftSize);
    const spectrum = new Float32Array(a.frequencyBinCount);
    let peakHold = -Infinity, peakT = 0;
    const step = () => {
      if (this.state !== "recording") return;
      a.getFloatTimeDomainData(scope);
      a.getFloatFrequencyData(spectrum);
      let sum = 0, peak = 0;
      for (const s of scope) { sum += s * s; const v = Math.abs(s); if (v > peak) peak = v; }
      const rmsDb = 10 * Math.log10(sum / scope.length + 1e-12);
      const peakDb = 20 * Math.log10(peak + 1e-12);
      const now = performance.now();
      if (peakDb > peakHold || now - peakT > 1500) { peakHold = peakDb; peakT = now; }   // 1.5 s peak hold
      this.onLevel?.({ rmsDb, peakDb: peakHold, clip: peak >= 0.99, spectrum, scope, sr: this.ctx.sampleRate });
      this.onTick?.(this.elapsed);
      if (this.elapsed >= this.limit) {
        this.stop().catch(() => {});   // error is delivered via `finished`
        return;
      }
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }
}

/** Draw the live view: log-frequency spectrum (filled) + oscilloscope trace. */
export function drawLive(canvas, { spectrum, scope, sr }, colors) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  // grid: 100 Hz, 1 kHz, 10 kHz lines (log axis 50 Hz … sr/2)
  const fmin = 50, fmax = sr / 2;
  const fx = (f) => (Math.log(f / fmin) / Math.log(fmax / fmin)) * w;
  g.strokeStyle = colors.grid; g.lineWidth = 1; g.fillStyle = colors.muted; g.font = "9px JetBrains Mono, monospace";
  for (const [f, label] of [[100, "100"], [1000, "1k"], [10000, "10k"]]) {
    if (f >= fmax) continue;
    const x = Math.round(fx(f)) + 0.5;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    g.fillText(label, x + 2, h - 3);
  }

  // spectrum (dBFS −110 … −10), amber = noisy input
  const db2y = (d) => h - ((Math.max(-110, Math.min(-10, d)) + 110) / 100) * h;
  const binHz = sr / 2 / spectrum.length;
  g.beginPath(); g.moveTo(0, h);
  for (let i = 1; i < spectrum.length; i++) {
    const f = i * binHz;
    if (f < fmin) continue;
    g.lineTo(fx(f), db2y(spectrum[i]));
  }
  g.lineTo(w, h); g.closePath();
  g.fillStyle = colors.noisy + "40"; g.fill();
  g.strokeStyle = colors.noisy; g.lineWidth = 1; g.stroke();

  // oscilloscope (last 2048 samples), white, centred
  g.beginPath();
  for (let i = 0; i < scope.length; i++) {
    const x = (i / (scope.length - 1)) * w, y = h / 2 - scope[i] * (h / 2) * 0.9;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.strokeStyle = colors.text + "99"; g.lineWidth = 1; g.stroke();
}
