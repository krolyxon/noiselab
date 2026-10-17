// Live denoise preview (stretch goal): mic → AudioWorklet (basic algorithm) → speakers/headphones.

import { CONSTRAINTS } from "./recorder.js";

export function liveSupport() {
  if (!window.isSecureContext) return "Live preview needs localhost or HTTPS (microphone access).";
  if (!window.AudioWorkletNode) return "This browser does not support AudioWorklet.";
  if (!navigator.mediaDevices?.getUserMedia) return "This browser does not support microphone input.";
  return null;
}

export class LivePreview {
  constructor({ onStatus, onLevel }) {
    this.onStatus = onStatus;     // ({learning, progress, bypass})
    this.onLevel = onLevel;       // same shape as Recorder.onLevel
    this.running = false;
  }

  /** opts: {N, alpha, beta, exponent, noiseSeconds, getNoiseMag: async (sr, N) => Float64Array|null} */
  async start(opts) {
    if (this.running) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
    try {
      await this.ctx.audioWorklet.addModule(new URL("./live-worklet.js", import.meta.url));
      const noiseMag = opts.getNoiseMag ? await opts.getNoiseMag(this.ctx.sampleRate, opts.N) : null;
      this.stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
      const src = this.ctx.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.ctx, "noiselab-live", {
        numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2],
        processorOptions: { N: opts.N, alpha: opts.alpha, beta: opts.beta, exponent: opts.exponent,
          noiseSeconds: opts.noiseSeconds, noiseMag: noiseMag ? Array.from(noiseMag) : null },
      });
      this.node.port.onmessage = (e) => this.onStatus?.(e.data);
      src.connect(this.node).connect(this.ctx.destination);
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.6;
      this.node.connect(this.analyser);            // show the *denoised* output in the live view
      if (this.ctx.state === "suspended") await this.ctx.resume();
    } catch (e) {
      this.stop();
      throw new Error(e.name === "NotAllowedError" ? "microphone permission was denied" : e.message);
    }
    this.running = true;
    this.fromProfile = !!opts.getNoiseMag;
    this._loop();
  }

  setParams({ alpha, beta, exponent }) { this.node?.port.postMessage({ type: "params", alpha, beta, exponent }); }
  setBypass(on) { this.node?.port.postMessage({ type: "bypass", on }); }
  relearn(noiseSeconds) { this.node?.port.postMessage({ type: "relearn", noiseSeconds }); }

  stop() {
    cancelAnimationFrame(this._raf);
    this.stream?.getTracks().forEach((t) => t.stop());
    if (this.node) this.node.port.onmessage = null;
    this.node?.disconnect();
    this.ctx?.close();
    this.stream = this.node = this.ctx = this.analyser = null;
    this.running = false;
  }

  _loop() {
    const a = this.analyser;
    const scope = new Float32Array(a.fftSize), spectrum = new Float32Array(a.frequencyBinCount);
    const step = () => {
      if (!this.running || !this.analyser) return;
      a.getFloatTimeDomainData(scope);
      a.getFloatFrequencyData(spectrum);
      let sum = 0, peak = 0;
      for (const s of scope) { sum += s * s; const v = Math.abs(s); if (v > peak) peak = v; }
      this.onLevel?.({ rmsDb: 10 * Math.log10(sum / scope.length + 1e-12), peakDb: 20 * Math.log10(peak + 1e-12),
        clip: peak >= 0.99, spectrum, scope, sr: this.ctx.sampleRate });
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }
}
