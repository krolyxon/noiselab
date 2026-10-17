// AudioWorklet processor for the live denoise preview (stretch goal).
// Runs the BASIC algorithm (ss_core.js) on the microphone in real time:
// 128-sample render quanta are collected into hops of H = N/2, each hop goes through
// SpectralSubtractor.processHop, and the output is read from a FIFO primed with H zeros.
// Total algorithmic latency: N samples (≈ 10.7 ms for N = 512 at 48 kHz).

import { SpectralSubtractor } from "./dsp/ss_core.js";

class NoiseLabLive extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions;
    this.core = new SpectralSubtractor({ N: o.N, alpha: o.alpha, beta: o.beta, exponent: o.exponent });
    this.H = o.N / 2;
    this.hop = new Float64Array(this.H);
    this.fill = 0;
    // output FIFO (ring buffer), primed with H zeros so it never underruns
    this.ring = new Float32Array(this.H * 8);
    this.r = 0;
    this.w = this.H;
    if (o.noiseMag) this.core.setNoise(Float64Array.from(o.noiseMag));
    else this.core.startLearning(o.noiseSeconds, sampleRate);   // `sampleRate` is a worklet global
    this.tick = 0;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "params") this.core.setParams(m);
      else if (m.type === "bypass") this.core.bypass = !!m.on;
      else if (m.type === "relearn") this.core.startLearning(m.noiseSeconds, sampleRate);
      else if (m.type === "noise") this.core.setNoise(Float64Array.from(m.noiseMag));
    };
  }

  process(inputs, outputs) {
    const inp = inputs[0] && inputs[0][0];
    const out = outputs[0];
    const n = out[0].length;
    const size = this.ring.length;
    for (let i = 0; i < n; i++) {
      this.hop[this.fill++] = inp ? inp[i] : 0;
      if (this.fill === this.H) {
        const y = this.core.processHop(this.hop);
        for (let j = 0; j < this.H; j++) { this.ring[this.w] = y[j]; this.w = (this.w + 1) % size; }
        this.fill = 0;
      }
    }
    for (let i = 0; i < n; i++) {
      const v = this.ring[this.r];
      this.r = (this.r + 1) % size;
      for (let c = 0; c < out.length; c++) out[c][i] = v;
    }
    if (++this.tick % 16 === 0) {      // ≈ 23 status messages per second at 48 kHz
      this.port.postMessage({ learning: this.core.learning, progress: this.core.learnProgress,
        bypass: this.core.bypass });
    }
    return true;
  }
}

registerProcessor("noiselab-live", NoiseLabLive);
