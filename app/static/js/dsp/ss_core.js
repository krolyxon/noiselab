// Real-time core of the BASIC algorithm, hop by hop: the same block chain as the Simulink
// model (docs/ALGORITHM.md, simulink/README.md). It is pure (no DOM / Web Audio), so it runs
// inside the AudioWorklet and under Node for the JS ⇄ Python parity test
// (tests/test_js_parity.py).
//
//   per hop of H = N/2 samples:
//     buffer ← [buffer(H:N), hop]                       Buffer block          Eq. (1)
//     Y = FFT(w · buffer)                               window + FFT          Eq. (2)
//     |S̃|^p = max(|Y|^p − α|N̂|^p, β|N̂|^p),  G = |S̃|/|Y|                    Eq. (7)–(8)
//     frame = Re IFFT(G · Y)                            noisy phase           Eq. (12)
//     out = frame(0:H) + previous frame(H:N)            overlap-add (delay)   Eq. (13)
//
// Only periodic Hann at 50 % overlap (its windows sum to 1, so no normalisation is needed).
// The extended options (Wiener rule, smoothing) are NOT implemented here.

const EPS = 1e-12;

/** w[n] = 0.5 − 0.5·cos(2πn/N), n = 0…N−1 (periodic Hann). */
export function hannPeriodic(N) {
  const w = new Float64Array(N);
  for (let n = 0; n < N; n++) w[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N);
  return w;
}

/** In-place iterative radix-2 FFT (N a power of two). inverse=true computes N·IFFT. */
export function fftInPlace(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {             // bit-reversal permutation
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (sign * 2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

/**
 * Noise magnitude from the first T seconds, Eq. (4): mean |FFT| over the complete frames
 * starting at 0, H, 2H, … that end by max(T·sr, N). One-sided (N/2 + 1 bins).
 * Same frames as Python's estimate_initial and MATLAB's nl_compute_noise_mag.
 */
export function computeNoiseMag(x, N, noiseSeconds, sr) {
  const H = N / 2, w = hannPeriodic(N), bins = N / 2 + 1;
  const last = Math.min(Math.max(Math.round(noiseSeconds * sr), N), x.length);
  const acc = new Float64Array(bins);
  const re = new Float64Array(N), im = new Float64Array(N);
  let frames = 0;
  for (let s = 0; s + N <= last; s += H) {
    for (let n = 0; n < N; n++) { re[n] = x[s + n] * w[n]; im[n] = 0; }
    fftInPlace(re, im);
    for (let k = 0; k < bins; k++) acc[k] += Math.hypot(re[k], im[k]);
    frames++;
  }
  if (!frames) throw new Error("signal shorter than one frame");
  for (let k = 0; k < bins; k++) acc[k] /= frames;
  return acc;
}

export class SpectralSubtractor {
  constructor({ N = 512, alpha = 2, beta = 0.02, exponent = 1 } = {}) {
    if (!(N >= 64 && (N & (N - 1)) === 0)) throw new Error("N must be a power of two ≥ 64");
    this.N = N;
    this.H = N / 2;
    this.bins = N / 2 + 1;
    this.w = hannPeriodic(N);
    this.buf = new Float64Array(N);          // Buffer block state (initial conditions 0)
    this.prev = new Float64Array(this.H);    // Unit Delay state
    this.re = new Float64Array(N);
    this.im = new Float64Array(N);
    this.noise = null;                       // one-sided |N̂|; null → pass-through (G = 1)
    this.bypass = false;
    this.setParams({ alpha, beta, exponent });
    this.learn = null;                       // {acc, frames, framesNeeded, k}
    this.frameIndex = 0;
  }

  setParams({ alpha, beta, exponent }) {
    if (alpha !== undefined) this.alpha = alpha;
    if (beta !== undefined) this.beta = beta;
    if (exponent !== undefined) this.p = exponent;
  }

  setNoise(mag) {
    if (mag.length !== this.bins) throw new Error(`noise vector must have ${this.bins} bins`);
    this.noise = Float64Array.from(mag);
    this.learn = null;
  }

  /** Learn |N̂| from the frames inside the next T seconds of input (live `initial` mode). */
  startLearning(noiseSeconds, sr) {
    const lastSample = Math.max(Math.round(noiseSeconds * sr), this.N);
    // frames start at stream samples 0, H, … and must end by lastSample
    const framesNeeded = Math.floor((lastSample - this.N) / this.H) + 1;
    this.noise = null;
    this.learn = { acc: new Float64Array(this.bins), frames: 0, framesNeeded, startFrame: this.frameIndex + 1 };
  }

  get learning() { return this.learn !== null; }
  get learnProgress() { return this.learn ? this.learn.frames / this.learn.framesNeeded : 1; }

  /** Process one hop of H input samples; returns H output samples (one hop of latency). */
  processHop(hop) {
    const { N, H, bins, re, im, w, buf } = this;
    buf.copyWithin(0, H);                    // Buffer: keep the last N − H samples …
    buf.set(hop, N - H);                     // … and append the new hop
    for (let n = 0; n < N; n++) { re[n] = buf[n] * w[n]; im[n] = 0; }
    fftInPlace(re, im);

    // Learning: frame k (k ≥ 1) is the first one fully inside the stream (frame 0 holds the
    // zero initial conditions), exactly the frames Eq. (4) uses.
    const L = this.learn;
    if (L && this.frameIndex >= L.startFrame) {
      for (let k = 0; k < bins; k++) L.acc[k] += Math.hypot(re[k], im[k]);
      if (++L.frames >= L.framesNeeded) {
        for (let k = 0; k < bins; k++) L.acc[k] /= L.frames;
        this.noise = L.acc;
        this.learn = null;
      }
    }
    this.frameIndex++;

    if (this.noise && !this.bypass) {
      const { alpha, beta, p, noise } = this;
      for (let k = 0; k < bins; k++) {
        const mag = Math.hypot(re[k], im[k]);
        const nk = noise[k];
        let s;
        if (p === 1) s = Math.max(mag - alpha * nk, beta * nk);
        else s = Math.sqrt(Math.max(mag * mag - alpha * nk * nk, beta * nk * nk));
        const g = s / Math.max(mag, EPS);
        re[k] *= g; im[k] *= g;
        if (k > 0 && k < N / 2) { re[N - k] *= g; im[N - k] *= g; }   // keep Hermitian symmetry
      }
    }
    fftInPlace(re, im, true);
    const out = new Float64Array(H);
    for (let n = 0; n < H; n++) out[n] = re[n] / N + this.prev[n];
    for (let n = 0; n < H; n++) this.prev[n] = re[n + H] / N;
    return out;
  }
}
