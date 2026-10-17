// Browser-side audio helpers: decode any format the browser understands, downmix,
// and encode 16-bit PCM WAV — so the server never needs ffmpeg.

/** Average all channels of an AudioBuffer into one Float32Array. */
export function downmix(buf) {
  const n = buf.length, ch = buf.numberOfChannels;
  if (ch === 1) return buf.getChannelData(0).slice();
  const out = new Float32Array(n);
  for (let c = 0; c < ch; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < n; i++) out[i] += d[i] / ch;
  }
  return out;
}

/** Encode mono float samples (−1…1) as a 16-bit PCM RIFF/WAVE Blob. */
export function encodeWav(samples, sampleRate) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true);           // PCM fmt chunk size
  v.setUint16(20, 1, true);                             // format = PCM
  v.setUint16(22, 1, true);                             // channels = 1
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);                // byte rate
  v.setUint16(32, 2, true);                             // block align
  v.setUint16(34, 16, true);                            // bits per sample
  str(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0, o = 44; i < n; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

let decodeCtx = null;
/** Decode an encoded file/blob with the Web Audio API (resamples to the context rate). */
export async function decodeAudio(arrayBuffer) {
  decodeCtx ??= new (window.AudioContext || window.webkitAudioContext)();
  return decodeCtx.decodeAudioData(arrayBuffer.slice(0));
}

/** Min/max envelope with ≤ maxPoints buckets, same shape as the server's viz payload. */
export function envelope(samples, sampleRate, maxPoints = 2000) {
  const n = samples.length, k = Math.min(maxPoints, n);
  const t = new Array(k), mn = new Array(k), mx = new Array(k);
  for (let b = 0; b < k; b++) {
    const a = Math.floor((b * n) / k), e = Math.max(a + 1, Math.floor(((b + 1) * n) / k));
    let lo = Infinity, hi = -Infinity;
    for (let i = a; i < e; i++) { const s = samples[i]; if (s < lo) lo = s; if (s > hi) hi = s; }
    t[b] = (a + e) / 2 / sampleRate; mn[b] = lo; mx[b] = hi;
  }
  return { t, min: mn, max: mx };
}
