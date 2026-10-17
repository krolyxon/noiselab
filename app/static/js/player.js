// A/B player: original / cleaned / removed-noise share ONE playhead.
// All three are decoded AudioBuffers on one AudioContext; switching source while
// playing restarts the new buffer at the same position with a 15 ms crossfade.

export const SOURCES = ["original", "cleaned", "noise", "clean"];   // "clean" = demo ground truth only
const CORE = ["original", "cleaned", "noise"];
const FADE = 0.015;

export class ABPlayer {
  constructor({ onTime, onState }) {
    this.onTime = onTime;
    this.onState = onState;
    this.ctx = null;
    this.buffers = {};
    this.source = "cleaned";
    this.offset = 0;              // position (s) while paused
    this.playing = false;
    this.node = null;             // {src, gain}
    this._raf = null;
    this._loadSeq = 0;
  }

  _context() {
    this.ctx ??= new (window.AudioContext || window.webkitAudioContext)();
    return this.ctx;
  }

  get duration() {
    const b = this.buffers.original || this.buffers.cleaned;
    return b ? b.duration : 0;
  }

  get time() {
    if (!this.playing) return this.offset;
    return Math.min(this.ctx.currentTime - this._t0, this.duration);
  }

  get ready() { return CORE.every((k) => this.buffers[k]); }

  get available() { return SOURCES.filter((k) => this.buffers[k]); }

  /** Fetch + decode the given URLs ({kind: url}); keeps position and play state. */
  async load(urls) {
    const seq = ++this._loadSeq;
    const ctx = this._context();
    const entries = await Promise.all(Object.entries(urls).map(async ([k, url]) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`could not load ${k} audio (${res.status})`);
      return [k, await ctx.decodeAudioData(await res.arrayBuffer())];
    }));
    if (seq !== this._loadSeq) return;           // a newer load superseded this one
    const wasPlaying = this.playing;
    const t = this.time;
    if (wasPlaying) this._stopNode(true);
    for (const [k, buf] of entries) this.buffers[k] = buf;
    this.offset = Math.min(t, this.duration);
    if (wasPlaying) this._startNode(this.offset);
    this._emitState();
  }

  reset() {
    this.pause();
    this.buffers = {};
    if (this.source === "clean") this.source = "cleaned";
    this.offset = 0;
    this.onTime?.(0);
    this._emitState();
  }

  async play() {
    if (!this.ready || this.playing) return;
    const ctx = this._context();
    if (ctx.state === "suspended") await ctx.resume();
    if (this.offset >= this.duration - 0.01) this.offset = 0;
    this._startNode(this.offset);
    this.playing = true;
    this._tick();
    this._emitState();
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.time;
    this._stopNode(true);
    this.playing = false;
    cancelAnimationFrame(this._raf);
    this.onTime?.(this.offset);
    this._emitState();
  }

  toggle() { return this.playing ? this.pause() : this.play(); }

  seek(t) {
    t = Math.max(0, Math.min(t, this.duration));
    if (this.playing) {
      this._stopNode(true);
      this._startNode(t);
    }
    this.offset = t;
    this.onTime?.(t);
  }

  setSource(kind) {
    if (!this.buffers[kind] || kind === this.source) return;
    const t = this.time;
    this.source = kind;
    if (this.playing) {
      this._stopNode(true);
      this._startNode(t);
    }
    this._emitState();
  }

  cycle() {
    const av = this.available;
    if (av.length) this.setSource(av[(av.indexOf(this.source) + 1) % av.length]);
  }

  // --- internals --------------------------------------------------------------------------
  _startNode(t) {
    const ctx = this.ctx;
    const src = ctx.createBufferSource();
    src.buffer = this.buffers[this.source];
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(1, ctx.currentTime + FADE);
    src.connect(gain).connect(ctx.destination);
    const node = { src, gain };
    src.onended = () => {
      if (this.node !== node) return;            // stopped deliberately
      this.playing = false;
      this.offset = this.duration;
      this.node = null;
      cancelAnimationFrame(this._raf);
      this.onTime?.(this.offset);
      this._emitState();
    };
    src.start(0, t);
    this._t0 = ctx.currentTime - t;
    this.node = node;
  }

  _stopNode(fade) {
    const n = this.node;
    if (!n) return;
    this.node = null;
    const now = this.ctx.currentTime;
    if (fade) {
      n.gain.gain.cancelScheduledValues(now);
      n.gain.gain.setValueAtTime(n.gain.gain.value, now);
      n.gain.gain.linearRampToValueAtTime(0, now + FADE);
      n.src.stop(now + FADE);
    } else {
      n.src.stop();
    }
  }

  _tick() {
    let last = 0;
    const step = (ts) => {
      if (!this.playing) return;
      if (ts - last > 33) { this.onTime?.(this.time); last = ts; }   // ~30 fps playhead
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }

  _emitState() {
    this.onState?.({ playing: this.playing, source: this.source, ready: this.ready, available: this.available });
  }
}
