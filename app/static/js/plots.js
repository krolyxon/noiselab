// Plotly wrappers. Rules: Plotly.react for data updates, shapes-only relayout for the
// playhead. The zoom state (shared time range for all plots, shared frequency range for
// the spectrograms) is owned here and passed explicitly on every redraw, so a reprocess
// never resets or desynchronises the view.
/* global Plotly */

import { decodeMatrix } from "./util.js";

// Inferno (perceptually uniform), sampled from matplotlib at 11 points.
const INFERNO = [
  [0.0, "#000004"], [0.1, "#160b39"], [0.2, "#420a68"], [0.3, "#6a176e"], [0.4, "#932667"],
  [0.5, "#bc3754"], [0.6, "#dd513a"], [0.7, "#f37819"], [0.8, "#fca50a"], [0.9, "#f6d746"], [1.0, "#fcffa4"],
];
// Font stacks with generic fallbacks: exported PNGs are rasterised from SVG without
// access to the page's webfonts, so they need a sans/mono fallback (not the serif default).
const SANS = "Inter, 'Helvetica Neue', Arial, sans-serif";
const MONO = "'JetBrains Mono', Consolas, 'DejaVu Sans Mono', monospace";
const DB_RANGE = 90;
const SEEK_CLICK_DELAY_MS = 220;   // single click → seek, unless it turns into a double-click        // colour scale spans [max − 90, max] dBFS
// Viridis (perceptually uniform) for the gain mask: 0 = fully suppressed, 1 = passed unchanged.
const VIRIDIS = [
  [0.0, "#440154"], [0.1, "#482878"], [0.2, "#3e4989"], [0.3, "#31688e"], [0.4, "#26828e"],
  [0.5, "#1f9e89"], [0.6, "#35b779"], [0.7, "#6ece58"], [0.8, "#b5de2b"], [0.9, "#dde318"], [1.0, "#fde725"],
];

/** "#rrggbb" → "rgba(r,g,b,a)" */
export function rgba(hex, a) {
  const n = parseInt(hex.replace("#", ""), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const colors = () => ({
  noisy: css("--noisy"), clean: css("--clean"), noise: css("--noise"), ok: css("--ok"),
  text: css("--text"), muted: css("--muted"), grid: css("--border"), panel: css("--panel"),
});

const CONFIG = {
  displaylogo: false, responsive: true, displayModeBar: "hover", scrollZoom: false,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines", "hoverClosestCartesian",
    "hoverCompareCartesian"],
  toImageButtonOptions: { format: "png", scale: 2 },
};

function axis(title, c, extra = {}) {
  return {
    title: { text: title, font: { size: 10, color: c.muted }, standoff: 4 },
    gridcolor: c.grid, zerolinecolor: c.grid, linecolor: c.grid, tickcolor: c.grid,
    tickfont: { family: MONO, size: 9, color: c.muted }, ticklen: 3, ...extra,
  };
}

function baseLayout(c, extra = {}) {
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: SANS, size: 10, color: c.muted }, showlegend: false,
    hoverlabel: { bgcolor: c.panel, bordercolor: c.grid, font: { family: MONO, size: 10, color: c.text } },
    dragmode: "zoom", ...extra,
  };
}

const playheadShape = (c, t = 0) => ({
  type: "line", xref: "x", yref: "paper", x0: t, x1: t, y0: 0, y1: 1,
  line: { color: c.text, width: 1.2 }, layer: "above",
});

// ----------------------------------------------------------------------------------------

export class Plots {
  constructor({ wave, specs, gain, energy, onSeek, onRegion }) {
    this.wave = wave;
    this.specs = specs;                         // {noisy, cleaned, removed}
    this.gain = gain;
    this.energy = energy;
    this.onSeek = onSeek;
    this.onRegion = onRegion;
    this.timePlots = [wave, specs.noisy, specs.cleaned, specs.removed, gain, energy];
    this.freqPlots = [specs.noisy, specs.cleaned, specs.removed, gain];   // share the frequency zoom
    this.xr = null;                             // shared time range [a, b] or null = full clip
    this.fr = null;                             // shared spectrogram frequency range or null = full
    this.region = null;                         // {x0, x1, label}
    this.hidden = { removed: true };
    this._syncing = false;
    this._ready = false;
  }

  /** Waveform only, from a browser-side envelope — the instant preview before the server replies. */
  preview(env, duration) {
    this.xr = this.fr = null;
    this.duration = duration;
    this._drawWave({ original: env }, duration);
  }

  /** Full update from a /api/process or /api/reprocess response. */
  update(res, { newJob }) {
    if (newJob) this.xr = this.fr = null;
    this.duration = res.duration;
    this._drawWave(res.viz.waveform, res.duration);
    this._drawSpecs(res.viz.spectrogram, res.duration);
    this._drawGain(res.viz.gain_mask, res.viz.spectrogram, res.duration);
    this._drawEnergy(res.viz.energy, res.duration);
    if (!this._ready) { this._wire(); this._ready = true; }
  }

  // --- waveform -------------------------------------------------------------------------
  _drawWave(w, duration) {
    const c = colors();
    const band = (env, color, name, visible = true) => [
      { x: env.t, y: env.max, type: "scatter", mode: "lines", line: { color, width: 0.8 }, hoverinfo: "skip",
        name, visible },
      { x: env.t, y: env.min, type: "scatter", mode: "lines", line: { color, width: 0.8 }, hoverinfo: "skip",
        fill: "tonexty", fillcolor: rgba(color, 0.33), name, visible },
    ];
    const traces = [
      ...band(w.original, c.noisy, "original"),
      ...(w.cleaned ? band(w.cleaned, c.clean, "cleaned") : []),
      ...(w.removed ? band(w.removed, c.noise, "removed", !this.hidden.removed) : []),
    ];
    const layout = baseLayout(c, {
      margin: { l: 44, r: 50, t: 4, b: 28 },
      xaxis: axis("time (s)", c, { range: this.xr ?? [0, duration] }),
      yaxis: axis("amplitude", c, { fixedrange: true, range: [-1, 1], zeroline: true }),
      shapes: [playheadShape(c, this._t ?? 0), ...this._regionShapes(c)],
      dragmode: this.region?.editable ? "select" : "zoom", selectdirection: "h",
    });
    Plotly.react(this.wave, traces, layout, CONFIG);
  }

  setTraceVisible(name, visible) {
    this.hidden[name] = !visible;
    const idx = this.wave.data?.map((t, i) => (t.name === name ? i : -1)).filter((i) => i >= 0) ?? [];
    if (idx.length) Plotly.restyle(this.wave, { visible }, idx);
  }

  // --- spectrograms ---------------------------------------------------------------------
  _drawSpecs(sp, duration) {
    const c = colors();
    const z = { noisy: decodeMatrix(sp.noisy), cleaned: decodeMatrix(sp.cleaned), removed: decodeMatrix(sp.removed) };
    let zmax = -Infinity;
    for (const row of z.noisy) for (const v of row) if (v > zmax) zmax = v;
    zmax = Math.ceil(zmax / 10) * 10;
    const zmin = zmax - DB_RANGE;
    const fmax = (this.fmax = sp.f[sp.f.length - 1]);

    for (const [key, div] of Object.entries(this.specs)) {
      const last = key === "removed";
      const trace = {
        type: "heatmap", x: sp.t, y: sp.f, z: z[key], zmin, zmax, colorscale: INFERNO, zsmooth: false,
        showscale: last,
        colorbar: last ? {
          title: { text: "dBFS", side: "right", font: { size: 10, color: c.muted } }, thickness: 8, len: 1, x: 1.02,
          xpad: 0, outlinewidth: 0, tickfont: { family: MONO, size: 9, color: c.muted },
        } : undefined,
        hovertemplate: "%{x:.3f} s<br>%{y:.0f} Hz<br>%{z:.1f} dBFS<extra></extra>",
      };
      const layout = baseLayout(c, {
        margin: { l: 44, r: 50, t: 4, b: 28 },
        xaxis: axis("time (s)", c, { range: this.xr ?? [0, duration] }),
        yaxis: axis(key === "noisy" ? "frequency (Hz)" : "", c, { range: this.fr ?? [0, fmax] }),
        shapes: [playheadShape(c, this._t ?? 0)],
      });
      Plotly.react(div, [trace], layout, CONFIG);
    }
  }

  // --- gain mask G(k, m) -----------------------------------------------------------------
  _drawGain(gm, sp, duration) {
    const c = colors();
    const trace = {
      type: "heatmap", x: sp.t, y: sp.f, z: decodeMatrix(gm), zmin: 0, zmax: 1, colorscale: VIRIDIS, zsmooth: false,
      colorbar: { title: { text: "G", side: "right", font: { size: 10, color: c.muted } }, thickness: 8, len: 1,
        x: 1.02, xpad: 0, outlinewidth: 0, tickvals: [0, 0.25, 0.5, 0.75, 1],
        tickfont: { family: MONO, size: 9, color: c.muted } },
      hovertemplate: "%{x:.3f} s<br>%{y:.0f} Hz<br>G = %{z:.3f}<extra></extra>",
    };
    Plotly.react(this.gain, [trace], baseLayout(c, {
      margin: { l: 44, r: 50, t: 4, b: 28 },
      xaxis: axis("time (s)", c, { range: this.xr ?? [0, duration] }),
      yaxis: axis("frequency (Hz)", c, { range: this.fr ?? [0, sp.f[sp.f.length - 1]] }),
      shapes: [playheadShape(c, this._t ?? 0)],
    }), CONFIG);
  }

  // --- frame energy + VAD -----------------------------------------------------------------
  _drawEnergy(en, duration) {
    const c = colors();
    const all = [...en.noisy_db, ...en.cleaned_db].filter(Number.isFinite);
    const lo = Math.max(-100, Math.floor(Math.min(...all) / 10) * 10), hi = Math.ceil(Math.max(...all) / 10) * 10 + 5;
    const traces = [
      // VAD shading on a hidden 0…1 axis: green where the energy VAD says "speech"
      { x: en.t, y: en.speech, yaxis: "y2", type: "scatter", mode: "lines", line: { width: 0, shape: "hv" },
        fill: "tozeroy", fillcolor: rgba(c.ok, 0.10), hoverinfo: "skip", name: "speech (VAD)" },
      { x: en.t, y: en.noisy_db, type: "scatter", mode: "lines", line: { color: c.noisy, width: 1 }, name: "noisy",
        hovertemplate: "%{x:.3f} s · noisy %{y:.1f} dBFS<extra></extra>" },
      { x: en.t, y: en.cleaned_db, type: "scatter", mode: "lines", line: { color: c.clean, width: 1 }, name: "cleaned",
        hovertemplate: "%{x:.3f} s · cleaned %{y:.1f} dBFS<extra></extra>" },
    ];
    Plotly.react(this.energy, traces, baseLayout(c, {
      margin: { l: 44, r: 50, t: 4, b: 28 },
      xaxis: axis("time (s)", c, { range: this.xr ?? [0, duration] }),
      yaxis: axis("energy (dBFS)", c, { range: [lo, hi], fixedrange: true }),
      yaxis2: { overlaying: "y", range: [0, 1], visible: false, fixedrange: true },
      shapes: [playheadShape(c, this._t ?? 0)],
    }), CONFIG);
  }

  // --- noise region overlay -------------------------------------------------------------
  /** region = {x0, x1, editable} or null. Drawn in magenta (estimated noise colour). */
  setRegion(region) {
    this.region = region;
    if (!this.wave.layout) return;
    const c = colors();
    Plotly.relayout(this.wave, {
      shapes: [playheadShape(c, this._t ?? 0), ...this._regionShapes(c)],
      dragmode: region?.editable ? "select" : "zoom",
    });
  }

  _regionShapes(c) {
    if (!this.region) return [];
    return [{
      type: "rect", xref: "x", yref: "paper", x0: this.region.x0, x1: this.region.x1, y0: 0, y1: 1,
      fillcolor: c.noise, opacity: 0.12, line: { color: c.noise, width: 1, dash: "dot" }, layer: "below",
    }];
  }

  // --- playhead -------------------------------------------------------------------------
  /** Move the playhead on every time plot — shapes only, no data re-render. */
  setPlayhead(t) {
    this._t = t;
    for (const div of this.timePlots) {
      if (div.layout?.shapes?.length) Plotly.relayout(div, { "shapes[0].x0": t, "shapes[0].x1": t });
    }
  }

  // --- interaction ----------------------------------------------------------------------
  _wire() {
    for (const div of this.timePlots) {
      div.on("plotly_relayout", (ev) => this._onRelayout(div, ev));
      this._clickToSeek(div);
    }
    this.wave.on("plotly_selected", (ev) => {
      if (ev?.range?.x && this.onRegion) {
        const [a, b] = ev.range.x;
        this.onRegion(Math.max(0, Math.min(a, b)), Math.min(this.duration ?? Infinity, Math.max(a, b)));
      }
      Plotly.restyle(this.wave, { selectedpoints: [null] });
      Plotly.relayout(this.wave, { selections: [] });
    });
    // In select (region) mode Plotly turns double-click into "deselect" instead of
    // "reset zoom"; treat it as a reset so zooming out works the same in every mode.
    this.wave.on("plotly_deselect", () => this.resetView());
  }

  resetView() {
    this._onRelayout(null, { "xaxis.autorange": true, "yaxis.autorange": true });
  }

  /** User zoomed/panned/reset one plot → update the shared view and apply it to all plots. */
  _onRelayout(src, ev) {
    if (this._syncing) return;
    const pick = (ax) => {
      if (ev[`${ax}.range[0]`] !== undefined) return [ev[`${ax}.range[0]`], ev[`${ax}.range[1]`]];
      if (ev[`${ax}.range`]) return ev[`${ax}.range`];
      if (ev[`${ax}.autorange`]) return null;                 // double-click reset
      return undefined;                                       // this axis did not change
    };
    const xr = pick("xaxis");
    const fr = this.freqPlots.includes(src) ? pick("yaxis") : undefined;
    if (xr === undefined && fr === undefined) return;         // e.g. playhead/shape updates
    if (xr !== undefined) this.xr = xr;
    if (fr !== undefined) this.fr = fr;
    this._syncing = true;
    const x = { "xaxis.range": this.xr ?? [0, this.duration] };
    const f = { ...x, "yaxis.range": this.fr ?? [0, this.fmax] };
    Promise.all(this.timePlots.filter((d) => d.layout).map((d) => Plotly.relayout(d, this.freqPlots.includes(d) ? f : x)))
      .finally(() => { this._syncing = false; });
  }

  /** A click (not a drag) anywhere in the plot area seeks to that time.
   *  pointerup is caught on window: during a gesture Plotly lays a full-page "cover slip"
   *  over the document, so the up event does not reach the plot's own div. */
  _clickToSeek(div) {
    div.addEventListener("pointerdown", (e) => {
      // A second press within the delay is a double-click (zoom reset): cancel the pending seek,
      // otherwise the playhead redraw between the two clicks can break Plotly's double-click.
      if (this._seekTimer) { clearTimeout(this._seekTimer); this._seekTimer = null; }
      this._down = { div, x: e.clientX, y: e.clientY };
    });
    if (this._upWired) return;
    this._upWired = true;
    window.addEventListener("pointerup", (e) => {
      const down = this._down;
      this._down = null;
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      const fl = down.div._fullLayout;
      if (!fl?.xaxis) return;
      const box = down.div.getBoundingClientRect();
      const px = e.clientX - box.left - fl._size.l;
      const py = e.clientY - box.top - fl._size.t;
      if (px < 0 || px > fl._size.w || py < 0 || py > fl._size.h) return;
      const t = fl.xaxis.p2c(px);
      if (!Number.isFinite(t) || !this.onSeek) return;
      this._seekTimer = setTimeout(() => {
        this._seekTimer = null;
        this.onSeek(Math.max(0, Math.min(t, this.duration ?? t)), down.div);
      }, SEEK_CLICK_DELAY_MS);
    }, true);
  }

  purgeAll() {
    for (const d of this.timePlots) Plotly.purge(d);
    this._ready = false;
  }
}

// ---------------------------------------------------------------------------------------
// Frequency-domain plots (not time-synchronised).

/** Noise estimate |N̂| with the mean noisy and cleaned spectra overlaid (dBFS vs Hz). */
export function drawSpectra(div, sp) {
  const c = colors();
  const line = (y, color, name, dash) => ({ x: sp.f, y, type: "scatter", mode: "lines", name,
    line: { color, width: 1.3, dash }, hovertemplate: `%{x:.0f} Hz · ${name} %{y:.1f} dBFS<extra></extra>` });
  Plotly.react(div, [
    line(sp.noisy_db, c.noisy, "mean noisy"),
    line(sp.cleaned_db, c.clean, "mean cleaned"),
    line(sp.noise_profile_db, c.noise, "noise |N̂|"),
  ], baseLayout(c, {
    margin: { l: 40, r: 6, t: 4, b: 28 },
    xaxis: axis("frequency (Hz)", c, { type: "log", range: [Math.log10(40), Math.log10(sp.f[sp.f.length - 1])],
      tickvals: [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ticktext: ["50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"] }),
    yaxis: axis("dBFS", c, { range: spectraRange([sp.noisy_db, sp.cleaned_db, sp.noise_profile_db]) }),
  }), { ...CONFIG, displayModeBar: false });
}

/** STFT frame inspector: |Y|, α|N̂| (what gets subtracted), floor β|N̂| and the result |S̃| = G·|Y|. */
export function drawInspector(div, fr) {
  const c = colors();
  const wiener = fr.method === "wiener";
  const lbl = wiener ? "√α·|N̂| (G = 0 below)" : `α^(1/p)·|N̂| subtracted (p = ${fr.exponent})`;
  const floorLbl = wiener ? "floor β·|Y|" : "floor β^(1/p)·|N̂|";
  document.getElementById("insp-l-sub").textContent = lbl;
  document.getElementById("insp-l-floor").textContent = floorLbl;
  const line = (y, color, name, extra = {}) => ({ x: fr.f, y, type: "scatter", mode: "lines", name,
    line: { color, width: 1.2, ...extra }, hovertemplate: `%{x:.0f} Hz · ${name.split(" ")[0]} %{y:.1f} dBFS<extra></extra>` });
  Plotly.react(div, [
    line(fr.noisy_db, c.noisy, "|Y| noisy"),
    line(fr.alpha_noise_db, c.noise, lbl, { dash: "dash" }),
    line(fr.floor_db, c.noise, floorLbl, { dash: "dot", width: 1 }),
    line(fr.cleaned_db, c.clean, "|S̃| = G·|Y|", { width: 1.6 }),
  ], baseLayout(c, {
    margin: { l: 44, r: 8, t: 6, b: 28 },
    xaxis: axis("frequency (Hz)", c, { range: [0, fr.f[fr.f.length - 1]] }),
    yaxis: axis("dBFS", c, { range: spectraRange([fr.noisy_db, fr.cleaned_db, fr.alpha_noise_db]) }),
    hovermode: "x unified",
  }), CONFIG);
}

function spectraRange(arrays) {
  let hi = -Infinity, lo = Infinity;
  for (const a of arrays) for (const v of a) { if (v > hi) hi = v; if (v < lo) lo = v; }
  hi = Math.ceil(hi / 10) * 10 + 5;
  return [Math.max(lo, hi - 100), hi];
}
