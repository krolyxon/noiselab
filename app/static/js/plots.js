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
const DB_RANGE = 90;        // colour scale spans [max − 90, max] dBFS

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
    tickfont: { family: "JetBrains Mono", size: 9, color: c.muted }, ticklen: 3, ...extra,
  };
}

function baseLayout(c, extra = {}) {
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter", size: 10, color: c.muted }, showlegend: false,
    hoverlabel: { bgcolor: c.panel, bordercolor: c.grid, font: { family: "JetBrains Mono", size: 10, color: c.text } },
    dragmode: "zoom", ...extra,
  };
}

const playheadShape = (c, t = 0) => ({
  type: "line", xref: "x", yref: "paper", x0: t, x1: t, y0: 0, y1: 1,
  line: { color: c.text, width: 1.2 }, layer: "above",
});

// ----------------------------------------------------------------------------------------

export class Plots {
  constructor({ wave, specs, onSeek, onRegion }) {
    this.wave = wave;
    this.specs = specs;                         // {noisy, cleaned, removed}
    this.onSeek = onSeek;
    this.onRegion = onRegion;
    this.timePlots = [wave, specs.noisy, specs.cleaned, specs.removed];
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
          xpad: 0, outlinewidth: 0, tickfont: { family: "JetBrains Mono", size: 9, color: c.muted },
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
    const fr = src === this.wave ? undefined : pick("yaxis");
    if (xr === undefined && fr === undefined) return;         // e.g. playhead/shape updates
    if (xr !== undefined) this.xr = xr;
    if (fr !== undefined) this.fr = fr;
    this._syncing = true;
    const x = { "xaxis.range": this.xr ?? [0, this.duration] };
    const f = { ...x, "yaxis.range": this.fr ?? [0, this.fmax] };
    Promise.all(this.timePlots.filter((d) => d.layout).map((d) => Plotly.relayout(d, d === this.wave ? x : f)))
      .finally(() => { this._syncing = false; });
  }

  /** A click (not a drag) anywhere in the plot area seeks to that time. */
  _clickToSeek(div) {
    let down = null;
    div.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY }; });
    div.addEventListener("pointerup", (e) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      const fl = div._fullLayout;
      if (!fl?.xaxis) return;
      const box = div.getBoundingClientRect();
      const px = e.clientX - box.left - fl._size.l;
      const py = e.clientY - box.top - fl._size.t;
      if (px < 0 || px > fl._size.w || py < 0 || py > fl._size.h) return;
      const t = fl.xaxis.p2c(px);
      if (Number.isFinite(t) && this.onSeek) this.onSeek(Math.max(0, Math.min(t, this.duration ?? t)), div);
    });
  }

  purgeAll() {
    for (const d of this.timePlots) Plotly.purge(d);
    this._ready = false;
  }
}
