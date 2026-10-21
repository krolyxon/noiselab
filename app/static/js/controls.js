// Parameter panel, built from the server's schema (/api/presets → schema) so ranges
// and defaults are defined in exactly one place (app/params.py).

import { $, $$, el, fmt } from "./util.js";

// UI-only metadata: label, symbol, slider step and readout format for each parameter.
const META = {
  n_fft: { label: "Frame length", sym: "N", labels: (v) => String(v) },
  overlap: { label: "Overlap", sym: "1 − H/N", labels: (v) => `${v * 100}%` },
  window: { label: "Window", sym: "w[n]", labels: (v) => ({ hann: "Hann", hamming: "Hamming", sqrt_hann: "√Hann" })[v] },
  noise_mode: { label: "Noise estimate", sym: "|N̂|",
    labels: (v) => ({ initial: "Initial segment", region: "Selected region", profile: "Noise profile clip",
      minstats: "Min. statistics (advanced)" })[v] },
  noise_seconds: { label: "Noise-only lead-in", sym: "T", step: 0.05, show: (v) => fmt(v, 2, "s") },
  region_start: { label: "Region start", sym: "t₀", step: 0.01 },
  region_end: { label: "Region end", sym: "t₁", step: 0.01 },
  minstats_percentile: { label: "Percentile", sym: "q", step: 1, show: (v) => fmt(v, 0, "%") },
  method: { label: "Gain rule", sym: "G", labels: (v) => ({ spectral_subtraction: "Subtraction", wiener: "Wiener" })[v] },
  alpha: { label: "Over-subtraction", sym: "α", step: 0.05, show: (v) => fmt(v, 2) },
  beta: { label: "Spectral floor", sym: "β", step: 0.005, show: (v) => fmt(v, 3) },
  exponent: { label: "Exponent", sym: "p", labels: (v) => (v === 1 ? "1 · |Y|" : "2 · |Y|²") },
  gain_smooth: { label: "Time smoothing", sym: "λ", step: 0.05, show: (v) => fmt(v, 2) },
  freq_smooth_bins: { label: "Freq. smoothing", sym: "B", step: 1, show: (v) => (v <= 1 ? "off" : `${v} bins`) },
  output_gain_db: { label: "Output gain", sym: "dB", step: 0.5,
    show: (v) => (v > 0 ? "+" : "") + fmt(v, 1, "dB") },
  atten_limit_db: { label: "Max attenuation", sym: "L", step: 1,
    show: (v) => (v >= 100 ? "off" : fmt(v, 0, "dB")) },
};

// Per-algorithm parameters (§14): label, symbol, readout. Keys not listed fall back to the
// schema's help text and a plain readout.
const ALGO_META = {
  dd_alpha: { label: "Decision-directed weight", sym: "α_dd", show: (v) => fmt(v, 3) },
  xi_min_db: { label: "A-priori SNR floor", sym: "ξ_min", show: (v) => fmt(v, 0, "dB") },
  gain_floor_db: { label: "Gain floor", sym: "G_min", show: (v) => fmt(v, 0, "dB") },
  g_min_db: { label: "Gain in speech absence", sym: "G_min", show: (v) => fmt(v, 0, "dB") },
  tracking_window_s: { label: "Noise-tracking window", sym: "D", show: (v) => fmt(v, 2, "s") },
  stationary: { label: "Noise model", labels: (v) => (v ? "Stationary" : "Non-stationary") },
  prop_decrease: { label: "Proportion removed", sym: "p", show: (v) => fmt(v * 100, 0, "%") },
  n_std_thresh: { label: "Threshold", sym: "σ", show: (v) => `${fmt(v, 2)} σ` },
  time_mask_smooth_ms: { label: "Mask smoothing (time)", show: (v) => fmt(v, 0, "ms") },
  freq_mask_smooth_hz: { label: "Mask smoothing (freq.)", show: (v) => fmt(v, 0, "Hz") },
  level: { label: "Suppression level", labels: (v) => ["", "mild", "moderate", "high", "very high"][v] },
  phase_compensation: { label: "HPF phase compensation", labels: (v) => (v ? "On" : "Off") },
  vad_threshold: { label: "VAD gate threshold", sym: "P(speech)", show: (v) => (v <= 0 ? "off" : fmt(v, 2)) },
  gate_db: { label: "Gate attenuation", show: (v) => fmt(v, 0, "dB") },
  min_gain_db: { label: "Minimum gain", sym: "G_min", show: (v) => fmt(v, 0, "dB") },
  post_filter: { label: "Post-filter", labels: (v) => (v ? "On" : "Off") },
  model: { label: "Pretrained model" },
};

const GROUPS = [
  { title: "Framing", keys: ["n_fft", "overlap", "window"] },
  { title: "Noise estimate", keys: ["noise_mode", "noise_seconds", "region_start", "region_end", "minstats_percentile"] },
  { title: "Subtraction", keys: ["method", "alpha", "beta", "exponent"] },
  { title: "Musical-noise smoothing", extended: true, keys: ["gain_smooth", "freq_smooth_bins"] },
  { title: "Output", keys: ["atten_limit_db", "output_gain_db"] },
];
const SS_ONLY_GROUPS = new Set(["Subtraction", "Musical-noise smoothing"]);

/** Order-insensitive deep equality for plain JSON values (preset matching). */
function same(a, b) {
  if (a === b) return true;
  if (typeof a !== "object" || typeof b !== "object" || !a || !b) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => same(a[k], b[k]));
}

export class Controls {
  constructor({ container, presetSelect, presetDesc, schema, presets, algorithms, onChange }) {
    this.container = container;
    this.presetSelect = presetSelect;
    this.presetDesc = presetDesc;
    this.schema = schema;
    this.presets = presets;
    this.algorithms = Object.fromEntries((algorithms || []).map((a) => [a.id, a]));
    this.onChange = onChange;
    this.values = Object.fromEntries(Object.entries(schema).map(([k, s]) => [k, s.default]));
    this.values.algo_params = this._algoDefaults(this.values.algorithm);
    this.algoInputs = {};
    this.inputs = {};
    this.sr = null;
    this.profileId = null;
    this.duration = null;
    this._build();
    this._sync();
  }

  get() { return { ...this.values, algo_params: { ...this.values.algo_params } }; }

  algorithm() { return this.algorithms[this.values.algorithm]; }

  _algoDefaults(id) {
    const a = this.algorithms[id];
    return a ? Object.fromEntries(Object.entries(a.schema).map(([k, s]) => [k, s.default])) : {};
  }

  /** Parameters actually sent to the server (drop the ones irrelevant to the chosen mode). */
  request() {
    const v = this.get();
    if (v.noise_mode !== "profile") v.profile_id = "";
    return v;
  }

  set(values, { emit = true } = {}) {
    const prevAlgo = this.values.algorithm;
    const { algo_params: ap, ...flat } = values;
    Object.assign(this.values, flat);
    if (this.values.algorithm !== prevAlgo) {
      this.values.algo_params = this._algoDefaults(this.values.algorithm);
      this._buildAlgo();
    }
    if (ap) Object.assign(this.values.algo_params, ap);
    this._sync();
    if (emit) this.onChange?.(this.get());
  }

  reset() {
    const d = Object.fromEntries(Object.entries(this.schema).map(([k, s]) => [k, s.default]));
    this.set({ ...d, algo_params: this._algoDefaults(d.algorithm) });
  }

  /** Default (algorithm-specific) parameters, keeping the current framing / noise / output settings. */
  setAlgorithm(id) {
    if (id === this.values.algorithm) return;
    this.set({ algorithm: id });
  }

  applyPreset(id) {
    const p = this.presets.find((x) => x.id === id);
    if (!p) return;
    const keep = { profile_id: this.values.profile_id, region_start: this.values.region_start,
      region_end: this.values.region_end };
    this.set({ ...p.params, ...keep });
  }

  setContext({ sr, duration }) {
    this.sr = sr ?? this.sr;
    this.duration = duration ?? this.duration;
    this._sync();
  }

  setProfile(id) {
    this.profileId = id;
    this.values.profile_id = id || "";
    this._sync();
  }

  // --- building ---------------------------------------------------------------------------
  _build() {
    for (const p of this.presets) this.presetSelect.append(el("option", { value: p.id, text: p.name }));
    this.presetSelect.addEventListener("change", () => this.applyPreset(this.presetSelect.value));

    const frag = document.createDocumentFragment();
    frag.append(this._algoSelector());
    this.algoBox = el("div", { class: "group algo-params" });
    frag.append(this.algoBox);
    for (const g of GROUPS) {
      const box = el("div", { class: "group", "data-group": g.title },
        el("div", { class: "group-title" }, g.title, g.extended ? el("span", { class: "badge", text: "extended",
          title: "Beyond the basic Boll/Berouti algorithm; not in the basic Simulink model. Off by default." }) : null));
      for (const k of g.keys) box.append(this._field(k));
      frag.append(box);
    }
    this.container.replaceChildren(frag);
    this._buildAlgo();
  }

  /** Algorithm dropdown grouped by family; unavailable models are disabled with the reason. */
  _algoSelector() {
    const sel = el("select", { id: "p-algorithm", "aria-label": "Algorithm" });
    const fams = {};
    for (const a of Object.values(this.algorithms)) (fams[a.family_name] ||= []).push(a);
    for (const [fam, list] of Object.entries(fams)) {
      const og = el("optgroup", { label: fam });
      for (const a of list) {
        og.append(el("option", { value: a.id, text: a.available ? a.name : `${a.name} — not installed`,
          disabled: !a.available, title: a.available ? a.description : a.reason }));
      }
      sel.append(og);
    }
    sel.addEventListener("change", () => this.setAlgorithm(sel.value));
    this.algoSelect = sel;
    this.algoDesc = el("p", { class: "algo-desc" });
    return el("div", { class: "group algo-group" },
      el("div", { class: "group-title" }, "Algorithm"),
      el("div", { class: "field", "data-key": "algorithm" }, sel), this.algoDesc);
  }

  /** (Re)build the controls of the selected algorithm's own parameters. */
  _buildAlgo() {
    if (!this.algoBox) return;
    const a = this.algorithm();
    this.algoInputs = {};
    const keys = a ? Object.keys(a.schema) : [];
    this.algoBox.hidden = keys.length === 0;
    if (!keys.length) { this.algoBox.replaceChildren(); return; }
    const title = el("div", { class: "group-title" }, `${a.name} parameters`,
      a.hand_written ? null : el("span", { class: "badge", text: "third-party",
        title: "A prebuilt comparison baseline (CLAUDE.md §14), not part of the Simulink algorithm." }));
    this.algoBox.replaceChildren(title, ...keys.map((k) => this._algoField(k, a.schema[k])));
  }

  _algoField(key, s) {
    const m = ALGO_META[key] || { label: key };
    const head = el("div", { class: "field-label" },
      el("span", {}, m.label, " ", el("span", { class: "sym", text: m.sym || "" })));
    const field = el("div", { class: "field", "data-algo-key": key, title: s.help });
    const setv = (v) => this.set({ algo_params: { [key]: v } });
    const choices = s.type === "bool" ? [true, false] : s.choices;
    if (choices) {
      const seg = el("div", { class: "seg", role: "radiogroup", "aria-label": m.label });
      for (const c of choices) {
        seg.append(el("button", { type: "button", role: "radio", "data-v": String(c),
          text: m.labels ? m.labels(c) : String(c), onclick: () => setv(c) }));
      }
      field.append(head, seg);
      this.algoInputs[key] = { kind: "seg", node: seg };
    } else {
      const step = s.type === "int" ? 1 : niceStep((s.max - s.min) / 100);
      const inp = el("input", { type: "range", min: s.min, max: s.max, step, "aria-label": m.label });
      const out = el("span", { class: "readout" });
      inp.addEventListener("input", () => {
        this.values.algo_params[key] = s.type === "int" ? parseInt(inp.value, 10) : parseFloat(inp.value);
        this._sync();
        this.onChange?.(this.get());
      });
      inp.addEventListener("dblclick", () => setv(s.default));
      head.append(out);
      field.append(head, inp);
      this.algoInputs[key] = { kind: "range", node: inp, out, show: m.show, s };
    }
    return field;
  }

  _field(key) {
    if (key === "algorithm") return el("span");          // built by _algoSelector()
    const s = this.schema[key], m = META[key] || { label: key };
    const head = el("div", { class: "field-label" },
      el("span", {}, m.label, " ", el("span", { class: "sym", text: m.sym || "" })));
    const field = el("div", { class: "field", "data-key": key, title: s.help });

    if (s.choices && s.choices.length <= 4 && key !== "noise_mode") {
      // segmented control
      const seg = el("div", { class: "seg", role: "radiogroup", "aria-label": m.label });
      for (const c of s.choices) {
        seg.append(el("button", { type: "button", role: "radio", "data-v": String(c), text: m.labels ? m.labels(c) : String(c),
          onclick: () => this.set({ [key]: c }) }));
      }
      field.append(head, seg);
      this.inputs[key] = { kind: "seg", node: seg };
    } else if (s.choices) {
      const sel = el("select", { id: `p-${key}`, "aria-label": m.label });
      for (const c of s.choices) sel.append(el("option", { value: c, text: m.labels ? m.labels(c) : c }));
      sel.addEventListener("change", () => this.set({ [key]: sel.value }));
      field.append(head, sel);
      this.inputs[key] = { kind: "select", node: sel };
    } else if (key === "region_start" || key === "region_end") {
      const inp = el("input", { type: "number", min: s.min, max: s.max, step: m.step, "aria-label": m.label });
      inp.addEventListener("change", () => {
        const v = parseFloat(inp.value);
        if (Number.isFinite(v)) this.set({ [key]: Math.max(s.min, Math.min(s.max, v)) });
      });
      head.append(el("span", { class: "readout", text: "s" }));
      field.append(head, inp);
      this.inputs[key] = { kind: "number", node: inp };
    } else if (s.type === "str") {
      return el("span");   // profile_id: not user-editable
    } else {
      const inp = el("input", { type: "range", min: s.min, max: s.max, step: m.step ?? "any", "aria-label": m.label });
      const out = el("span", { class: "readout" });
      inp.addEventListener("input", () => {
        const v = s.type === "int" ? parseInt(inp.value, 10) : parseFloat(inp.value);
        this.values[key] = v;
        this._sync();
        this.onChange?.(this.get());
      });
      inp.addEventListener("dblclick", () => this.set({ [key]: s.default }));   // double-click → default
      head.append(out);
      field.append(head, inp);
      this.inputs[key] = { kind: "range", node: inp, out, show: m.show };
    }
    return field;
  }

  // --- reflect state into the DOM -----------------------------------------------------------
  _sync() {
    const v = this.values;
    for (const [k, inp] of Object.entries(this.inputs)) {
      const val = v[k];
      if (inp.kind === "seg") {
        for (const b of $$("button", inp.node)) b.setAttribute("aria-checked", String(b.dataset.v === String(val)));
      } else if (inp.kind === "select") {
        inp.node.value = String(val);
      } else if (inp.kind === "number") {
        if (document.activeElement !== inp.node) inp.node.value = Number(val).toFixed(2);
      } else {
        inp.node.value = String(val);
        const s = this.schema[k];
        inp.node.style.setProperty("--fill", `${((val - s.min) / (s.max - s.min)) * 100}%`);
        inp.out.textContent = inp.show ? inp.show(val) : String(val);
      }
    }

    // algorithm selector + its own parameters
    const algo = this.algorithm();
    if (this.algoSelect) {
      this.algoSelect.value = v.algorithm;
      this.algoDesc.textContent = algo ? algo.description : "";
      this.algoDesc.title = algo ? algo.reference : "";
    }
    for (const [k, inp] of Object.entries(this.algoInputs)) {
      const val = v.algo_params[k];
      if (inp.kind === "seg") {
        for (const b of $$("button", inp.node)) b.setAttribute("aria-checked", String(b.dataset.v === String(val)));
      } else {
        inp.node.value = String(val);
        inp.node.style.setProperty("--fill", `${((val - inp.s.min) / (inp.s.max - inp.s.min)) * 100}%`);
        inp.out.textContent = inp.show ? inp.show(val) : String(val);
      }
    }
    const isSS = v.algorithm === "spectral_subtraction";
    for (const g of $$(".group[data-group]", this.container)) {
      const name = g.dataset.group;
      g.hidden = (SS_ONLY_GROUPS.has(name) && !isSS) || (name === "Noise estimate" && algo && !algo.uses_noise_estimate);
      if (name === "Framing") {
        g.classList.toggle("muted-group", !!algo && !algo.hand_written);
        g.title = algo && !algo.hand_written
          ? "This model uses its own framing; these settings only affect the plots and the effective gain mask."
          : "";
      }
    }

    // mode-dependent visibility
    const show = (k, on) => { const f = $(`.field[data-key="${k}"]`, this.container); if (f) f.hidden = !on; };
    show("noise_seconds", v.noise_mode === "initial");
    show("region_start", v.noise_mode === "region");
    show("region_end", v.noise_mode === "region");
    show("minstats_percentile", v.noise_mode === "minstats");
    const expField = $(`.field[data-key="exponent"]`, this.container);
    expField?.classList.toggle("disabled", v.method === "wiener");
    if (expField) expField.title = v.method === "wiener" ? "Not used by the Wiener rule (Eq. 9)" : this.schema.exponent.help;

    // profile mode needs a recorded profile
    const modeSel = this.inputs.noise_mode?.node;
    const profOpt = modeSel && [...modeSel.options].find((o) => o.value === "profile");
    if (profOpt) {
      profOpt.disabled = !this.profileId;
      profOpt.textContent = this.profileId ? "Noise profile clip" : "Noise profile clip (record one first)";
    }

    // resolution hint on the frame-length field
    const nf = $(`.field[data-key="n_fft"]`, this.container);
    if (nf && this.sr) {
      const hop = v.n_fft * (1 - v.overlap);
      nf.title = `${this.schema.n_fft.help}\nAt ${this.sr} Hz: frame ${fmt((v.n_fft / this.sr) * 1000, 1)} ms, ` +
        `hop ${fmt((hop / this.sr) * 1000, 1)} ms, bin spacing ${fmt(this.sr / v.n_fft, 2)} Hz.`;
    }

    // which preset (if any) matches the current values?
    const match = this.presets.find((p) => Object.entries(p.params).every(([k, pv]) =>
      ["profile_id", "region_start", "region_end"].includes(k) || same(v[k], pv)));
    this.presetSelect.value = match ? match.id : "";
    this.presetDesc.textContent = match ? match.description : "Custom settings.";
  }
}

/** A slider step of 1, 2 or 5 × 10^k close to x. */
function niceStep(x) {
  const e = 10 ** Math.floor(Math.log10(x));
  const f = x / e;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : 5) * e;
}
