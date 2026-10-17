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
};

const GROUPS = [
  { title: "Framing", keys: ["n_fft", "overlap", "window"] },
  { title: "Noise estimate", keys: ["noise_mode", "noise_seconds", "region_start", "region_end", "minstats_percentile"] },
  { title: "Subtraction", keys: ["method", "alpha", "beta", "exponent"] },
  { title: "Musical-noise smoothing", extended: true, keys: ["gain_smooth", "freq_smooth_bins"] },
  { title: "Output", keys: ["output_gain_db"] },
];

export class Controls {
  constructor({ container, presetSelect, presetDesc, schema, presets, onChange }) {
    this.container = container;
    this.presetSelect = presetSelect;
    this.presetDesc = presetDesc;
    this.schema = schema;
    this.presets = presets;
    this.onChange = onChange;
    this.values = Object.fromEntries(Object.entries(schema).map(([k, s]) => [k, s.default]));
    this.inputs = {};
    this.sr = null;
    this.profileId = null;
    this.duration = null;
    this._build();
    this._sync();
  }

  get() { return { ...this.values }; }

  /** Parameters actually sent to the server (drop the ones irrelevant to the chosen mode). */
  request() {
    const v = this.get();
    if (v.noise_mode !== "profile") v.profile_id = "";
    return v;
  }

  set(values, { emit = true } = {}) {
    Object.assign(this.values, values);
    this._sync();
    if (emit) this.onChange?.(this.get());
  }

  reset() { this.set(Object.fromEntries(Object.entries(this.schema).map(([k, s]) => [k, s.default]))); }

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
    for (const g of GROUPS) {
      const box = el("div", { class: "group" },
        el("div", { class: "group-title" }, g.title, g.extended ? el("span", { class: "badge", text: "extended",
          title: "Beyond the basic Boll/Berouti algorithm; not in the basic Simulink model. Off by default." }) : null));
      for (const k of g.keys) box.append(this._field(k));
      frag.append(box);
    }
    this.container.replaceChildren(frag);
  }

  _field(key) {
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
      ["profile_id", "region_start", "region_end"].includes(k) || v[k] === pv));
    this.presetSelect.value = match ? match.id : "";
    this.presetDesc.textContent = match ? match.description : "Custom settings.";
  }
}
