// Metric cards: monospace values, fixed decimals, units, delta arrows. "n/a" when not computable.

import { tip } from "./glossary.js";
import { el, fmt, signed } from "./util.js";

function card(label, value, unit, { decimals = 2, sub = null, cls = "", wide = false, title = "" } = {}) {
  const na = value === null || value === undefined || !Number.isFinite(value);
  return el("div", { class: `card ${cls} ${wide ? "wide" : ""}`, title },
    el("div", { class: "card-label", text: label }),
    el("div", { class: `card-value ${na ? "na" : ""}` }, na ? "n/a" : fmt(value, decimals),
      na || !unit ? null : el("span", { class: "unit", text: unit })),
    sub);
}

/** "in X → Δ" line; `higherIsBetter` decides the arrow colour. */
function delta(before, after, unit, higherIsBetter = true) {
  if (!Number.isFinite(before) || !Number.isFinite(after)) return null;
  const d = after - before;
  const good = higherIsBetter ? d > 0 : d < 0;
  const arrow = d > 0 ? "▲" : d < 0 ? "▼" : "•";
  return el("div", { class: "card-sub", title: `before: ${fmt(before, 2, unit)}, change: ${signed(d)} ${unit}` },
    `${fmt(before, 2)} `,
    el("span", { class: Math.abs(d) < 0.005 ? "" : good ? "up" : "down", text: `${arrow}${signed(d)}` }));
}

function group(title, cards, note = "") {
  return el("div", { class: "mgroup" },
    el("div", { class: "mgroup-title" }, el("span", { text: title }), note ? el("span", { text: note }) : null),
    el("div", { class: "mgrid" }, cards));
}

export function renderMetrics(root, m) {
  const gt = m.ground_truth;
  const groundTruth = gt
    ? group("Compared with clean speech (demo)", [
      card("Signal-to-noise (SNR)", gt.snr_out_db, "dB", { sub: delta(gt.snr_in_db, gt.snr_out_db, "dB"),
        cls: "c-clean", title: tip("d_snr", "Eq. (16), output vs clean reference.") }),
      card("Clarity (SI-SDR)", gt.si_sdr_out_db, "dB", { sub: delta(gt.si_sdr_in_db, gt.si_sdr_out_db, "dB"),
        cls: "c-clean", title: tip("d_si_sdr", "Eq. (17).") }),
      card("Frame-wise SNR", gt.seg_snr_out_db, "dB", { sub: delta(gt.seg_snr_in_db, gt.seg_snr_out_db, "dB"),
        title: tip("seg_snr", "Eq. (18), clamped to [−10, 35] dB per frame.") }),
      card("Distortion (LSD) ↓", gt.lsd_out_db, "dB", { sub: delta(gt.lsd_in_db, gt.lsd_out_db, "dB", false),
        title: tip("lsd", "Eq. (35).") }),
      card("Sound quality (PESQ)", gt.pesq_out, "", { decimals: 2, sub: delta(gt.pesq_in, gt.pesq_out, ""),
        cls: "c-clean", title: tip("pesq", "Needs the optional 'pesq' package.") }),
      card("Intelligibility (ESTOI)", gt.estoi_out, "", { decimals: 3, sub: delta(gt.estoi_in, gt.estoi_out, ""),
        cls: "c-clean", title: tip("estoi", "Needs the optional 'pystoi' package.") }),
    ], "vs clean reference")
    : group("Compared with clean speech", [
      card("Clarity · quality · intelligibility", null, "", { wide: true,
        sub: el("div", { class: "card-sub", text: "n/a (no clean reference) — use demo mode" }) }),
    ]);

  root.replaceChildren(
    groundTruth,
    group("Noise", [
      card("Noise removed", m.noise_reduction_db, "dB", { cls: "c-clean",
        title: tip("noise_reduction", "Eq. (19), over frames the energy VAD labels as noise.") }),
      card("Speech lost ↓", m.speech_attenuation_db, "dB", {
        title: tip("speech_attenuation", "Eq. (36). Speech frames also contain noise, so even a perfect method shows a little.") }),
      card("Noise floor (est.)", m.noise_floor_dbfs, "dBFS", { title: "Level implied by |N̂|" }),
      card("SNR estimate out", m.vad_snr_out_db, "dB", { sub: delta(m.vad_snr_in_db, m.vad_snr_out_db, "dB"),
        title: "Reference-free, from the energy VAD (rough)" }),
      card("Speech frames", m.speech_fraction === null ? null : m.speech_fraction * 100, "%", { decimals: 1,
        title: "Fraction of frames the energy VAD labels as speech (Eq. 20)" }),
    ], "reference-free"),
    group("Levels", [
      card("RMS out", m.output_rms_dbfs, "dBFS", { sub: delta(m.input_rms_dbfs, m.output_rms_dbfs, "dB", false) }),
      card("Peak out", m.output_peak_dbfs, "dBFS", { sub: delta(m.input_peak_dbfs, m.output_peak_dbfs, "dB", false) }),
    ]),
    group("Run", [
      card("Processing", m.processing_ms, "ms", { decimals: 1, title: "Server time for this run" }),
      card("CPU cost (RTF) ↓", m.real_time_factor, "×", { decimals: 4,
        title: tip("rtf", "Measured on this server for this run (other load included).") }),
      card("Duration", m.duration_s, "s", { decimals: 3 }),
      card("Sample rate", m.sr, "Hz", { decimals: 0 }),
    ]),
  );
}

/** DNSMOS card group (§14): filled on demand, because scoring takes ~0.1 s per 9 s of audio. */
export function renderDnsmos(root, { available, result, busy, onRun }) {
  const cards = result
    ? [["Overall", "ovrl"], ["Speech", "sig"], ["Background", "bak"]].map(
      ([what, k]) => card(`${what} (${k.toUpperCase()})`, result.output?.[k], "", { decimals: 2,
        sub: delta(result.input?.[k], result.output?.[k], ""), cls: k === "ovrl" ? "c-clean" : "",
        title: tip(`dnsmos_${k}`) }))
    : [card("DNSMOS P.835", null, "", { wide: true, sub: el("div", { class: "card-sub" },
      available ? el("button", { class: "btn btn-sm", type: "button", disabled: busy, onclick: onRun,
        text: busy ? "scoring…" : "Score with DNSMOS" })
        : "n/a: install requirements-ml.txt and run scripts/fetch_models.py") })];
  root.replaceChildren(group("AI listener score (DNSMOS, 1–5)", cards, "no clean speech needed"));
}
