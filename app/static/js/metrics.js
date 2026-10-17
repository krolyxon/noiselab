// Metric cards: monospace values, fixed decimals, units, delta arrows. "n/a" when not computable.

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
    ? group("Ground truth (demo)", [
      card("SNR out", gt.snr_out_db, "dB", { sub: delta(gt.snr_in_db, gt.snr_out_db, "dB"), cls: "c-clean",
        title: "Eq. (16), output vs clean reference" }),
      card("SI-SDR out", gt.si_sdr_out_db, "dB", { sub: delta(gt.si_sdr_in_db, gt.si_sdr_out_db, "dB"), cls: "c-clean",
        title: "Eq. (17), scale-invariant SDR" }),
      card("Segmental SNR", gt.seg_snr_out_db, "dB", { sub: delta(gt.seg_snr_in_db, gt.seg_snr_out_db, "dB"),
        wide: true, title: "Eq. (18), per-frame SNR clamped to [−10, 35] dB" }),
    ], "vs clean reference")
    : group("Ground truth", [
      card("SNR · SI-SDR · seg. SNR", null, "", { wide: true,
        sub: el("div", { class: "card-sub", text: "n/a (no clean reference) — use demo mode" }) }),
    ]);

  root.replaceChildren(
    groundTruth,
    group("Noise", [
      card("Noise reduction", m.noise_reduction_db, "dB", { cls: "c-clean",
        title: "Eq. (19): mean attenuation over VAD noise frames" }),
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
      card("Real-time factor", m.real_time_factor, "×", { decimals: 4, title: "processing time / clip duration" }),
      card("Duration", m.duration_s, "s", { decimals: 3 }),
      card("Sample rate", m.sr, "Hz", { decimals: 0 }),
    ]),
  );
}
