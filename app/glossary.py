"""Plain-language vocabulary for every metric and term shown to readers.

One source of wording for the report figures (scripts/report_figures.py), the summary,
the web UI (/api/glossary) and docs/GLOSSARY.md, so a term is always explained the same
way. Rule: plain words first, the standard technical name second (for the literature and
the viva). Pure Python, no Flask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Term:
    short: str          # plain label for an axis / column / card
    tech: str           # standard technical name
    better: str         # "higher", "lower" or "" (not a score)
    explain: str        # one or two simple sentences
    analogy: str = ""   # everyday comparison (docs + glossary dialog)
    unit: str = ""


METRICS: dict[str, Term] = {
    "d_si_sdr": Term(
        "Speech clarity gain", "ΔSI-SDR", "higher",
        "How much closer the cleaned waveform is to the clean speech than the noisy input was. "
        "+3 dB means the remaining error energy was halved; +10 dB means it is 10× smaller.",
        "Like counting spelling mistakes before and after proof-reading, on the waveform.", "dB"),
    "d_snr": Term(
        "Signal-to-noise gain", "ΔSNR", "higher",
        "How much louder the speech is than the leftover noise, compared with before cleaning.",
        "How much easier it became to hear the speaker over the background.", "dB"),
    "pesq": Term(
        "Sound-quality score", "PESQ", "higher",
        "An ITU-standard program that predicts how listeners would rate the quality, on a "
        "1 (bad) to 4.5 (excellent) scale. Needs the clean speech for comparison.",
        "An automatic judge trained to grade audio the way people do.", "1–4.5"),
    "d_pesq": Term(
        "Sound-quality improvement", "ΔPESQ", "higher",
        "How many points the sound-quality score (PESQ, 1–4.5) went up after cleaning.",
        "Going from a 1.5-star to a 2.5-star recording is +1.0.", "points"),
    "estoi": Term(
        "Intelligibility score", "ESTOI", "higher",
        "Predicts what fraction of the words a listener would understand (0 to 1). "
        "Needs the clean speech for comparison.",
        "0.8 ≈ about 8 words in 10 understood.", "0–1"),
    "d_estoi": Term(
        "Intelligibility improvement", "ΔESTOI", "higher",
        "How much the predicted fraction of understood words went up (0.1 = 10 percentage points).",
        "", "0–1"),
    "lsd": Term(
        "Speech distortion", "LSD = log-spectral distance", "lower",
        "How much the shape of the speech spectrum differs from the clean speech. "
        "Noise left over and speech damaged both increase it.",
        "How much the 'colour' of the voice was changed.", "dB"),
    "noise_reduction": Term(
        "Noise removed in pauses", "NR = noise reduction", "higher",
        "How much quieter the gaps between words became. Does not say whether the speech survived.",
        "Turning down the background between sentences.", "dB"),
    "speech_attenuation": Term(
        "Speech lost", "speech attenuation", "lower",
        "How much quieter the speech itself became: the damage side of noise removal.",
        "Accidentally turning down the speaker too.", "dB"),
    "dnsmos_ovrl": Term(
        "AI listener score (overall)", "DNSMOS OVRL", "higher",
        "A neural network trained on thousands of human ratings guesses the overall rating (1–5). "
        "Needs no clean reference, so it works on your own recordings, but it is a prediction.",
        "Asking a trained AI critic instead of a panel of listeners.", "1–5"),
    "dnsmos_sig": Term(
        "AI listener score (speech)", "DNSMOS SIG", "higher",
        "Predicted rating of how natural and undamaged the speech sounds (1–5).", "", "1–5"),
    "dnsmos_bak": Term(
        "AI listener score (background)", "DNSMOS BAK", "higher",
        "Predicted rating of how unobtrusive the background noise is (1–5).", "", "1–5"),
    "d_dnsmos": Term(
        "AI listener score improvement", "ΔDNSMOS OVRL", "higher",
        "How much the AI listener's overall rating went up after cleaning.", "", "points"),
    "rtf": Term(
        "CPU cost", "RTF = real-time factor", "lower",
        "Processing time ÷ audio length on one CPU core. 0.01 = 1 % of one core; above 1 = cannot run live.",
        "0.02 means 1 minute of audio is cleaned in about 1.2 seconds.", "fraction of one core"),
    "snr_in": Term(
        "Starting noise level", "input SNR", "",
        "How loud the speech is compared with the noise before cleaning. Lower = noisier: "
        "0 dB = equally loud, −5 dB = noise louder than speech, 20 dB = only faint noise.",
        "", "dB"),
    "seg_snr": Term(
        "Frame-by-frame signal-to-noise", "segmental SNR", "higher",
        "Signal-to-noise ratio measured on short pieces and averaged, so quiet words count too.", "", "dB"),
}

TERMS: dict[str, Term] = {
    "ci": Term(
        "± uncertainty", "95 % confidence interval", "",
        "The true average very likely (95 %) lies within ± this value. Overlapping ranges = no clear winner."),
    "wilcoxon": Term(
        "Is the difference real?", "Wilcoxon signed-rank p-value", "",
        "Tests on the same clips whether the winner is really better. p < 0.05 = yes, very likely; "
        "p ≥ 0.05 = could be chance.",
        "Like checking that a cricketer's higher average is not just luck from a few innings."),
    "pareto": Term(
        "Best value-for-cost line", "Pareto front", "",
        "Methods on this line are not beaten by anything that is both cheaper and better."),
    "effective_mask": Term(
        "Gain the model actually applied", "effective gain mask", "",
        "For ready-made models that do not report their gain, it is measured as output ÷ input in "
        "each time–frequency cell (0 = removed, 1 = kept)."),
    "gain_mask": Term(
        "Keep/remove map", "gain mask G(k, m)", "",
        "For every time and frequency: 1 = keep, 0 = remove. The algorithm's decision, as a picture."),
    "stationary": Term(
        "Steady vs changing noise", "stationary / non-stationary", "",
        "Steady noise (fan, hum, hiss) sounds the same all the time; changing noise (babble, clicks, "
        "traffic) does not. Classical methods assume steady noise."),
    "musical_noise": Term(
        "Musical noise", "musical noise", "",
        "Short random tones ('twinkling') left behind when noise is removed bin by bin."),
}

FAMILIES: dict[str, Term] = {
    "classical": Term("Classical: our Simulink method", "spectral subtraction", "",
                      "Measure the average noise spectrum and subtract it."),
    "statistical": Term("Statistical: probability-based (hand-written)", "MMSE / Wiener estimators", "",
                        "Smarter subtraction that weighs how likely each piece is to be speech."),
    "library": Term("Ready-made DSP library", "DSP baselines", "",
                    "Well-known noise suppressors from existing software (no neural network)."),
    "deep": Term("Neural network (pretrained)", "deep learning", "",
                 "A network trained on many hours of noisy speech learns what to remove."),
}

# One-line plain description of every algorithm (UI dropdown, figure 0, docs).
ALGORITHMS: dict[str, str] = {
    "spectral_subtraction": "Measures the average noise in the first half-second and subtracts it from every "
                            "frame. Our Simulink method.",
    "wiener_dd": "Like subtraction, but turns each frequency down by a smooth fraction based on how likely it is "
                 "to be speech. Less 'twinkling' leftover noise.",
    "mmse_stsa": "Probability-based: picks the gain that makes the cleaned loudness of each frequency closest "
                 "to the clean speech, on average.",
    "log_mmse": "Same idea as MMSE-STSA, but matches loudness on a log (decibel) scale, closer to how we hear.",
    "omlsa": "Log-MMSE that also keeps re-measuring the noise while you talk, so it follows noise that changes.",
    "noisereduce": "A noise gate per frequency: anything below a learned noise threshold is muted.",
    "webrtc_ns": "The noise suppressor built into Chrome / Google Meet calls (classic signal processing).",
    "rnnoise": "A very small neural network (used in Jitsi, OBS) that turns 22 frequency bands up or down.",
    "nsnet2": "Microsoft's neural network for noise suppression (from their DNS Challenge).",
    "deepfilternet3": "A modern neural network that also repairs the harmonics of the voice. Runs live on a laptop.",
    "demucs": "Meta's large neural network that rebuilds the clean waveform directly. Best quality, most CPU.",
}

# Plain names of swept / tuned parameters (figure 7, UI tooltips).
PARAMS: dict[str, str] = {
    "alpha": "α: how much noise to subtract",
    "dd_alpha": "α_dd: how smooth the decision is over time",
    "g_min_db": "G_min: how much to turn down noise-only parts (dB)",
    "prop_decrease": "how much of the gated noise to remove",
    "level": "suppression strength (1 mild … 4 very high)",
    "atten_limit_db": "max. noise removal allowed (dB)",
}


def label(key: str, *, tech: bool = True, unit: bool = True, arrow: bool = True) -> str:
    """'Speech clarity gain, dB (ΔSI-SDR) ↑' — plain first, technical name in brackets."""
    t = METRICS[key]
    s = t.short + (f", {t.unit}" if unit and t.unit else "")
    if tech:
        s += f" ({t.tech})"
    if arrow and t.better:
        s += " ↑ better" if t.better == "higher" else " ↓ better"
    return s


def average(text: str) -> str:
    """'Average speech clarity gain…' (keeps acronyms such as 'AI' intact)."""
    if len(text) > 1 and text[1].islower():
        text = text[0].lower() + text[1:]
    return "Average " + text


GLOSSARY_ORDER = ("snr_in", "d_si_sdr", "d_snr", "seg_snr", "pesq", "d_pesq", "estoi", "d_estoi", "lsd",
                  "noise_reduction", "speech_attenuation", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "d_dnsmos",
                  "rtf")


def as_json() -> dict:
    """Dicts keyed by id (lookup) plus `order` lists, because JSON object order is not guaranteed."""
    return {"metrics": {k: asdict(v) for k, v in METRICS.items()},
            "terms": {k: asdict(v) for k, v in TERMS.items()},
            "families": {k: asdict(v) for k, v in FAMILIES.items()},
            "params": PARAMS, "algorithms": ALGORITHMS,
            "order": {"metrics": list(GLOSSARY_ORDER), "terms": list(TERMS), "families": list(FAMILIES),
                      "algorithms": list(ALGORITHMS)}}
