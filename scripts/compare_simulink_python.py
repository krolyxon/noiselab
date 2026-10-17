#!/usr/bin/env python3
"""Compare the Simulink model's output with the Python pipeline on the same input.

    python scripts/compare_simulink_python.py samples/noisy.wav out/cleaned_simulink.wav

1. Runs the Python pipeline on <noisy.wav> with the "Match Simulink basic" preset.
2. Aligns the Simulink output to it by cross-correlation (Simulink's Buffer / delay
   chain adds latency; Python has none).
3. Prints RMS error, correlation, SNR between the two outputs and the max difference,
   both over the full overlap and with the first/last frame excluded (edge frames).
4. Saves a figure (waveforms, spectrograms, difference) and a JSON report.

Expected result for a correct model: SNR between outputs > 40 dB after alignment.
Known sources of small differences are listed in docs/ALGORITHM.md §10.

matplotlib is used only by this script (lazy import); the web app never needs it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.dsp.noise_estimation import estimate_initial  # noqa: E402
from app.dsp.spectral_subtraction import denoise  # noqa: E402
from app.dsp.stft import StftConfig, stft  # noqa: E402
from app.params import PRESETS, validate  # noqa: E402

EPS = 1e-20


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return x.mean(axis=1), int(sr)


def python_pipeline(x: np.ndarray, sr: int) -> np.ndarray:
    """The Python implementation with exactly the Simulink model's settings."""
    p = validate(PRESETS["match_simulink_basic"]["params"])
    cfg = p.stft
    spec = stft(x, cfg)
    noise = estimate_initial(np.abs(spec), len(x), sr, cfg, p.noise_seconds)
    return denoise(x, noise, cfg, p.subtraction, p.output_gain_db, spec=spec).cleaned


def find_lag(ref: np.ndarray, other: np.ndarray, max_lag: int) -> int:
    """Lag L (samples) such that other[n + L] ≈ ref[n]; searched in [−max_lag, max_lag]."""
    c = correlate(other, ref, mode="full", method="fft")
    zero = len(ref) - 1
    lo, hi = max(0, zero - max_lag), min(len(c), zero + max_lag + 1)
    return int(np.argmax(c[lo:hi]) + lo - zero)


def align(ref: np.ndarray, other: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        o = other[lag:]
        r = ref
    else:
        o = other
        r = ref[-lag:]
    n = min(len(r), len(o))
    return r[:n], o[:n]


def compare(p: np.ndarray, s: np.ndarray) -> dict[str, float]:
    d = s - p
    return {
        "samples": int(len(p)),
        "rms_error": float(np.sqrt(np.mean(d**2))),
        "rms_error_dbfs": float(10 * np.log10(np.mean(d**2) + EPS)),
        "max_abs_diff": float(np.max(np.abs(d))),
        "correlation": float(np.corrcoef(p, s)[0, 1]) if np.std(p) > 0 and np.std(s) > 0 else float("nan"),
        "snr_between_outputs_db": float(10 * np.log10(np.sum(p**2) / (np.sum(d**2) + EPS) + EPS)),
        "best_gain_simulink_over_python": float(np.dot(s, p) / (np.dot(p, p) + EPS)),
    }


def verdict(snr: float) -> str:
    if snr > 40:
        return "MATCH: the Simulink and Python implementations agree."
    if snr > 20:
        return "CLOSE: small systematic differences; see docs/ALGORITHM.md §10 (precision, edge frames, noise frames)."
    return ("MISMATCH: check the model against simulink/README.md (window periodic? α, β, T? magnitude "
            "subtraction p = 1? noise frames start at 0, H, 2H…? overlap-add delay?).")


def make_figure(path: Path, sr: int, x: np.ndarray, p: np.ndarray, s: np.ndarray, lag: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = StftConfig(512, 0.5, "hann")
    t = np.arange(len(p)) / sr

    def spec_db(y):
        m = np.abs(stft(y, cfg)).T
        return 20 * np.log10(m / (cfg.n_fft / 4) + 1e-12)

    d = s - p
    sp, ss, sd = spec_db(p), spec_db(s), spec_db(d)
    vmax = np.ceil(sp.max() / 10) * 10
    extent = [0, len(p) / sr, 0, sr / 2]

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, x[: len(t)] if len(x) >= len(t) else np.pad(x, (0, len(t) - len(x))), color="#ffb454",
            lw=0.5, alpha=0.5, label="noisy input")
    ax.plot(t, p, color="#4cc9f0", lw=0.6, label="Python")
    ax.plot(t, s, color="#333333", lw=0.6, ls="--", label=f"Simulink (aligned, lag {lag} samples)")
    ax.set(xlabel="time (s)", ylabel="amplitude", title="Waveforms", xlim=(0, t[-1]))
    ax.legend(loc="upper right", fontsize=8)

    for i, (z, title) in enumerate([(sp, "Python cleaned"), (ss, "Simulink cleaned"), (sd, "Difference (Simulink − Python)")]):
        a = fig.add_subplot(gs[1, i])
        im = a.imshow(z, origin="lower", aspect="auto", extent=extent, cmap="inferno", vmin=vmax - 90, vmax=vmax)
        a.set(title=title, xlabel="time (s)", ylabel="frequency (Hz)" if i == 0 else None)
    fig.colorbar(im, ax=a, label="dBFS")

    a = fig.add_subplot(gs[2, :])
    a.plot(t, d, color="#e56bd5", lw=0.6)
    a.set(xlabel="time (s)", ylabel="difference", title=f"Difference waveform (max |d| = {np.max(np.abs(d)):.2e})",
          xlim=(0, t[-1]))
    fig.suptitle("Simulink vs Python — basic spectral subtraction (N=512, 50 % Hann, α=2, β=0.02, p=1)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("noisy", type=Path, help="the noisy input WAV given to the Simulink model")
    ap.add_argument("simulink", type=Path, help="cleaned_simulink.wav written by simulink/run_and_export.m")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "compare",
                    help="output folder for the figure, JSON report and Python WAV (default: out/compare)")
    ap.add_argument("--max-lag-ms", type=float, default=200.0, help="search range for the alignment lag")
    ap.add_argument("--no-figure", action="store_true", help="skip the matplotlib figure")
    args = ap.parse_args(argv)

    x, sr = load_mono(args.noisy)
    s_raw, sr_s = load_mono(args.simulink)
    if sr != sr_s:
        print(f"error: sample rates differ (noisy {sr} Hz, Simulink {sr_s} Hz). "
              "run_and_export.m must write at the input rate.", file=sys.stderr)
        return 2

    p = python_pipeline(x, sr)
    lag = find_lag(p, s_raw, int(args.max_lag_ms / 1000 * sr))
    pa, sa = align(p, s_raw, lag)

    H, N = 256, 512
    full = compare(pa, sa)
    inner = compare(pa[N:-N], sa[N:-N]) if len(pa) > 4 * N else None

    report = {
        "noisy": str(args.noisy), "simulink": str(args.simulink), "sample_rate": sr,
        "python_samples": len(p), "simulink_samples": len(s_raw),
        "lag_samples": lag, "lag_ms": round(1000 * lag / sr, 3), "lag_hops": round(lag / H, 3),
        "full_overlap": full, "excluding_edge_frames": inner,
        "preset": PRESETS["match_simulink_basic"]["params"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out / "cleaned_python.wav"), np.clip(p, -1, 1), sr, subtype="PCM_16")
    (args.out / "compare_report.json").write_text(json.dumps(report, indent=2))

    print(f"Input:     {args.noisy}  ({len(x)} samples @ {sr} Hz)")
    print(f"Simulink:  {args.simulink}  ({len(s_raw)} samples)")
    print(f"Alignment: lag = {lag} samples = {1000 * lag / sr:.2f} ms = {lag / H:.2f} hops "
          f"(0 if run_and_export.m already removed the overlap-add latency)")
    for name, m in (("Full overlap", full), ("Excluding first/last frame", inner)):
        if m is None:
            continue
        print(f"\n{name} ({m['samples']} samples):")
        print(f"  RMS error              {m['rms_error']:.3e}  ({m['rms_error_dbfs']:.1f} dBFS)")
        print(f"  max |difference|       {m['max_abs_diff']:.3e}")
        print(f"  correlation            {m['correlation']:.6f}")
        print(f"  SNR between outputs    {m['snr_between_outputs_db']:.2f} dB")
        print(f"  best gain Sim/Python   {m['best_gain_simulink_over_python']:.5f}")
    key = inner or full
    print("\n" + verdict(key["snr_between_outputs_db"]))

    if not args.no_figure:
        fig = args.out / "compare_simulink_python.png"
        make_figure(fig, sr, x, pa, sa, lag)
        print(f"Figure:    {fig}")
    print(f"Report:    {args.out / 'compare_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
