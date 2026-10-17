"""Emulate the basic Simulink block chain frame-by-frame and compare with the Python pipeline.

This checks the *design* of the Simulink model (simulink/README.md block table and
simulink/spectral_subtraction_reference.m) — not MATLAB syntax, which cannot be run here.
Each line below corresponds to one block:

  source (H samples/step) → Buffer (N, overlap N−H, ic 0) → × Hann → FFT → |·|, ∠
  → max(|Y| − α·noiseMag, β·noiseMag) → Mag-Angle to Complex → IFFT → Re
  → [first half] + UnitDelay([second half]) → output (H samples/step)
"""

import numpy as np
import pytest

from app.dsp import synth
from app.dsp.noise_estimation import estimate_initial
from app.dsp.spectral_subtraction import denoise
from app.dsp.stft import StftConfig, stft
from app.params import PRESETS, validate

SIM = validate(PRESETS["match_simulink_basic"]["params"])


def compute_noise_mag(x, N, H, noise_seconds, fs):
    """Port of simulink/nl_compute_noise_mag.m (full N-point spectrum)."""
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)
    last = min(max(round(noise_seconds * fs), N), len(x))
    starts = range(0, last - N + 1, H)
    return np.mean([np.abs(np.fft.fft(x[s:s + N] * w)) for s in starts], axis=0)


def simulink_chain(x, fs, N=512, alpha=2.0, beta=0.02, noise_seconds=0.5):
    """Port of simulink/spectral_subtraction_reference.m (block-by-block)."""
    H = N // 2
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)
    noise_mag = compute_noise_mag(x, N, H, noise_seconds, fs)
    n_steps = int(np.ceil(len(x) / H)) + 1
    xin = np.concatenate([x, np.zeros(n_steps * H - len(x))])
    buf = np.zeros(N)
    prev_second = np.zeros(H)
    out = np.zeros(n_steps * H)
    for k in range(n_steps):
        hop = xin[k * H:(k + 1) * H]                       # source block
        buf = np.concatenate([buf[H:], hop])               # Buffer
        Y = np.fft.fft(buf * w)                            # Product + FFT
        mag, ph = np.abs(Y), np.angle(Y)                   # Complex to Magnitude-Angle
        S = np.maximum(mag - alpha * noise_mag, beta * noise_mag)   # Subtract, Gain, MinMax
        frame = np.real(np.fft.ifft(S * np.exp(1j * ph)))  # Mag-Angle to Complex, IFFT, Re
        out[k * H:(k + 1) * H] = frame[:H] + prev_second   # Selector ×2, Unit Delay, Add
        prev_second = frame[H:]
    return out, noise_mag, H


@pytest.mark.parametrize("kind,snr", [("white", 5.0), ("hum50", 0.0), ("babble", 10.0)])
def test_block_chain_equals_python_pipeline(kind, snr):
    fs = 16000
    x = synth.make_demo(kind, snr, fs, duration=3.0, seed=4)["noisy"]
    out, noise_full, H = simulink_chain(x, fs)

    # Python pipeline with the "Match Simulink basic" preset
    cfg = SIM.stft
    spec = stft(x, cfg)
    noise_py = estimate_initial(np.abs(spec), len(x), fs, cfg, SIM.noise_seconds)
    res = denoise(x, noise_py, cfg, SIM.subtraction, SIM.output_gain_db, spec=spec)

    # 1. same noise estimate (Python keeps the one-sided half of the N-point vector)
    assert np.allclose(noise_full[: cfg.n_bins], noise_py, rtol=1e-12, atol=1e-12)
    # 2. the chain output is the Python output delayed by exactly one hop (H samples)
    sim = out[H:H + len(x)]
    assert np.max(np.abs(sim - res.cleaned_raw)) < 1e-10
    # 3. and the delay really is H: without the shift they differ
    assert np.max(np.abs(out[: len(x)] - res.cleaned_raw)) > 1e-3


def test_preset_matches_matlab_config():
    """Values in simulink/nl_default_config.m must equal the 'Match Simulink basic' preset."""
    from pathlib import Path
    import re

    text = (Path(__file__).resolve().parents[1] / "simulink" / "nl_default_config.m").read_text()
    val = lambda name: float(re.search(rf"cfg\.{name}\s*=\s*([0-9.]+)", text).group(1))
    assert val("N") == SIM.n_fft
    assert val("overlap") == SIM.overlap
    assert val("noiseSeconds") == SIM.noise_seconds
    assert val("alpha") == SIM.alpha
    assert val("beta") == SIM.beta
    assert SIM.exponent == 1 and SIM.window == "hann" and SIM.noise_mode == "initial"


def test_compare_script_on_emulated_simulink_output(tmp_path, capsys):
    """Feed the comparison script a 'Simulink' file produced by the block-chain emulation
    WITHOUT latency removal (worst case) — it must find the 1-hop lag and report a match."""
    import soundfile as sf

    from scripts.compare_simulink_python import main

    fs = 16000
    x = synth.make_demo("pink", 5.0, fs, duration=4.0, seed=11)["noisy"]
    out, _, H = simulink_chain(x, fs)
    noisy, sim = tmp_path / "noisy.wav", tmp_path / "cleaned_simulink.wav"
    sf.write(noisy, x, fs, subtype="FLOAT")
    sf.write(sim, np.clip(out, -1, 1), fs, subtype="PCM_16")      # 16-bit, like run_and_export.m

    assert main([str(noisy), str(sim), "--out", str(tmp_path / "cmp")]) == 0
    text = capsys.readouterr().out
    assert "lag = 256 samples" in text and "\nMATCH:" in text
    report = __import__("json").loads((tmp_path / "cmp" / "compare_report.json").read_text())
    assert report["lag_samples"] == H
    assert report["excluding_edge_frames"]["snr_between_outputs_db"] > 40       # 16-bit quantisation only
    assert (tmp_path / "cmp" / "compare_simulink_python.png").stat().st_size > 10_000


def test_compare_script_detects_a_wrong_model(tmp_path, capsys):
    """A 'model' with the wrong α must not be reported as a match."""
    import soundfile as sf

    from scripts.compare_simulink_python import main

    fs = 16000
    x = synth.make_demo("white", 5.0, fs, duration=3.0, seed=2)["noisy"]
    out, _, H = simulink_chain(x, fs, alpha=4.0)
    noisy, sim = tmp_path / "n.wav", tmp_path / "s.wav"
    sf.write(noisy, x, fs, subtype="FLOAT")
    sf.write(sim, np.clip(out[H:H + len(x)], -1, 1), fs, subtype="FLOAT")
    main([str(noisy), str(sim), "--out", str(tmp_path / "cmp"), "--no-figure"])
    text = capsys.readouterr().out
    assert "\nMATCH:" not in text and ("MISMATCH:" in text or "CLOSE:" in text)
