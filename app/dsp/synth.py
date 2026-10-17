"""Synthetic test signals: a voiced-like clean source, noise generators and an exact-SNR mixer.

Everything is generated from a seed, so the repo needs no third-party audio
(no licence issues) and tests are reproducible.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

NOISE_TYPES = ("white", "pink", "hum50", "fan", "babble", "clicks")

# Rough formant frequencies (Hz) and bandwidths for a few vowels.
_VOWELS = {
    "a": ((730, 90), (1090, 110), (2440, 160)),
    "i": ((270, 60), (2290, 100), (3010, 200)),
    "u": ((300, 60), (870, 90), (2240, 150)),
    "e": ((530, 70), (1840, 110), (2480, 160)),
    "o": ((570, 80), (840, 90), (2410, 160)),
}


def _resonator(x: np.ndarray, f: float, bw: float, sr: int) -> np.ndarray:
    """Second-order all-pole resonator (one formant)."""
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * f / sr
    a = [1.0, -2 * r * np.cos(theta), r * r]
    return lfilter([1.0 - r], a, x)


def clean_voiced(sr: int = 16000, duration: float = 6.0, seed: int = 0,
                 lead_silence: float = 0.8) -> np.ndarray:
    """Harmonic, speech-like signal: moving pitch, vowel formants, syllable envelope.

    The first `lead_silence` seconds are exactly zero, so the `initial` noise
    mode (default 0.5 s) sees a noise-only segment once noise is added.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * sr))
    t = np.arange(n) / sr

    # Pitch contour: slow sinusoidal intonation + random drift, ~90–170 Hz.
    drift = np.cumsum(rng.normal(0, 1, n)) / np.sqrt(sr) * 8.0
    f0 = 125 + 25 * np.sin(2 * np.pi * 0.4 * t) + drift
    f0 = np.clip(f0, 85, 180)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    out = np.zeros(n)
    pos = int(lead_silence * sr)
    vowels = list(_VOWELS)
    while pos < n - int(0.15 * sr):
        syl_len = int(rng.uniform(0.15, 0.32) * sr)
        gap = int(rng.uniform(0.03, 0.10) * sr)
        if rng.random() < 0.2:                       # occasional word pause
            gap += int(rng.uniform(0.15, 0.35) * sr)
        end = min(pos + syl_len, n)
        seg = slice(pos, end)

        # Harmonic source with 1/k roll-off, harmonics kept below 0.45·sr.
        src = np.zeros(end - pos)
        for k in range(1, 60):
            hk = k * f0[seg]
            src += np.where(hk < 0.45 * sr, np.sin(k * phase[seg]) / k, 0.0)

        filt = np.zeros_like(src)
        for f, bw in _VOWELS[vowels[rng.integers(len(vowels))]]:
            filt += _resonator(src, f, bw, sr)
        env = np.hanning(len(filt)) ** 0.7 * rng.uniform(0.6, 1.0)
        out[seg] += filt * env
        pos = end + gap

    peak = np.max(np.abs(out))
    return (out / peak * 0.5 if peak > 0 else out).astype(np.float64)


def _unit_rms(x: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.mean(x**2))
    return x / r if r > 0 else x


def make_noise(kind: str, n: int, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Noise of the given type, unit RMS, length n."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    if kind == "white":
        x = rng.normal(0, 1, n)
    elif kind == "pink":
        # Shape white noise by 1/sqrt(f) in the frequency domain (−3 dB/octave).
        spec = np.fft.rfft(rng.normal(0, 1, n))
        f = np.fft.rfftfreq(n, 1 / sr)
        spec[1:] /= np.sqrt(f[1:])
        spec[0] = 0
        x = np.fft.irfft(spec, n)
    elif kind == "hum50":
        # 50 Hz mains hum (India) + harmonics, slight random phases, small hiss.
        x = np.zeros(n)
        for h, amp in zip(range(1, 9), (1.0, 0.5, 0.6, 0.25, 0.3, 0.12, 0.1, 0.05)):
            x += amp * np.sin(2 * np.pi * 50 * h * t + rng.uniform(0, 2 * np.pi))
        x += 0.02 * rng.normal(0, 1, n)
    elif kind == "fan":
        # Fan / AC: low-passed broadband rumble + a weak blade-pass tone + hiss.
        sos = butter(4, 500, btype="low", fs=sr, output="sos")
        x = _unit_rms(sosfilt(sos, rng.normal(0, 1, n)))
        x += 0.15 * np.sin(2 * np.pi * 120 * t) + 0.1 * rng.normal(0, 1, n)
    elif kind == "babble":
        # Babble-ish: overlapping band-limited noise bursts with syllable-rate envelopes.
        x = np.zeros(n)
        for _ in range(8):
            lo = rng.uniform(250, 600)
            hi = rng.uniform(1500, 3400)
            sos = butter(2, [lo, min(hi, 0.45 * sr)], btype="band", fs=sr, output="sos")
            burst = sosfilt(sos, rng.normal(0, 1, n))
            rate = rng.uniform(2.5, 5.0)
            env = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi))
            x += burst * env**2
    elif kind == "clicks":
        # Sparse decaying impulses (~6 per second) over a faint noise floor.
        x = 0.05 * rng.normal(0, 1, n)
        decay = np.exp(-np.arange(int(0.004 * sr)) / (0.0008 * sr))
        for pos in rng.integers(0, max(n - len(decay), 1), size=max(1, int(6 * n / sr))):
            x[pos : pos + len(decay)] += rng.choice([-1, 1]) * rng.uniform(2, 6) * decay
    else:
        raise ValueError(f"unknown noise type {kind!r}; expected one of {NOISE_TYPES}")
    return _unit_rms(x)


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale noise so that 10·log10(P_s / P_n) = snr_db exactly, Eq. (21).

    Returns (noisy, scaled_noise). The noise is tiled/trimmed to len(clean).
    """
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.resize(np.asarray(noise, dtype=np.float64), clean.shape)
    p_s = np.mean(clean**2)
    p_n = np.mean(noise**2)
    if p_s == 0 or p_n == 0:
        raise ValueError("clean and noise must both have non-zero power")
    scaled = noise * np.sqrt(p_s / (p_n * 10.0 ** (snr_db / 10.0)))
    return clean + scaled, scaled


def make_demo(noise_type: str = "white", snr_db: float = 5.0, sr: int = 16000,
              duration: float = 6.0, seed: int = 0, clean: np.ndarray | None = None
              ) -> dict[str, np.ndarray]:
    """Clean + noise + noisy, jointly scaled (same factor, so SNR is preserved) to peak ≤ 0.9."""
    if clean is None:
        clean = clean_voiced(sr, duration, seed)
    noise = make_noise(noise_type, len(clean), sr, seed + 1)
    noisy, noise = mix(clean, noise, snr_db)
    peak = np.max(np.abs(noisy))
    k = 0.9 / peak if peak > 0.9 else 1.0
    return {"clean": clean * k, "noise": noise * k, "noisy": noisy * k}
