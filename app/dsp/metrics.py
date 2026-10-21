"""Objective metrics, Eq. (15)–(20) in docs/ALGORITHM.md.

Reference-based metrics (SNR, SI-SDR, segmental SNR) need the clean signal and
are only reported in demo mode. Reference-free metrics (RMS, VAD-based SNR
estimate, noise reduction) work on any recording.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12
SEG_SNR_RANGE = (-10.0, 35.0)


def to_db(power: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(power, _EPS))


def rms_dbfs(x: np.ndarray) -> float:
    """Eq. (15): 20·log10( sqrt(mean x²) ); 0 dBFS = RMS of 1.0 (a full-scale square wave)."""
    return float(to_db(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def peak_dbfs(x: np.ndarray) -> float:
    return float(to_db(np.max(np.abs(x)) ** 2)) if len(x) else float("-inf")


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Eq. (16): 10·log10( Σ s² / Σ (s − ŝ)² )."""
    err = estimate - clean
    return float(to_db(np.sum(clean**2)) - to_db(np.sum(err**2)))


def si_sdr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Eq. (17): scale-invariant SDR (Le Roux et al., 2019), zero-mean signals."""
    s = clean - clean.mean()
    e = estimate - estimate.mean()
    a = np.dot(e, s) / max(np.dot(s, s), _EPS)
    target = a * s
    return float(to_db(np.sum(target**2)) - to_db(np.sum((e - target) ** 2)))


def _frame_power(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    n = 1 + max(len(x) - frame, 0) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.mean(x[idx] ** 2, axis=1)


def segmental_snr_db(clean: np.ndarray, estimate: np.ndarray, frame: int = 512,
                     hop: int = 256, silence_db: float = -40.0) -> float:
    """Eq. (18): mean over frames of per-frame SNR, clamped to [−10, 35] dB.

    Frames where the clean signal is more than 40 dB below its loudest frame are
    skipped (the synthetic source contains exact digital silence).
    """
    p_s = _frame_power(clean, frame, hop)
    p_e = _frame_power(estimate - clean, frame, hop)
    keep = to_db(p_s) > to_db(p_s.max()) + silence_db
    seg = np.clip(to_db(p_s[keep]) - to_db(p_e[keep]), *SEG_SNR_RANGE)
    return float(seg.mean()) if seg.size else float("nan")


def frame_energy_db(x: np.ndarray, frame: int = 512, hop: int = 256) -> np.ndarray:
    """Per-frame mean power in dBFS (used for the VAD and the energy timeline)."""
    return to_db(_frame_power(np.asarray(x, dtype=np.float64), frame, hop))


def energy_vad(energy_db: np.ndarray, margin_db: float = 3.0, floor_percentile: float = 20.0,
               hangover: int = 3) -> np.ndarray:
    """Eq. (20): frame is speech if E(m) > P20{E} + 3 dB, extended by a short hangover.

    Returns a boolean array (True = speech).
    """
    thr = np.percentile(energy_db, floor_percentile) + margin_db
    speech = energy_db > thr
    if hangover > 0 and speech.any():
        # Keep 'speech' on for `hangover` frames after each active frame.
        kernel = np.ones(hangover + 1)
        speech = np.convolve(speech.astype(float), kernel)[: len(speech)] > 0
    return speech


def vad_snr_db(energy_db: np.ndarray, speech: np.ndarray) -> float | None:
    """Reference-free SNR estimate: 10·log10( (P̄_speech − P̄_noise) / P̄_noise )."""
    if speech.all() or not speech.any():
        return None
    p = 10.0 ** (energy_db / 10.0)
    p_speech, p_noise = p[speech].mean(), p[~speech].mean()
    return float(to_db(max(p_speech - p_noise, _EPS)) - to_db(p_noise))


def noise_reduction_db(noisy_energy_db: np.ndarray, cleaned_energy_db: np.ndarray,
                       speech: np.ndarray) -> float | None:
    """Eq. (19): mean attenuation (dB) over frames the VAD labels as noise."""
    noise = ~speech
    if not noise.any():
        return None
    return float(np.mean(noisy_energy_db[noise] - cleaned_energy_db[noise]))


def speech_attenuation_db(noisy_energy_db: np.ndarray, cleaned_energy_db: np.ndarray,
                          speech: np.ndarray) -> float | None:
    """Eq. (36): mean attenuation (dB) over frames the VAD labels as speech.

    The reference-free counterpart of Eq. (19). A good suppressor has a high noise
    reduction and a speech attenuation close to 0 dB. Together the two show the
    noise-removal vs speech-damage trade-off on recordings without a clean reference.
    (Speech frames still contain noise, so even a perfect method shows a small value here.)
    """
    if not speech.any():
        return None
    return float(np.mean(noisy_energy_db[speech] - cleaned_energy_db[speech]))


def log_spectral_distance_db(clean: np.ndarray, estimate: np.ndarray, frame: int = 512, hop: int = 256,
                             silence_db: float = -40.0, range_db: float = 50.0) -> float:
    """Eq. (35): mean over speech frames of the RMS difference of the log power spectra (dB).

    LSD = mean_m sqrt( mean_k (L_k(m) − L̂_k(m))² ),   L = max(10·log10 |S|², peak_m − R)

    It measures **speech distortion**: how much the spectral shape of the speech was
    changed. 0 dB = identical. Both log spectra are clipped to R = 50 dB below the clean
    frame's peak bin, so differences in bins far below the speech (between harmonics,
    or above its bandwidth) do not dominate. Frames where the clean signal is > 40 dB
    below its loudest frame are skipped, as in Eq. (18). Periodic Hann window.
    """
    w = np.hanning(frame + 1)[:-1]
    n = 1 + max(len(clean) - frame, 0) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    ls = to_db(np.abs(np.fft.rfft(clean[idx] * w, axis=1)) ** 2)
    le = to_db(np.abs(np.fft.rfft(estimate[idx] * w, axis=1)) ** 2)
    e_s = to_db(np.mean(clean[idx] ** 2, axis=1))
    keep = e_s > e_s.max() + silence_db
    lo = ls[keep].max(axis=1, keepdims=True) - range_db
    d = np.maximum(ls[keep], lo) - np.maximum(le[keep], lo)
    return float(np.mean(np.sqrt(np.mean(d**2, axis=1))))
