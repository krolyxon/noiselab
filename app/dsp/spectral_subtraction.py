"""Spectral subtraction (Boll 1979, Berouti et al. 1979) and a Wiener-style variant.

Equation numbers refer to docs/ALGORITHM.md. All functions are pure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from .stft import StftConfig, istft, stft

METHODS = ("spectral_subtraction", "wiener")
_EPS = 1e-12


@dataclass(frozen=True)
class SubtractionParams:
    alpha: float = 2.0             # over-subtraction factor α
    beta: float = 0.02             # spectral floor β
    exponent: int = 2              # p: 1 = magnitude, 2 = power subtraction
    method: str = "spectral_subtraction"
    gain_smooth: float = 0.0       # extended: time smoothing λ of G
    freq_smooth_bins: int = 0      # extended: moving-average width over frequency


def subtracted_magnitude(mag: np.ndarray, noise_mag: np.ndarray, alpha: float,
                         beta: float, exponent: int) -> np.ndarray:
    """Eq. (7): |S̃|^p = max( |Y|^p − α·|N̂|^p , β·|N̂|^p )."""
    p = float(exponent)
    y_p = mag**p
    n_p = noise_mag[None, :] ** p
    return np.maximum(y_p - alpha * n_p, beta * n_p) ** (1.0 / p)


def spectral_subtraction_gain(mag: np.ndarray, noise_mag: np.ndarray, alpha: float,
                              beta: float, exponent: int) -> np.ndarray:
    """Eq. (8): G = |S̃| / |Y|."""
    s_mag = subtracted_magnitude(mag, noise_mag, alpha, beta, exponent)
    return s_mag / np.maximum(mag, _EPS)


def wiener_gain(mag: np.ndarray, noise_mag: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Eq. (9): G = max( 1 − α·|N̂|²/|Y|² , β )."""
    ratio = noise_mag[None, :] ** 2 / np.maximum(mag**2, _EPS)
    return np.maximum(1.0 - alpha * ratio, beta)


def smooth_gain(gain: np.ndarray, gain_smooth: float, freq_smooth_bins: int) -> np.ndarray:
    """Extended refinements (off by default) to reduce musical noise.

    Eq. (10), time:      G_s(m) = λ·G_s(m−1) + (1 − λ)·G(m)
    Eq. (11), frequency: moving average of width (freq_smooth_bins) along k

    Before smoothing, G is capped at 1. The floor in Eq. (7) is absolute, so in
    bins where |Y| < √β·|N̂| the gain exceeds 1 (it *fills* the bin up to the
    floor). That is harmless per bin, but smoothing would spread those large
    gains into neighbouring bins/frames and amplify them. The basic (unsmoothed)
    path is left untouched so it still matches the Simulink model exactly.
    """
    if gain_smooth <= 0.0 and freq_smooth_bins <= 1:
        return gain
    g = np.minimum(gain, 1.0)
    if gain_smooth > 0.0:
        lam = float(gain_smooth)
        for m in range(1, g.shape[0]):
            g[m] = lam * g[m - 1] + (1.0 - lam) * g[m]
    if freq_smooth_bins > 1:
        g = uniform_filter1d(g, size=int(freq_smooth_bins), axis=1, mode="nearest")
    return g


def compute_gain(mag: np.ndarray, noise_mag: np.ndarray, params: SubtractionParams) -> np.ndarray:
    """Gain mask G(k, m) for the chosen method, followed by optional smoothing."""
    if params.method == "spectral_subtraction":
        g = spectral_subtraction_gain(mag, noise_mag, params.alpha, params.beta, params.exponent)
    elif params.method == "wiener":
        g = wiener_gain(mag, noise_mag, params.alpha, params.beta)
    else:
        raise ValueError(f"unknown method {params.method!r}; expected one of {METHODS}")
    return smooth_gain(g, params.gain_smooth, params.freq_smooth_bins)


def output_stage(x: np.ndarray, output_gain_db: float = 0.0, peak_limit: float = 0.99) -> np.ndarray:
    """Eq. (14): apply output gain, then scale down only if the peak exceeds 0.99."""
    x = x * 10.0 ** (output_gain_db / 20.0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > peak_limit:
        x = x * (peak_limit / peak)
    return x


@dataclass(frozen=True)
class DenoiseResult:
    cleaned: np.ndarray        # ŝ[n] after the output stage, same length as input
    cleaned_raw: np.ndarray    # ŝ[n] before the output stage (used for y − ŝ)
    gain: np.ndarray           # G(k, m), shape (n_frames, n_bins)
    noise_mag: np.ndarray      # |N̂_k|, shape (n_bins,)


def denoise_spectrum(spec: np.ndarray, noise_mag: np.ndarray, params: SubtractionParams
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Steps 2–6 on a precomputed STFT. Returns (Ŝ, G).

    Eq. (12): Ŝ = G · Y = |S̃| · e^{j∠Y} — the noisy phase is kept unchanged,
    because multiplying by a real, non-negative gain only scales the magnitude.
    """
    gain = compute_gain(np.abs(spec), noise_mag, params)
    return gain * spec, gain


def denoise(y: np.ndarray, noise_mag: np.ndarray, cfg: StftConfig, params: SubtractionParams,
            output_gain_db: float = 0.0, spec: np.ndarray | None = None) -> DenoiseResult:
    """Full chain: STFT → gain → noisy-phase reconstruction → ISTFT → output stage.

    `spec` may be passed in to reuse a cached STFT of `y` (fast reprocessing).
    """
    if spec is None:
        spec = stft(y, cfg)
    s_hat, gain = denoise_spectrum(spec, noise_mag, params)
    raw = istft(s_hat, len(y), cfg)
    return DenoiseResult(
        cleaned=output_stage(raw, output_gain_db),
        cleaned_raw=raw,
        gain=gain,
        noise_mag=noise_mag,
    )
