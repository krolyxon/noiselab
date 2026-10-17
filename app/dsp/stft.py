"""Framing, windowing, STFT and overlap-add ISTFT.

Equation numbers refer to docs/ALGORITHM.md.

Conventions
-----------
* Spectra are stored as complex arrays of shape ``(n_frames, n_bins)`` with
  ``n_bins = n_fft // 2 + 1`` (one-sided spectrum of a real signal).
* The signal is zero-padded by ``n_fft - hop`` samples at the start, so the
  first frame that contains real audio starts at original sample 0 and every
  original sample is covered by the same number of frames (needed for exact
  reconstruction at the edges).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window

WINDOWS = ("hann", "hamming", "sqrt_hann")
_ENV_EPS = 1e-10


@dataclass(frozen=True)
class StftConfig:
    n_fft: int = 512
    overlap: float = 0.5
    window: str = "hann"

    @property
    def hop(self) -> int:
        # Eq. (1): H = N · (1 − overlap)
        return int(round(self.n_fft * (1.0 - self.overlap)))

    @property
    def pad(self) -> int:
        return self.n_fft - self.hop

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def key(self) -> tuple[int, float, str]:
        return (self.n_fft, self.overlap, self.window)


def analysis_window(cfg: StftConfig) -> np.ndarray:
    """Periodic ("DFT-even") analysis window w[n], Eq. (2).

    Periodic windows are used because a periodic Hann at 50 % or 75 % overlap
    sums to an exact constant, which makes overlap-add reconstruction exact.
    MATLAB equivalent: ``hann(N, 'periodic')``.
    """
    if cfg.window == "hann":
        return get_window("hann", cfg.n_fft, fftbins=True)
    if cfg.window == "hamming":
        return get_window("hamming", cfg.n_fft, fftbins=True)
    if cfg.window == "sqrt_hann":
        return np.sqrt(get_window("hann", cfg.n_fft, fftbins=True))
    raise ValueError(f"unknown window {cfg.window!r}; expected one of {WINDOWS}")


def uses_synthesis_window(cfg: StftConfig) -> bool:
    """sqrt-Hann is applied twice (analysis + synthesis); the others once."""
    return cfg.window == "sqrt_hann"


def n_frames_for(length: int, cfg: StftConfig) -> int:
    padded = cfg.pad + length + cfg.pad
    return 1 + int(np.ceil(max(padded - cfg.n_fft, 0) / cfg.hop))


def frame_starts(n_frames: int, cfg: StftConfig) -> np.ndarray:
    """Start index of each frame in *original* (unpadded) sample coordinates."""
    return np.arange(n_frames) * cfg.hop - cfg.pad


def frame_times(n_frames: int, sr: int, cfg: StftConfig) -> np.ndarray:
    """Centre time (s) of each frame, for plotting."""
    return (frame_starts(n_frames, cfg) + cfg.n_fft / 2) / sr


def _frame(y: np.ndarray, cfg: StftConfig) -> np.ndarray:
    """Eq. (1): y_m[n] = y[mH + n], 0 ≤ n < N (on the zero-padded signal)."""
    n_frames = n_frames_for(len(y), cfg)
    total = (n_frames - 1) * cfg.hop + cfg.n_fft
    padded = np.zeros(total, dtype=np.float64)
    padded[cfg.pad : cfg.pad + len(y)] = y
    idx = np.arange(cfg.n_fft)[None, :] + cfg.hop * np.arange(n_frames)[:, None]
    return padded[idx]


def stft(y: np.ndarray, cfg: StftConfig) -> np.ndarray:
    """Eq. (2): Y_k(m) = FFT{ w · y_m }. Returns shape (n_frames, n_bins)."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("stft expects a 1-D mono signal")
    frames = _frame(y, cfg) * analysis_window(cfg)[None, :]
    return np.fft.rfft(frames, n=cfg.n_fft, axis=1)


def istft(spec: np.ndarray, length: int, cfg: StftConfig) -> np.ndarray:
    """Overlap-add inverse STFT with window normalisation, Eq. (13).

    x̂[n] = Σ_m v[n − mH] · IFFT{Ŝ(m)}[n − mH]  /  Σ_m w[n − mH] · v[n − mH]

    where v = 1 (no synthesis window) for Hann/Hamming and v = w for sqrt-Hann.
    For periodic Hann at 50 % overlap the denominator is exactly 1, so this is
    plain overlap-add — identical to what the Simulink model does.
    """
    n_frames = spec.shape[0]
    w = analysis_window(cfg)
    v = w if uses_synthesis_window(cfg) else np.ones_like(w)
    frames = np.fft.irfft(spec, n=cfg.n_fft, axis=1) * v[None, :]

    total = (n_frames - 1) * cfg.hop + cfg.n_fft
    out = np.zeros(total)
    env = np.zeros(total)
    wv = w * v
    for m in range(n_frames):
        s = m * cfg.hop
        out[s : s + cfg.n_fft] += frames[m]
        env[s : s + cfg.n_fft] += wv

    out = np.divide(out, env, out=np.zeros_like(out), where=env > _ENV_EPS)
    return out[cfg.pad : cfg.pad + length]
