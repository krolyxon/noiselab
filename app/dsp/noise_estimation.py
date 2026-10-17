"""Noise magnitude estimators |N̂_k|, Eq. (4)–(6) in docs/ALGORITHM.md.

Every estimator returns a one-sided magnitude vector of length n_fft//2 + 1.
(The Simulink model stores the full N-point spectrum; the extra bins are just
the mirror image k → N − k of these, so the two are equivalent.)
"""

from __future__ import annotations

import numpy as np

from .stft import StftConfig, frame_starts, stft

NOISE_MODES = ("initial", "region", "profile", "minstats")


def _frames_inside(n_frames: int, n_samples: int, cfg: StftConfig, t0: int, t1: int) -> np.ndarray:
    """Indices of frames lying entirely inside original samples [t0, t1)."""
    starts = frame_starts(n_frames, cfg)
    inside = (starts >= max(t0, 0)) & (starts + cfg.n_fft <= min(t1, n_samples))
    return np.flatnonzero(inside)


def mean_magnitude(mag: np.ndarray, frame_idx: np.ndarray) -> np.ndarray:
    """Eq. (4): |N̂_k| = (1/M) Σ_{m ∈ frames} |Y_k(m)|."""
    if frame_idx.size == 0:
        raise ValueError("no complete frames in the noise segment")
    return mag[frame_idx].mean(axis=0)


def estimate_initial(mag: np.ndarray, n_samples: int, sr: int, cfg: StftConfig,
                     noise_seconds: float = 0.5) -> np.ndarray:
    """Mode `initial`: mean magnitude over frames within the first T seconds.

    Frames are the complete frames starting at sample 0, H, 2H, … whose end is
    ≤ T·sr — exactly the frames the Simulink InitFcn uses. If T is shorter than
    one frame, the single frame starting at sample 0 is used.
    """
    t1 = int(round(noise_seconds * sr))
    idx = _frames_inside(mag.shape[0], n_samples, cfg, 0, max(t1, cfg.n_fft))
    return mean_magnitude(mag, idx)


def estimate_region(mag: np.ndarray, n_samples: int, sr: int, cfg: StftConfig,
                    t_start: float, t_end: float) -> np.ndarray:
    """Mode `region`, Eq. (5): Eq. (4) over frames inside a user-selected [t_start, t_end] (s)."""
    if t_end <= t_start:
        raise ValueError("region end must be after region start")
    a, b = int(round(t_start * sr)), int(round(t_end * sr))
    idx = _frames_inside(mag.shape[0], n_samples, cfg, a, b)
    if idx.size == 0:
        raise ValueError(f"region is shorter than one frame ({cfg.n_fft / sr:.3f} s)")
    return mean_magnitude(mag, idx)


def estimate_profile(noise_clip: np.ndarray, cfg: StftConfig) -> np.ndarray:
    """Mode `profile`: Eq. (4) over all complete frames of a noise-only clip."""
    noise_clip = np.asarray(noise_clip, dtype=np.float64)
    if len(noise_clip) < cfg.n_fft:
        raise ValueError("noise profile clip is shorter than one frame")
    mag = np.abs(stft(noise_clip, cfg))
    idx = _frames_inside(mag.shape[0], len(noise_clip), cfg, 0, len(noise_clip))
    return mean_magnitude(mag, idx)


def estimate_minstats(mag: np.ndarray, percentile: float = 10.0) -> np.ndarray:
    """Mode `minstats` (simplified minimum statistics), Eq. (6).

    |N̂_k| = sqrt( P_q{ |Y_k(m)|² over all m } ), q = low percentile.
    Needs no silent segment, but biases low (it picks the quietest frames).
    """
    return np.sqrt(np.percentile(mag**2, percentile, axis=0))
