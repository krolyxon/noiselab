"""Resampling to and from a model's native rate (polyphase, scipy.signal.resample_poly)."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return np.asarray(x, dtype=np.float64)
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(np.asarray(x, dtype=np.float64), sr_to // g, sr_from // g)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to exactly n samples."""
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x))])


def run_at_rate(y: np.ndarray, sr: int, native_sr: int, fn) -> np.ndarray:
    """Resample y to native_sr, apply fn, and resample back to the original length."""
    out = fn(resample(y, sr, native_sr))
    return fit_length(resample(np.asarray(out, dtype=np.float64), native_sr, sr), len(y))
