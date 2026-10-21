"""Helpers shared by the third-party wrappers."""

from __future__ import annotations

import numpy as np

INT16 = 32767.0


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * INT16), -32768, 32767).astype(np.int16)


def shift(x: np.ndarray, delay: int) -> np.ndarray:
    """Remove a known output delay: out[n] = x[n + delay], zero-filled at the end."""
    if delay <= 0:
        return x
    return np.concatenate([x[delay:], np.zeros(delay)])
