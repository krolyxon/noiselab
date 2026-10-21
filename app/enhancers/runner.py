"""Run any registered algorithm on a plain array: no Flask, no job store (used by the benchmark).

Same steps as services/processing.enhance(): STFT → noise estimate (Eq. 4–6) → algorithm
→ attenuation limit (Eq. 34). The output stage (Eq. 14) is not applied, because metrics are
computed before it.
"""

from __future__ import annotations

import numpy as np

from ..dsp.noise_estimation import estimate_initial, estimate_minstats, estimate_region
from ..dsp.stft import stft
from ..params import Params
from . import registry
from .base import EnhanceContext, apply_attenuation_limit, warm


def noise_estimate(y: np.ndarray, sr: int, p: Params, mag: np.ndarray) -> np.ndarray:
    if p.noise_mode == "initial":
        return estimate_initial(mag, len(y), sr, p.stft, p.noise_seconds)
    if p.noise_mode == "region":
        return estimate_region(mag, len(y), sr, p.stft, p.region_start, p.region_end)
    if p.noise_mode == "minstats":
        return estimate_minstats(mag, p.minstats_percentile)
    raise ValueError("runner supports noise modes initial, region and minstats")


def prepare(p: Params) -> None:
    """Load the algorithm's model now (so a caller's timer measures inference only)."""
    warm(registry.get(p.algorithm), p)


def enhance_signal(y: np.ndarray, sr: int, p: Params) -> np.ndarray:
    """ŝ for noisy y with parameters p, time-aligned and of the same length."""
    y = np.asarray(y, dtype=np.float64)
    spec = stft(y, p.stft)
    mag = np.abs(spec)
    ctx = EnhanceContext(y=y, sr=sr, cfg=p.stft, spec=spec, mag=mag, noise_mag=noise_estimate(y, sr, p, mag))
    res = registry.get(p.algorithm).process(ctx, p)
    out, _ = apply_attenuation_limit(y, res.cleaned, None, p.atten_limit_db)
    return out
