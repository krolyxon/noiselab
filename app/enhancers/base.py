"""Common interface of every algorithm in the comparison bench (CLAUDE.md §14).

An *enhancer* takes the noisy signal (plus its cached STFT and noise estimate) and
returns the cleaned signal ŝ. Hand-written methods also return their gain mask
G(k, m). For black-box models the caller derives an *effective* mask |STFT(ŝ)| / |Y|.

This module never imports heavy libraries. Third-party wrappers import theirs
lazily inside process(), so listing the registry is always cheap.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ..dsp.stft import StftConfig
from ..schema import Spec
from . import model_files

if TYPE_CHECKING:                                # avoid an import cycle with app.params
    from ..params import Params

FAMILIES = {
    "classical": "Classical (Simulink algorithm)",
    "statistical": "Statistical estimators (hand-written)",
    "library": "DSP libraries",
    "deep": "Deep learning",
}


@dataclass(frozen=True)
class AlgoSpec:
    id: str
    name: str
    family: str                                  # key of FAMILIES
    description: str
    reference: str                               # citation shown in the UI / docs
    schema: dict[str, Spec] = field(default_factory=dict)   # algorithm-specific parameters
    native_sr: int | None = None                 # None = runs at any rate
    uses_noise_estimate: bool = False            # True if it uses |N̂| from the noise mode
    latency_ms: float | None = None              # algorithmic latency (frame + look-ahead)
    requires: tuple[str, ...] = ()               # importable modules needed
    model_file: str | None = None                # key in model_files.FILES
    hand_written: bool = False                   # True for code in app/dsp (viva-explainable)

    def availability(self) -> tuple[bool, str]:
        """(available, reason). Cheap: checks installs and files, imports nothing."""
        for mod in self.requires:
            try:
                found = importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                return False, f"Python package '{mod}' is not installed (see requirements-ml.txt)"
        if self.model_file and model_files.locate(self.model_file) is None:
            return False, f"model file missing: run scripts/fetch_models.py ({model_files.FILES[self.model_file].name})"
        return True, ""

    def defaults(self) -> dict[str, Any]:
        return {k: s.default for k, s in self.schema.items()}


@dataclass
class EnhanceContext:
    y: np.ndarray                 # noisy signal, float64, mono
    sr: int
    cfg: StftConfig               # analysis STFT used for plots / gains
    spec: np.ndarray              # Y(m, k) for cfg (cached by the job)
    mag: np.ndarray               # |Y|
    noise_mag: np.ndarray         # |N̂_k| from the selected noise mode (Eq. 4–6)


@dataclass
class EnhanceResult:
    cleaned: np.ndarray                          # ŝ[n], same length as y, before output stage
    gain: np.ndarray | None = None               # explicit G(k, m) on ctx.cfg's grid, if any
    noise_power: np.ndarray | None = None        # time-varying λ_d(k, m), if the method tracks noise
    extra: dict[str, Any] = field(default_factory=dict)


class Enhancer(Protocol):
    spec: AlgoSpec

    def process(self, ctx: EnhanceContext, p: Params) -> EnhanceResult: ...


def attenuation_limit(db: float) -> float:
    """Eq. (34): lim = 10^(−L/20). L ≥ 100 dB means 'off' (lim = 0)."""
    return 0.0 if db >= 100.0 else float(10.0 ** (-db / 20.0))


def apply_attenuation_limit(y: np.ndarray, cleaned: np.ndarray, gain: np.ndarray | None,
                            limit_db: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Eq. (34): ŝ' = lim·y + (1 − lim)·ŝ, the same for every algorithm.

    No bin can be attenuated by more than L dB relative to the input. For a gain-based
    method this is exactly G' = lim + (1 − lim)·G, because the STFT is linear. It is
    the same rule DeepFilterNet uses for its `atten_lim_db`, applied to every method,
    so the comparison is fair.
    """
    lim = attenuation_limit(limit_db)
    if lim == 0.0:
        return cleaned, gain
    out = lim * y + (1.0 - lim) * cleaned
    return out, (None if gain is None else lim + (1.0 - lim) * gain)
