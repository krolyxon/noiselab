"""Parameter schema, validation and presets (CLAUDE.md §4 parameter table).

Pure Python — no Flask. Invalid input raises ParamError (the API turns it into 400).
Out-of-range values are *rejected*, not clamped, so the user always knows what ran.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from .dsp.noise_estimation import NOISE_MODES
from .dsp.spectral_subtraction import METHODS, SubtractionParams
from .dsp.stft import WINDOWS, StftConfig


class ParamError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    kind: type                       # int, float or str
    default: Any
    choices: tuple | None = None     # discrete set, or
    lo: float | None = None          # continuous range [lo, hi]
    hi: float | None = None
    extended: bool = False           # "extended" (beyond the basic Simulink algorithm)
    help: str = ""


SCHEMA: dict[str, Spec] = {
    "n_fft": Spec(int, 512, choices=(256, 512, 1024, 2048),
                  help="Frame length N. Longer = finer frequency resolution, coarser time resolution."),
    "overlap": Spec(float, 0.5, choices=(0.5, 0.75), help="Fraction of each frame shared with the next."),
    "window": Spec(str, "hann", choices=WINDOWS, help="Analysis window (periodic)."),
    "noise_mode": Spec(str, "initial", choices=NOISE_MODES, help="How the noise spectrum |N̂| is estimated."),
    "noise_seconds": Spec(float, 0.5, lo=0.1, hi=3.0, help="Length of the noise-only lead-in (initial mode)."),
    "region_start": Spec(float, 0.0, lo=0.0, hi=120.0, help="Noise region start (s), region mode."),
    "region_end": Spec(float, 0.5, lo=0.0, hi=120.0, help="Noise region end (s), region mode."),
    "profile_id": Spec(str, "", help="Noise profile id from /api/noise-profile, profile mode."),
    "minstats_percentile": Spec(float, 10.0, lo=1.0, hi=50.0, extended=True,
                                help="Low percentile of per-bin power used as the noise floor."),
    "alpha": Spec(float, 2.0, lo=1.0, hi=6.0, help="Over-subtraction factor α."),
    "beta": Spec(float, 0.02, lo=0.0, hi=0.2, help="Spectral floor β."),
    "exponent": Spec(int, 2, choices=(1, 2), help="1 = magnitude subtraction, 2 = power subtraction."),
    "method": Spec(str, "spectral_subtraction", choices=METHODS, help="Gain rule."),
    "gain_smooth": Spec(float, 0.0, lo=0.0, hi=0.95, extended=True, help="Time smoothing λ of the gain."),
    "freq_smooth_bins": Spec(int, 0, lo=0, hi=9, extended=True, help="Frequency smoothing width (bins)."),
    "output_gain_db": Spec(float, 0.0, lo=-12.0, hi=12.0, help="Output gain (dB) before peak protection."),
}


@dataclass(frozen=True)
class Params:
    n_fft: int
    overlap: float
    window: str
    noise_mode: str
    noise_seconds: float
    region_start: float
    region_end: float
    profile_id: str
    minstats_percentile: float
    alpha: float
    beta: float
    exponent: int
    method: str
    gain_smooth: float
    freq_smooth_bins: int
    output_gain_db: float

    @property
    def stft(self) -> StftConfig:
        return StftConfig(self.n_fft, self.overlap, self.window)

    @property
    def subtraction(self) -> SubtractionParams:
        return SubtractionParams(self.alpha, self.beta, self.exponent, self.method,
                                 self.gain_smooth, self.freq_smooth_bins)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULTS = Params(**{k: s.default for k, s in SCHEMA.items()})


def _coerce(name: str, spec: Spec, value: Any) -> Any:
    if spec.kind is str:
        if not isinstance(value, str):
            raise ParamError(f"{name} must be a string")
        if spec.choices is not None and value not in spec.choices:
            raise ParamError(f"{name} must be one of {list(spec.choices)}, got {value!r}")
        if len(value) > 64:
            raise ParamError(f"{name} is too long")
        return value

    # Numbers: accept JSON numbers only (bool is an int subclass in Python — reject it).
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParamError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ParamError(f"{name} must be finite")
    if spec.kind is int:
        if float(value) != int(value):
            raise ParamError(f"{name} must be an integer")
        value = int(value)
    else:
        value = float(value)

    if spec.choices is not None and value not in spec.choices:
        raise ParamError(f"{name} must be one of {list(spec.choices)}, got {value}")
    if spec.lo is not None and not (spec.lo <= value <= spec.hi):
        raise ParamError(f"{name} must be in [{spec.lo}, {spec.hi}], got {value}")
    return value


def validate(raw: dict[str, Any] | None, base: Params = DEFAULTS) -> Params:
    """Validate a (partial) parameter dict on top of `base`. Unknown keys are an error."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ParamError("params must be a JSON object")
    unknown = set(raw) - set(SCHEMA)
    if unknown:
        raise ParamError(f"unknown parameter(s): {', '.join(sorted(unknown))}")

    values = {k: _coerce(k, SCHEMA[k], v) for k, v in raw.items()}
    p = replace(base, **values)

    if p.noise_mode == "region" and not p.region_end > p.region_start:
        raise ParamError("region_end must be greater than region_start")
    if p.noise_mode == "profile" and not p.profile_id:
        raise ParamError("noise_mode 'profile' needs a profile_id (record a noise profile first)")
    return p


def schema_json() -> dict[str, dict[str, Any]]:
    out = {}
    for k, s in SCHEMA.items():
        d: dict[str, Any] = {"type": s.kind.__name__, "default": s.default, "extended": s.extended,
                             "help": s.help}
        if s.choices is not None:
            d["choices"] = list(s.choices)
        if s.lo is not None:
            d["min"], d["max"] = s.lo, s.hi
        out[k] = d
    return out


# --- Presets -------------------------------------------------------------------------------
# Each preset lists only what differs from DEFAULTS; validate() fills in the rest, so every
# preset fully determines the algorithm.

_PRESETS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("match_simulink_basic", "Match Simulink basic",
     "Exactly the settings of the basic Simulink model: N=512, 50 % Hann, first 0.5 s noise, "
     "magnitude subtraction (p=1), α=2, β=0.02, no smoothing.",
     {"n_fft": 512, "overlap": 0.5, "window": "hann", "noise_mode": "initial", "noise_seconds": 0.5,
      "alpha": 2.0, "beta": 0.02, "exponent": 1, "method": "spectral_subtraction",
      "gain_smooth": 0.0, "freq_smooth_bins": 0, "output_gain_db": 0.0}),
    ("faithful_light", "Faithful (light)",
     "Gentle power subtraction; keeps speech natural, leaves some noise.",
     {"alpha": 1.5, "beta": 0.05, "exponent": 2}),
    ("balanced", "Balanced",
     "Moderate over-subtraction with light gain smoothing (extended) to tame musical noise.",
     {"overlap": 0.75, "alpha": 2.5, "beta": 0.02, "gain_smooth": 0.5, "freq_smooth_bins": 3}),
    ("aggressive", "Aggressive",
     "Strong noise removal; expect more speech distortion.",
     {"overlap": 0.75, "alpha": 4.0, "beta": 0.005, "gain_smooth": 0.6, "freq_smooth_bins": 3}),
    ("mains_hum_50", "Mains hum 50 Hz",
     "Long frames resolve the 50 Hz harmonic lines (bin spacing f_s/2048).",
     {"n_fft": 2048, "overlap": 0.75, "alpha": 3.0, "beta": 0.01}),
    # Tuned by a grid search on the synthetic fan noise (5 dB, seeds 0–4): +7.1 dB SI-SDR vs +3.9 dB
    # for the previous heavier-smoothing setting. Frequency smoothing hurt on this noise.
    ("fan_ac", "Fan / AC noise",
     "Stationary low-frequency rumble: longer frames for frequency resolution, strong over-subtraction, "
     "light time smoothing.",
     {"n_fft": 1024, "overlap": 0.75, "alpha": 4.0, "beta": 0.02, "gain_smooth": 0.3,
      "freq_smooth_bins": 0}),
]

PRESETS: dict[str, dict[str, Any]] = {
    pid: {"id": pid, "name": name, "description": desc, "params": validate(vals).to_dict()}
    for pid, name, desc, vals in _PRESETS
}
