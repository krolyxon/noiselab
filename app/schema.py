"""Parameter-schema primitives shared by app/params.py and the algorithm registry.

Pure Python, no Flask. Values are *rejected* when out of range, never clamped,
so the user always knows what ran.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class ParamError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    kind: type                       # int, float, str or bool
    default: Any
    choices: tuple | None = None     # discrete set, or
    lo: float | None = None          # continuous range [lo, hi]
    hi: float | None = None
    extended: bool = False           # "extended" (beyond the basic Simulink algorithm)
    help: str = ""


def coerce(name: str, spec: Spec, value: Any) -> Any:
    if spec.kind is bool:
        if not isinstance(value, bool):
            raise ParamError(f"{name} must be true or false")
        return value
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


def spec_json(s: Spec) -> dict[str, Any]:
    d: dict[str, Any] = {"type": s.kind.__name__, "default": s.default, "extended": s.extended,
                         "help": s.help}
    if s.choices is not None:
        d["choices"] = list(s.choices)
    if s.lo is not None:
        d["min"], d["max"] = s.lo, s.hi
    return d
