"""The ordered list of algorithms, and their availability (CLAUDE.md §14)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..schema import spec_json
from .base import FAMILIES, AlgoSpec, Enhancer
from .classical import CLASSICAL


def _thirdparty() -> tuple[Enhancer, ...]:
    try:
        from .thirdparty import ENHANCERS
    except ImportError:            # package not added yet / broken: classical still works
        return ()
    return ENHANCERS


_ALL: dict[str, Enhancer] = {e.spec.id: e for e in (*CLASSICAL, *_thirdparty())}
ALGORITHM_IDS: tuple[str, ...] = tuple(_ALL)
DEFAULT_ALGORITHM = "spectral_subtraction"


def get(algo_id: str) -> Enhancer:
    return _ALL[algo_id]


def spec(algo_id: str) -> AlgoSpec:
    return _ALL[algo_id].spec


@lru_cache(maxsize=None)
def availability(algo_id: str) -> tuple[bool, str]:
    """Cached for the process lifetime (installs don't change while the server runs)."""
    return _ALL[algo_id].spec.availability()


def refresh() -> None:
    availability.cache_clear()


def available_ids() -> list[str]:
    return [a for a in ALGORITHM_IDS if availability(a)[0]]


def algorithms_json() -> list[dict[str, Any]]:
    out = []
    for a in ALGORITHM_IDS:
        s = spec(a)
        ok, reason = availability(a)
        out.append({
            "id": s.id, "name": s.name, "family": s.family, "family_name": FAMILIES[s.family],
            "description": s.description, "reference": s.reference,
            "native_sr": s.native_sr, "latency_ms": s.latency_ms,
            "uses_noise_estimate": s.uses_noise_estimate, "hand_written": s.hand_written,
            "available": ok, "reason": reason,
            "schema": {k: spec_json(v) for k, v in s.schema.items()},
        })
    return out
