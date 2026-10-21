"""In-memory job store: LRU + TTL, with a per-job temp directory for WAV files.

A job keeps the noisy signal and caches its STFT per (n_fft, overlap, window),
so /api/reprocess only redoes the cheap part (gain, ISTFT, metrics, viz).
"""

from __future__ import annotations

import re
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..dsp.stft import StftConfig, stft

_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def new_id() -> str:
    return secrets.token_hex(16)


def valid_id(job_id: str) -> bool:
    return bool(_ID_RE.match(job_id))


@dataclass
class Job:
    id: str
    y: np.ndarray
    sr: int
    dir: Path
    name: str = "audio"
    clean: np.ndarray | None = None              # ground truth (demo mode only)
    demo: dict[str, Any] | None = None
    created: float = field(default_factory=time.time)
    accessed: float = field(default_factory=time.time)
    version: int = 0
    _stft: dict[tuple, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)
    last: dict[str, Any] = field(default_factory=dict, repr=False)   # latest params/result
    # Enhancer outputs keyed by the parameters that produce them, so slow models (DNNs)
    # are not re-run when only the attenuation limit / output gain changes (§14).
    results: OrderedDict[str, Any] = field(default_factory=OrderedDict, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def duration(self) -> float:
        return len(self.y) / self.sr

    def spectrum(self, cfg: StftConfig) -> tuple[np.ndarray, np.ndarray]:
        """(Y, |Y|) for this config, computed once and cached."""
        if cfg.key not in self._stft:
            if len(self._stft) >= 3:                      # bound memory: keep 3 configs
                self._stft.pop(next(iter(self._stft)))
            spec = stft(self.y, cfg)
            self._stft[cfg.key] = (spec, np.abs(spec))
        return self._stft[cfg.key]


    def remember_result(self, key: str, value: Any, keep: int = 8) -> None:
        self.results[key] = value
        self.results.move_to_end(key)
        while len(self.results) > keep:
            self.results.popitem(last=False)


@dataclass
class NoiseProfile:
    id: str
    y: np.ndarray
    sr: int
    created: float = field(default_factory=time.time)


class _TTLStore:
    def __init__(self, max_items: int, ttl: float):
        self.max_items, self.ttl = max_items, ttl
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def _drop(self, item: Any) -> None:
        pass

    def purge(self) -> None:
        now = time.time()
        with self._lock:
            for k in [k for k, v in self._items.items() if now - self._stamp(v) > self.ttl]:
                self._drop(self._items.pop(k))

    def _stamp(self, item: Any) -> float:
        return getattr(item, "accessed", item.created)

    def put(self, key: str, item: Any) -> None:
        with self._lock:
            self._items[key] = item
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                _, old = self._items.popitem(last=False)
                self._drop(old)

    def get(self, key: str) -> Any | None:
        self.purge()
        with self._lock:
            item = self._items.get(key)
            if item is not None:
                self._items.move_to_end(key)
                if hasattr(item, "accessed"):
                    item.accessed = time.time()
            return item

    def __len__(self) -> int:
        return len(self._items)


class JobStore(_TTLStore):
    def __init__(self, root: Path, max_items: int, ttl: float):
        super().__init__(max_items, ttl)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, y: np.ndarray, sr: int, name: str = "audio", **kw: Any) -> Job:
        jid = new_id()
        d = self.root / jid
        d.mkdir(mode=0o700)
        job = Job(id=jid, y=np.asarray(y, dtype=np.float64), sr=sr, dir=d, name=name, **kw)
        self.put(jid, job)
        return job

    def _drop(self, item: Job) -> None:
        shutil.rmtree(item.dir, ignore_errors=True)


class ProfileStore(_TTLStore):
    def create(self, y: np.ndarray, sr: int) -> NoiseProfile:
        p = NoiseProfile(id=new_id(), y=np.asarray(y, dtype=np.float64), sr=sr)
        self.put(p.id, p)
        return p
