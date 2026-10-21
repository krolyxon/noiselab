"""Perceptual quality metrics from third-party packages (CLAUDE.md §14), all optional.

- PESQ (ITU-T P.862.2 wide-band, `pesq`): intrusive, needs the clean reference. Range −0.5…4.5.
- STOI / ESTOI (`pystoi`): intrusive intelligibility index, 0…1.
- DNSMOS P.835 (Microsoft, ONNX): **non-intrusive**, so it works on any recording. A network
  trained to predict listener ratings: SIG (speech quality), BAK (background noise),
  OVRL (overall), each on a 1–5 scale. It is a *prediction* of a MOS, not a measurement.

Every function returns None when its package or model is missing (the UI shows "n/a").
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Any

import numpy as np

from . import model_files
from .resample import resample

DNSMOS_SR = 16000
DNSMOS_LEN = int(9.01 * DNSMOS_SR)     # the model's fixed input: 9.01 s
# Polynomial calibration of the raw outputs (non-personalised model), from Microsoft's
# reference DNSMOS/dnsmos_local.py: SIG, BAK, OVRL.
_DNSMOS_POLY = (np.poly1d([-0.08397278, 1.22083953, 0.0052439]),
                np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
                np.poly1d([-0.06766283, 1.11546468, 0.04602535]))
_lock = threading.Lock()
_dnsmos = None


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def available() -> dict[str, bool]:
    return {"pesq": _has("pesq"), "stoi": _has("pystoi"),
            "dnsmos": _has("onnxruntime") and model_files.locate("dnsmos_p835") is not None}


def pesq_wb(clean: np.ndarray, estimate: np.ndarray, sr: int) -> float | None:
    """Wide-band PESQ (P.862.2). Both signals are resampled to 16 kHz, as the standard requires."""
    if not _has("pesq"):
        return None
    from pesq import pesq
    from pesq.cypesq import PesqError
    try:
        return float(pesq(16000, resample(clean, sr, 16000), resample(estimate, sr, 16000), "wb"))
    except PesqError:            # e.g. no utterance detected in a near-silent clip
        return None


def stoi(clean: np.ndarray, estimate: np.ndarray, sr: int, extended: bool = False) -> float | None:
    if not _has("pystoi"):
        return None
    from pystoi import stoi as _stoi
    return float(_stoi(clean, estimate, sr, extended=extended))


def _dnsmos_session():
    global _dnsmos
    with _lock:
        if _dnsmos is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            _dnsmos = ort.InferenceSession(str(model_files.locate("dnsmos_p835")), so,
                                           providers=["CPUExecutionProvider"])
        return _dnsmos


def dnsmos(x: np.ndarray, sr: int, hop_s: float = 9.01) -> dict[str, float] | None:
    """DNSMOS P.835 of a signal: mean SIG/BAK/OVRL over 9.01 s windows.

    Microsoft's script slides the window by 1 s. The default here is non-overlapping
    windows (the last one aligned to the end), which is ~9× faster and differs by a few
    hundredths on stationary material. Clips shorter than 9.01 s are tiled, as in the
    reference.
    """
    if not available()["dnsmos"]:
        return None
    a = resample(x, sr, DNSMOS_SR).astype(np.float32)
    if len(a) == 0 or not np.any(a):
        return None
    while len(a) < DNSMOS_LEN:
        a = np.concatenate([a, a])
    hop = int(hop_s * DNSMOS_SR)
    starts = list(range(0, len(a) - DNSMOS_LEN + 1, hop))
    if starts[-1] != len(a) - DNSMOS_LEN:
        starts.append(len(a) - DNSMOS_LEN)
    sess = _dnsmos_session()
    raw = np.array([sess.run(None, {"input_1": a[s: s + DNSMOS_LEN][None]})[0][0] for s in starts])
    sig, bak, ovr = (float(np.mean(_DNSMOS_POLY[k](raw[:, k]))) for k in range(3))
    return {"sig": sig, "bak": bak, "ovrl": ovr}


def intrusive(clean: np.ndarray, estimate: np.ndarray, sr: int) -> dict[str, Any]:
    """All reference-based perceptual metrics; None for those whose package is missing."""
    return {"pesq": pesq_wb(clean, estimate, sr), "stoi": stoi(clean, estimate, sr),
            "estoi": stoi(clean, estimate, sr, extended=True)}
