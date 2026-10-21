"""RNNoise (J.-M. Valin, Xiph): hybrid DSP + small recurrent network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...schema import Spec
from ..base import AlgoSpec, EnhanceContext, EnhanceResult
from ..resample import fit_length, run_at_rate
from ._common import INT16, shift

if TYPE_CHECKING:
    from ...params import Params

SR = 48000
FRAME = 480                # 10 ms
DELAY = 960                # output lags input by exactly 2 frames (measured on real speech; tests/test_thirdparty.py)


class RnNoise:
    spec = AlgoSpec(
        id="rnnoise", name="RNNoise", family="deep",
        description="A GRU network (~60 k weights) predicts gains for 22 Bark-like bands, plus a "
                    "pitch comb filter, on 10 ms frames at 48 kHz. Designed to run on a CPU in real "
                    "time. Used in Jitsi, OBS Studio, Mumble and many VoIP apps. Optional VAD gate: "
                    "frames with a speech probability below the threshold are attenuated further.",
        reference="J.-M. Valin, \"A hybrid DSP/deep learning approach to real-time full-band speech "
                  "enhancement,\" MMSP 2018",
        schema={
            "vad_threshold": Spec(float, 0.0, lo=0.0, hi=0.95,
                                  help="Gate frames whose RNNoise speech probability is below this "
                                       "(0 = gate off)."),
            "gate_db": Spec(float, 30.0, lo=6.0, hi=60.0, help="Extra attenuation of gated frames (dB)."),
        },
        native_sr=SR, latency_ms=10.0, requires=("pyrnnoise",),
    )

    def process(self, ctx: EnhanceContext, p: Params) -> EnhanceResult:
        a = p.algo_params
        probs: list[np.ndarray] = []

        def fn(x: np.ndarray) -> np.ndarray:
            out, pr = _run(x)
            probs.append(pr)
            if a["vad_threshold"] > 0:
                out = out * _gate(pr, len(out), a["vad_threshold"], a["gate_db"])
            return out

        y = run_at_rate(ctx.y, ctx.sr, SR, fn)
        return EnhanceResult(cleaned=y, extra={"speech_prob_frames": probs[0]})


def _run(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import ctypes

    from pyrnnoise.rnnoise import lib     # ctypes handle to librnnoise

    n = len(x)
    pad = np.concatenate([x, np.zeros(-(n + DELAY) % FRAME + DELAY)]) * INT16   # RNNoise expects int16 scale
    buf = np.ascontiguousarray(pad, dtype=np.float32)
    out = np.empty_like(buf)
    probs = np.empty(len(buf) // FRAME)
    state = lib.rnnoise_create(None)
    try:
        for i in range(len(probs)):
            frame = np.ascontiguousarray(buf[i * FRAME:(i + 1) * FRAME])
            o = np.empty(FRAME, dtype=np.float32)
            probs[i] = lib.rnnoise_process_frame(state, o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            out[i * FRAME:(i + 1) * FRAME] = o
    finally:
        lib.rnnoise_destroy(state)
    y = shift(out.astype(np.float64) / INT16, DELAY)
    return fit_length(y, n), probs


def _gate(probs: np.ndarray, n: int, thr: float, gate_db: float) -> np.ndarray:
    """Per-frame gate from the VAD probability, smoothed (≈30 ms) and interpolated per sample."""
    g_low = 10.0 ** (-gate_db / 20.0)
    g = np.where(probs >= thr, 1.0, g_low)
    # The probability of frame i describes output samples ending DELAY later.
    sm = np.empty_like(g)
    acc = 1.0
    for i, v in enumerate(g):
        acc = 0.7 * acc + 0.3 * v
        sm[i] = acc
    centres = np.arange(len(sm)) * FRAME + FRAME / 2 - DELAY
    return np.interp(np.arange(n), centres, sm)
