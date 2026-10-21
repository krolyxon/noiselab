"""NSNet2: Microsoft's noise-suppression baseline network (Deep Noise Suppression Challenge)."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

import numpy as np

from ...schema import Spec
from .. import model_files
from ..base import AlgoSpec, EnhanceContext, EnhanceResult
from ..resample import run_at_rate

if TYPE_CHECKING:
    from ...params import Params

SR = 16000
WIN, HOP = 320, 160        # 20 ms window, 10 ms hop (as in the reference enhance_onnx.py)
_lock = threading.Lock()
_session = None


class NsNet2:
    spec = AlgoSpec(
        id="nsnet2", name="NSNet2", family="deep",
        description="A recurrent network (GRUs) predicts a suppression gain for each of 161 STFT bins "
                    "from log-power features (20 ms √Hann window, 10 ms hop, 16 kHz). Microsoft's "
                    "baseline for the DNS Challenge, from the same line of work as Teams' noise "
                    "suppression. Noisy phase, like Eq. (12).",
        reference="S. Braun, I. Tashev, \"Data augmentation and loss normalization for deep noise "
                  "suppression,\" SPECOM 2020; github.com/microsoft/DNS-Challenge",
        schema={
            "min_gain_db": Spec(float, -80.0, lo=-80.0, hi=-6.0,
                                help="Lowest gain the network's mask may apply (reference: −80 dB)."),
        },
        native_sr=SR, latency_ms=20.0, requires=("onnxruntime",), model_file="nsnet2",
    )

    def warm(self, p: Params) -> None:
        _get_session()

    def process(self, ctx: EnhanceContext, p: Params) -> EnhanceResult:
        g_min = 10.0 ** (p.algo_params["min_gain_db"] / 20.0)
        return EnhanceResult(cleaned=run_at_rate(ctx.y, ctx.sr, SR, lambda x: _run(x, g_min)))


def _get_session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            path = model_files.locate("nsnet2")
            if path is None:
                raise OSError("NSNet2 model missing — run scripts/fetch_models.py")
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("NOISELAB_ORT_THREADS", "2"))
            _session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        return _session


def _run(x: np.ndarray, g_min: float) -> np.ndarray:
    """Same framing as the reference featurelib.py: sqrt of a symmetric Hann, N − H zeros of
    lead-in, log10 power features, gain clipped to [g_min, 1], overlap-add."""
    win = np.sqrt(np.hanning(WIN))
    n = len(x)
    n_frames = int(np.ceil((n + WIN - HOP) / HOP))
    padded = np.zeros((n_frames - 1) * HOP + WIN)
    padded[WIN - HOP: WIN - HOP + n] = x
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(padded[idx] * win, axis=1)                      # (frames, 161)
    feat = np.log10(np.maximum(np.abs(spec) ** 2, 1e-12)).astype(np.float32)[None]
    sess = _get_session()
    gain = sess.run(None, {sess.get_inputs()[0].name: feat})[0][0]
    gain = np.clip(gain, g_min, 1.0)
    frames = np.fft.irfft(spec * gain, n=WIN, axis=1) * win
    out = np.zeros(len(padded))
    env = np.zeros(len(padded))
    for m in range(n_frames):
        out[m * HOP: m * HOP + WIN] += frames[m]
        env[m * HOP: m * HOP + WIN] += win**2
    out = np.divide(out, env, out=np.zeros_like(out), where=env > 1e-8)
    return out[WIN - HOP: WIN - HOP + n]
