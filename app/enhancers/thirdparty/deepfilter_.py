"""DeepFilterNet3 (H. Schröter et al.): ERB-gain + deep-filtering network at 48 kHz."""

from __future__ import annotations

import sys
import threading
import types
from typing import TYPE_CHECKING, Any

import numpy as np

from ...schema import Spec
from ..base import AlgoSpec, EnhanceContext, EnhanceResult
from ..resample import fit_length, run_at_rate

if TYPE_CHECKING:
    from ...params import Params

SR = 48000
_lock = threading.Lock()
_models: dict[bool, Any] = {}


class DeepFilterNet3:
    spec = AlgoSpec(
        id="deepfilternet3", name="DeepFilterNet3", family="deep",
        description="Two-stage network. Stage 1 predicts gains on 32 ERB bands (the spectral "
                    "envelope). Stage 2 predicts complex 'deep filter' coefficients that also restore "
                    "the periodic (harmonic) structure below 5 kHz, which a real gain cannot do. Full "
                    "band, 48 kHz, 20 ms window, 10 ms hop. State of the art among small real-time "
                    "models.",
        reference="H. Schröter et al., \"DeepFilterNet: perceptually motivated real-time speech "
                  "enhancement,\" Interspeech 2023; github.com/Rikorose/DeepFilterNet",
        schema={
            "post_filter": Spec(bool, False, help="DeepFilterNet's post-filter: extra attenuation of "
                                                  "very noisy bins."),
        },
        native_sr=SR, latency_ms=40.0, requires=("df", "torch", "torchaudio"),
    )

    def warm(self, p: Params) -> None:
        """Load the model outside the timed region (see processing.enhance)."""
        load_model(bool(p.algo_params["post_filter"]))

    def process(self, ctx: EnhanceContext, p: Params) -> EnhanceResult:
        pf = bool(p.algo_params["post_filter"])
        return EnhanceResult(cleaned=run_at_rate(ctx.y, ctx.sr, SR, lambda x: _run(x, pf)))


def _install_torchaudio_shim() -> None:
    """deepfilternet 0.5.6 imports torchaudio.backend.common.AudioMetaData (a type annotation only).
    torchaudio ≥ 2.9 removed that module, so register an empty stand-in before importing df."""
    if "torchaudio.backend.common" in sys.modules:
        return
    try:
        import torchaudio.backend.common  # noqa: F401
        return
    except ImportError:
        pass
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # noqa: D401 - placeholder type
        pass

    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


def load_model(post_filter: bool = False):
    with _lock:
        if post_filter not in _models:
            _install_torchaudio_shim()
            from df.enhance import init_df
            model, state, _ = init_df(post_filter=post_filter, log_level="ERROR", log_file=None)
            model.eval()
            _models[post_filter] = (model, state)
        return _models[post_filter]


def _run(x: np.ndarray, post_filter: bool) -> np.ndarray:
    import torch

    model, state = load_model(post_filter)       # installs the torchaudio shim before importing df
    from df.enhance import enhance
    with _lock, torch.no_grad():               # the Rust DF state is not thread-safe
        out = enhance(model, state, torch.from_numpy(x.astype(np.float32))[None], atten_lim_db=None)
    return fit_length(out[0].numpy().astype(np.float64), len(x))
