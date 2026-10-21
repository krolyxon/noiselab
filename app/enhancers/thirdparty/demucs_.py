"""Demucs speech enhancement (A. Défossez et al., Meta): waveform-domain U-Net + LSTM."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from ...schema import Spec
from ..base import AlgoSpec, EnhanceContext, EnhanceResult
from ..resample import fit_length, run_at_rate

if TYPE_CHECKING:
    from ...params import Params

SR = 16000
_lock = threading.Lock()
_models: dict[str, Any] = {}


class Demucs:
    spec = AlgoSpec(
        id="demucs", name="Demucs (denoiser)", family="deep",
        description="Works on the waveform directly (no STFT, no noisy phase). A convolutional "
                    "encoder–decoder (U-Net) with an LSTM bottleneck outputs the clean signal. "
                    "Trained on the DNS-Challenge data (dns48/dns64) or DNS + Valentini (master64). "
                    "Causal, 16 kHz. The largest model here (dns64: 33.5 M parameters).",
        reference="A. Défossez, G. Synnaeve, Y. Adi, \"Real time speech enhancement in the waveform "
                  "domain,\" Interspeech 2020; github.com/facebookresearch/denoiser",
        schema={
            "model": Spec(str, "dns64", choices=("dns48", "dns64", "master64"),
                          help="Pretrained model: dns48 (18.9 M params), dns64 (33.5 M), master64 "
                               "(dns64 architecture, more training data)."),
        },
        native_sr=SR, requires=("denoiser", "torch"),
    )

    def process(self, ctx: EnhanceContext, p: Params) -> EnhanceResult:
        name = p.algo_params["model"]
        return EnhanceResult(cleaned=run_at_rate(ctx.y, ctx.sr, SR, lambda x: _run(x, name)))


def load_model(name: str = "dns64"):
    with _lock:
        if name not in _models:
            from denoiser import pretrained
            m = getattr(pretrained, name)()
            m.eval()
            _models[name] = m
        return _models[name]


def _run(x: np.ndarray, name: str) -> np.ndarray:
    import torch

    model = load_model(name)
    with torch.no_grad():
        out = model(torch.from_numpy(x.astype(np.float32))[None, None])
    return fit_length(out[0, 0].numpy().astype(np.float64), len(x))
