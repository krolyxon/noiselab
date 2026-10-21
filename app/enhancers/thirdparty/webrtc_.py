"""WebRTC noise suppression: the noise suppressor of the WebRTC audio-processing module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...schema import Spec
from ..base import AlgoSpec, EnhanceContext, EnhanceResult
from ..resample import run_at_rate
from ._common import INT16, shift, to_int16

if TYPE_CHECKING:
    from ...params import Params

SR = 16000
CHUNK = SR // 100          # the module processes 10 ms chunks
NS_DELAY = 96              # samples added by the suppressor at levels ≥ 1 (measured; tests/test_thirdparty.py)
_hpf_phase: np.ndarray | None = None


class WebRtcNs:
    spec = AlgoSpec(
        id="webrtc_ns", name="WebRTC NS", family="library",
        description="The noise suppressor in WebRTC (Chrome, Google Meet and most browser calls). "
                    "Classical statistical DSP: noise tracking plus a Wiener-type gain with "
                    "speech/noise likelihoods, on 10 ms frames. Runs at 16 kHz. The module's "
                    "high-pass filter is also active. Automatic gain control is off.",
        reference="WebRTC audio processing module (webrtc.org), via the webrtc-noise-gain package",
        schema={
            "level": Spec(int, 2, choices=(1, 2, 3, 4),
                          help="Suppression level: 1 = mild … 4 = very high."),
            "phase_compensation": Spec(bool, True,
                                       help="Undo the phase (not the magnitude) of WebRTC's fixed "
                                            "high-pass filter, and its 6 ms delay. The phase shift is "
                                            "inaudible but costs waveform metrics such as SI-SDR "
                                            "several dB."),
        },
        native_sr=SR, latency_ms=10.0, requires=("webrtc_noise_gain",),
    )

    def process(self, ctx: EnhanceContext, p: Params) -> EnhanceResult:
        level = int(p.algo_params["level"])
        comp = bool(p.algo_params["phase_compensation"])

        def fn(x: np.ndarray) -> np.ndarray:
            out = _run(x, level)
            return _compensate(out) if comp else out

        return EnhanceResult(cleaned=run_at_rate(ctx.y, ctx.sr, SR, fn))


def _run(x: np.ndarray, level: int) -> np.ndarray:
    from webrtc_noise_gain import AudioProcessor

    ap = AudioProcessor(0, level)
    n = len(x)
    pcm = to_int16(np.concatenate([x, np.zeros(-n % CHUNK)]))
    out = np.empty(len(pcm), dtype=np.int16)
    for i in range(0, len(pcm), CHUNK):
        res = ap.Process10ms(pcm[i: i + CHUNK].tobytes())
        out[i: i + CHUNK] = np.frombuffer(res.audio, dtype=np.int16)
    return out[:n].astype(np.float64) / INT16


def _hpf_phase_response(n_fft: int = 4096) -> np.ndarray:
    """Phase of WebRTC's built-in high-pass filter, identified once.

    Level 0 disables noise suppression, so the module is then just its fixed LTI
    high-pass filter. Feeding it white noise and taking the cross-spectrum gives
    H(f) = S_xo(f) / S_xx(f) (Welch estimate). Returned on rfftfreq(n_fft, 1/SR).
    """
    global _hpf_phase
    if _hpf_phase is None:
        from scipy.signal import csd, welch
        x = np.random.default_rng(0).standard_normal(SR * 8) * 0.05
        o = _run(x, 0)
        _, sxo = csd(x, o, fs=SR, nperseg=n_fft)
        _, sxx = welch(x, fs=SR, nperseg=n_fft)
        _hpf_phase = np.unwrap(np.angle(sxo / sxx))
    return _hpf_phase


def _compensate(out: np.ndarray) -> np.ndarray:
    """Remove the suppressor's delay, then the high-pass filter's phase (zero-phase correction)."""
    out = shift(out, NS_DELAY)
    phase = _hpf_phase_response()
    n = len(out)
    nfft = 1 << int(np.ceil(np.log2(n + 4096)))
    f_grid = np.linspace(0.0, SR / 2, len(phase))
    f_full = np.fft.rfftfreq(nfft, 1 / SR)
    spec = np.fft.rfft(out, nfft) * np.exp(-1j * np.interp(f_full, f_grid, phase))
    return np.fft.irfft(spec, nfft)[:n]
