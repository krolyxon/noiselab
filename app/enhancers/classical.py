"""Hand-written algorithms (app/dsp) wrapped in the enhancer interface.

These are the algorithms the author can derive line by line. The third-party
models in thirdparty/ are comparison baselines only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dsp import statistical as ST
from ..dsp.spectral_subtraction import compute_gain
from ..dsp.stft import istft
from ..schema import Spec
from .base import AlgoSpec, EnhanceContext, EnhanceResult

if TYPE_CHECKING:
    from ..params import Params


class SpectralSubtraction:
    """Eq. (7)–(13): the basic algorithm, identical to the Simulink model's (§4)."""

    spec = AlgoSpec(
        id="spectral_subtraction", name="Spectral subtraction", family="classical",
        description="Boll/Berouti over-subtraction with a spectral floor, or the simple Wiener rule "
                    "(Eq. 7–9). The algorithm of the Simulink model; all §4 parameters apply.",
        reference="S. F. Boll, IEEE Trans. ASSP 27(2), 1979; M. Berouti et al., ICASSP 1979",
        uses_noise_estimate=True, hand_written=True,
    )

    def process(self, ctx: EnhanceContext, p: Params) -> EnhanceResult:
        gain = compute_gain(ctx.mag, ctx.noise_mag, p.subtraction)
        return EnhanceResult(cleaned=istft(gain * ctx.spec, len(ctx.y), ctx.cfg), gain=gain)


_DD_SCHEMA = {
    "dd_alpha": Spec(float, 0.98, lo=0.80, hi=0.998,
                     help="Decision-directed weight α_dd (Eq. 24): higher = smoother ξ, less musical "
                          "noise, slower reaction to speech onsets."),
    "xi_min_db": Spec(float, -25.0, lo=-40.0, hi=-5.0,
                      help="Floor on the a-priori SNR ξ_min (dB). Lower = more noise removed, more "
                           "artefacts."),
    "gain_floor_db": Spec(float, -40.0, lo=-60.0, hi=-5.0,
                          help="Lowest gain G_min (dB) any bin can receive."),
}


class _DecisionDirected:
    method = ""

    def process(self, ctx: EnhanceContext, p: Params) -> EnhanceResult:
        a = p.algo_params
        prm = ST.DDParams(a["dd_alpha"], a["xi_min_db"], a["gain_floor_db"])
        gain = ST.decision_directed_gain(ctx.mag**2, ST.noise_power_from_mag(ctx.noise_mag), self.method, prm)
        return EnhanceResult(cleaned=istft(gain * ctx.spec, len(ctx.y), ctx.cfg), gain=gain)


class WienerDD(_DecisionDirected):
    method = "wiener_dd"
    spec = AlgoSpec(
        id="wiener_dd", name="Wiener (decision-directed)", family="statistical",
        description="Wiener gain ξ/(1+ξ) with the a-priori SNR ξ from the decision-directed rule "
                    "(Eq. 24–25).",
        reference="P. Scalart & J. Vieira Filho, ICASSP 1996",
        schema=_DD_SCHEMA, uses_noise_estimate=True, hand_written=True,
    )


class MmseStsa(_DecisionDirected):
    method = "mmse_stsa"
    spec = AlgoSpec(
        id="mmse_stsa", name="MMSE-STSA", family="statistical",
        description="Ephraim–Malah minimum mean-square error estimate of the spectral amplitude "
                    "(Eq. 26).",
        reference="Y. Ephraim & D. Malah, IEEE Trans. ASSP 32(6), 1984",
        schema=_DD_SCHEMA, uses_noise_estimate=True, hand_written=True,
    )


class LogMmse(_DecisionDirected):
    method = "log_mmse"
    spec = AlgoSpec(
        id="log_mmse", name="Log-MMSE (LSA)", family="statistical",
        description="Ephraim–Malah estimator minimising the error of the log-amplitude, closer to "
                    "loudness perception (Eq. 27).",
        reference="Y. Ephraim & D. Malah, IEEE Trans. ASSP 33(2), 1985",
        schema=_DD_SCHEMA, uses_noise_estimate=True, hand_written=True,
    )


class Omlsa:
    spec = AlgoSpec(
        id="omlsa", name="OM-LSA + IMCRA", family="statistical",
        description="Log-MMSE gain weighted by the speech-presence probability, with IMCRA noise "
                    "tracking, so the noise estimate adapts while speech is present (Eq. 28–33). "
                    "The noise mode only initialises the tracker.",
        reference="I. Cohen & B. Berdugo, Signal Processing 81(11), 2001; I. Cohen, IEEE Trans. "
                  "SAP 11(5), 2003",
        schema={
            "dd_alpha": Spec(float, 0.92, lo=0.80, hi=0.99, help="Decision-directed weight α (Eq. 24)."),
            "g_min_db": Spec(float, -25.0, lo=-40.0, hi=-5.0,
                             help="Gain G_min (dB) applied where speech is absent (Eq. 28)."),
            "tracking_window_s": Spec(float, 1.0, lo=0.4, hi=3.0,
                                      help="IMCRA minimum-search window (s). Shorter = follows "
                                           "changing noise faster, but may eat long vowels."),
        },
        uses_noise_estimate=True, hand_written=True,
    )

    def process(self, ctx: EnhanceContext, p: Params) -> EnhanceResult:
        a = p.algo_params
        prm = ST.OmlsaParams(dd_alpha=a["dd_alpha"], g_min_db=a["g_min_db"],
                             tracking_window_s=a["tracking_window_s"])
        r = ST.omlsa_gain(ctx.mag**2, ST.noise_power_from_mag(ctx.noise_mag), ctx.cfg.hop / ctx.sr, prm)
        return EnhanceResult(cleaned=istft(r.gain * ctx.spec, len(ctx.y), ctx.cfg), gain=r.gain,
                             noise_power=r.noise_power, extra={"speech_prob": r.speech_prob})


CLASSICAL = (SpectralSubtraction(), WienerDD(), MmseStsa(), LogMmse(), Omlsa())
