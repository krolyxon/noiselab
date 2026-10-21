"""noisereduce: spectral gating (T. Sainburg), a popular Python library baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...schema import Spec
from ..base import AlgoSpec, EnhanceContext, EnhanceResult

if TYPE_CHECKING:
    from ...params import Params


class NoiseReduce:
    spec = AlgoSpec(
        id="noisereduce", name="noisereduce (spectral gating)", family="library",
        description="Spectral gating. A bin is kept if it rises above a per-frequency noise threshold "
                    "and gated otherwise, with a smoothed mask. In stationary mode the threshold comes "
                    "from the noise-only lead-in of the selected noise mode (mean + n·std of its dB "
                    "spectrum). In non-stationary mode it comes from a running (time_constant_s) "
                    "smoothed spectrogram.",
        reference="T. Sainburg et al., PLoS Comput. Biol. 16(10), 2020; github.com/timsainburg/noisereduce",
        schema={
            "stationary": Spec(bool, True, help="Stationary (fixed threshold from the noise clip) or "
                                                "non-stationary (running threshold)."),
            "prop_decrease": Spec(float, 1.0, lo=0.0, hi=1.0,
                                  help="Fraction of the gated signal removed (1 = full)."),
            "n_std_thresh": Spec(float, 1.5, lo=0.5, hi=4.0,
                                 help="Threshold above the noise mean, in standard deviations "
                                      "(stationary mode)."),
            "time_mask_smooth_ms": Spec(float, 50.0, lo=0.0, hi=200.0,
                                        help="Smoothing of the mask over time (ms)."),
            "freq_mask_smooth_hz": Spec(float, 500.0, lo=0.0, hi=2000.0,
                                        help="Smoothing of the mask over frequency (Hz)."),
        },
        uses_noise_estimate=True, requires=("noisereduce",),
    )

    def process(self, ctx: EnhanceContext, p: Params) -> EnhanceResult:
        import noisereduce as nr

        a = p.algo_params
        kw = {}
        if a["stationary"]:
            kw["y_noise"] = _noise_clip(ctx, p)
        out = nr.reduce_noise(
            y=ctx.y.astype(np.float32), sr=ctx.sr, stationary=a["stationary"],
            prop_decrease=a["prop_decrease"], n_std_thresh_stationary=a["n_std_thresh"],
            time_mask_smooth_ms=max(a["time_mask_smooth_ms"], 1e-3),
            freq_mask_smooth_hz=max(a["freq_mask_smooth_hz"], 1e-3),
            n_jobs=1, use_tqdm=False, **kw,
        )
        return EnhanceResult(cleaned=np.asarray(out, dtype=np.float64)[: len(ctx.y)])


def _noise_clip(ctx: EnhanceContext, p: Params) -> np.ndarray | None:
    """Noise-only samples for the stationary threshold, matching the selected noise mode.

    initial / region use those samples. Otherwise None, and noisereduce estimates from
    the whole clip.
    """
    if p.noise_mode == "initial":
        return ctx.y[: max(int(p.noise_seconds * ctx.sr), 2048)].astype(np.float32)
    if p.noise_mode == "region":
        a, b = int(p.region_start * ctx.sr), int(p.region_end * ctx.sr)
        return ctx.y[a:b].astype(np.float32)
    return None
