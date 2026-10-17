import numpy as np
import pytest

from app.dsp import metrics, synth
from app.dsp.noise_estimation import estimate_initial, estimate_minstats, estimate_profile, estimate_region
from app.dsp.spectral_subtraction import (
    SubtractionParams,
    denoise,
    output_stage,
    subtracted_magnitude,
)
from app.dsp.stft import StftConfig, stft

SR = 16000


@pytest.fixture(scope="module")
def demo():
    return synth.make_demo("white", snr_db=5.0, sr=SR, duration=6.0, seed=0)


def _run(noisy, cfg=StftConfig(), params=SubtractionParams(), **kw):
    spec = stft(noisy, cfg)
    noise_mag = estimate_initial(np.abs(spec), len(noisy), SR, cfg, 0.5)
    return denoise(noisy, noise_mag, cfg, params, spec=spec, **kw)


def test_snr_improvement_white_5db(demo):
    res = _run(demo["noisy"])
    snr_in = metrics.snr_db(demo["clean"], demo["noisy"])
    snr_out = metrics.snr_db(demo["clean"], res.cleaned)
    assert snr_in == pytest.approx(5.0, abs=1e-6)
    assert snr_out - snr_in > 3.0
    assert np.all(np.isfinite(res.cleaned))
    assert len(res.cleaned) == len(demo["noisy"])


@pytest.mark.parametrize("params", [
    SubtractionParams(exponent=1),
    SubtractionParams(method="wiener"),
    SubtractionParams(gain_smooth=0.8, freq_smooth_bins=5),
])
@pytest.mark.parametrize("cfg", [StftConfig(1024, 0.75, "hamming"), StftConfig(256, 0.5, "sqrt_hann")])
def test_variants_finite_and_helpful(demo, params, cfg):
    res = _run(demo["noisy"], cfg, params)
    assert np.all(np.isfinite(res.cleaned)) and len(res.cleaned) == len(demo["noisy"])
    assert metrics.snr_db(demo["clean"], res.cleaned) > metrics.snr_db(demo["clean"], demo["noisy"])


@pytest.mark.parametrize("exponent", [1, 2])
def test_identity_alpha0_beta1(demo, exponent):
    # With α = 0 and β = 1, Eq. (7) becomes max(|Y|^p, |N̂|^p), which equals |Y|
    # wherever |Y| ≥ |N̂|. With a zero noise estimate that holds everywhere,
    # so the chain must return the input exactly (up to float error).
    y = demo["noisy"]
    cfg = StftConfig()
    res = denoise(y, np.zeros(cfg.n_bins), cfg, SubtractionParams(alpha=0.0, beta=1.0, exponent=exponent))
    assert np.max(np.abs(res.cleaned - y)) < 1e-6


def test_floor_is_applied():
    mag = np.array([[0.1, 1.0, 5.0]])
    noise = np.array([1.0, 1.0, 1.0])
    s = subtracted_magnitude(mag, noise, alpha=2.0, beta=0.04, exponent=2)
    # bins 0,1 hit the floor sqrt(0.04·1) = 0.2; bin 2: sqrt(25 − 2) .
    assert np.allclose(s, [[0.2, 0.2, np.sqrt(23.0)]])


def test_output_stage_limits_and_gains():
    x = np.array([0.0, 0.5, -2.0])
    assert np.max(np.abs(output_stage(x))) == pytest.approx(0.99)
    assert output_stage(np.array([0.1]), 6.0)[0] == pytest.approx(0.1 * 10 ** 0.3)


def test_noise_estimators_shapes(demo):
    cfg = StftConfig()
    y = demo["noisy"]
    mag = np.abs(stft(y, cfg))
    for est in (
        estimate_initial(mag, len(y), SR, cfg, 0.5),
        estimate_region(mag, len(y), SR, cfg, 0.1, 0.7),
        estimate_profile(demo["noise"][:SR], cfg),
        estimate_minstats(mag),
    ):
        assert est.shape == (cfg.n_bins,) and np.all(est >= 0) and np.all(np.isfinite(est))


def test_initial_estimate_matches_true_noise(demo):
    # First 0.5 s is noise-only, so the estimate should be close to the true noise spectrum.
    cfg = StftConfig()
    y = demo["noisy"]
    est = estimate_initial(np.abs(stft(y, cfg)), len(y), SR, cfg, 0.5)
    true = estimate_profile(demo["noise"][: SR // 2], cfg)
    assert np.allclose(est, true)


def test_region_too_short_raises(demo):
    cfg = StftConfig()
    mag = np.abs(stft(demo["noisy"], cfg))
    with pytest.raises(ValueError):
        estimate_region(mag, len(demo["noisy"]), SR, cfg, 1.0, 1.01)


@pytest.mark.parametrize("q", [5.0, 10.0, 20.0, 30.0])
def test_minstats_is_unbiased_on_pure_noise(q):
    # Eq. (6): the bias-compensated percentile must match the mean-magnitude estimate (Eq. 4)
    cfg = StftConfig()
    noise = synth.make_noise("white", SR * 10, SR, seed=0) * 0.05
    est = estimate_minstats(np.abs(stft(noise, cfg)), q)
    true = estimate_profile(noise, cfg)
    assert abs(20 * np.log10(np.median(est / true))) < 0.3
