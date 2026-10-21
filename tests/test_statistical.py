"""Hand-written statistical estimators (Eq. 22–34) and the algorithm registry (§14)."""

import json

import numpy as np
import pytest

from app.dsp import metrics, statistical as ST, synth
from app.dsp.noise_estimation import estimate_initial
from app.dsp.spectral_subtraction import SubtractionParams, denoise
from app.dsp.stft import StftConfig, istft, stft
from app.enhancers import registry
from app.enhancers.base import apply_attenuation_limit, attenuation_limit
from app.params import ParamError, validate

SR = 16000
CFG = StftConfig(512, 0.75, "hann")
DD = ("wiener_dd", "mmse_stsa", "log_mmse")


@pytest.fixture(scope="module")
def demo():
    d = synth.make_demo("white", snr_db=5.0, sr=SR, duration=6.0, seed=0)
    spec = stft(d["noisy"], CFG)
    mag = np.abs(spec)
    d["spec"], d["mag"] = spec, mag
    d["noise_mag"] = estimate_initial(mag, len(d["noisy"]), SR, CFG, 0.5)
    return d


def _si_sdr_gain(d, gain):
    out = istft(gain * d["spec"], len(d["noisy"]), CFG)
    assert np.all(np.isfinite(out)) and len(out) == len(d["noisy"])
    return metrics.si_sdr_db(d["clean"], out) - metrics.si_sdr_db(d["clean"], d["noisy"])


# --- gain functions -------------------------------------------------------------------------

def test_noise_power_from_mag_matches_rayleigh():
    """Eq. (22): for complex Gaussian noise, (4/π)(E|D|)² = E|D|²."""
    rng = np.random.default_rng(0)
    d = (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)) * np.sqrt(0.5 * 3.0)
    est = ST.noise_power_from_mag(np.array([np.abs(d).mean()]))[0]
    assert est == pytest.approx(3.0, rel=0.01)


def test_mmse_stsa_tends_to_wiener_at_high_snr():
    xi = np.array([100.0, 1000.0])
    gamma = xi + 1.0
    np.testing.assert_allclose(ST.mmse_stsa_gain(xi, gamma), ST.wiener_gain(xi), rtol=0.01)
    np.testing.assert_allclose(ST.log_mmse_gain(xi, gamma), ST.wiener_gain(xi), rtol=0.01)


def test_mmse_gains_known_value():
    """Hand-checked point: ξ = 1, γ = 2 → v = 1."""
    from scipy.special import exp1, i0, i1
    xi, gamma, v = np.array([1.0]), np.array([2.0]), 1.0
    expect = np.sqrt(np.pi) / 2 * np.sqrt(v) / 2.0 * np.exp(-v / 2) * ((1 + v) * i0(v / 2) + v * i1(v / 2))
    assert ST.mmse_stsa_gain(xi, gamma)[0] == pytest.approx(expect, rel=1e-12)
    assert ST.log_mmse_gain(xi, gamma)[0] == pytest.approx(0.5 * np.exp(0.5 * exp1(v)), rel=1e-12)


def test_gains_finite_for_extreme_inputs():
    xi = np.array([1e-6, 1e-3, 1.0, 1e3, 1e6])
    gamma = np.array([0.0, 1e-9, 1.0, 1e3, 1e9])
    for g in (ST.mmse_stsa_gain(xi, gamma), ST.log_mmse_gain(xi, gamma)):
        assert np.all(np.isfinite(g))


# --- full methods on the demo ---------------------------------------------------------------

@pytest.mark.parametrize("method", DD)
def test_decision_directed_improves_and_beats_spectral_subtraction(demo, method):
    lam = ST.noise_power_from_mag(demo["noise_mag"])
    g = ST.decision_directed_gain(demo["mag"] ** 2, lam, method, ST.DDParams())
    assert g.shape == demo["mag"].shape
    assert np.all((g >= 10 ** (-40 / 20) - 1e-12) & (g <= 1.0))
    gain_db = _si_sdr_gain(demo, g)
    ss = denoise(demo["noisy"], demo["noise_mag"], CFG, SubtractionParams())
    ss_db = metrics.si_sdr_db(demo["clean"], ss.cleaned_raw) - metrics.si_sdr_db(demo["clean"], demo["noisy"])
    assert gain_db > 6.0
    assert gain_db > ss_db


def test_omlsa_improves_and_tracks_noise(demo):
    lam = ST.noise_power_from_mag(demo["noise_mag"])
    r = ST.omlsa_gain(demo["mag"] ** 2, lam, CFG.hop / SR, ST.OmlsaParams())
    assert r.gain.shape == r.noise_power.shape == r.speech_prob.shape == demo["mag"].shape
    assert np.all((r.speech_prob >= 0) & (r.speech_prob <= 1))
    assert _si_sdr_gain(demo, r.gain) > 6.0


def test_omlsa_follows_a_noise_level_step():
    """IMCRA must re-track noise whose level jumps by +10 dB mid-clip (no speech)."""
    rng = np.random.default_rng(1)
    y = rng.standard_normal(SR * 8) * 0.01
    y[SR * 4:] *= np.sqrt(10.0)
    spec = stft(y, CFG)
    lam0 = ST.noise_power_from_mag(estimate_initial(np.abs(spec), len(y), SR, CFG, 0.5))
    r = ST.omlsa_gain(np.abs(spec) ** 2, lam0, CFG.hop / SR, ST.OmlsaParams())
    n = r.noise_power.shape[0]
    early = r.noise_power[n // 4: n // 2 - 20].mean()
    late = r.noise_power[-n // 8:].mean()
    assert 10 * np.log10(late / early) == pytest.approx(10.0, abs=2.0)


# --- attenuation limit (Eq. 34) -------------------------------------------------------------

def test_attenuation_limit_rule():
    assert attenuation_limit(100.0) == 0.0
    assert attenuation_limit(20.0) == pytest.approx(0.1)
    y, s, g = np.ones(4), np.zeros(4), np.zeros((2, 3))
    out, g2 = apply_attenuation_limit(y, s, g, 20.0)
    np.testing.assert_allclose(out, 0.1)
    np.testing.assert_allclose(g2, 0.1)
    out, g2 = apply_attenuation_limit(y, s, g, 100.0)
    assert out is s and g2 is g


# --- registry + params ----------------------------------------------------------------------

def test_registry_classical_algorithms_always_available():
    for a in ("spectral_subtraction", *DD, "omlsa"):
        assert a in registry.ALGORITHM_IDS
        assert registry.availability(a) == (True, "")
        assert registry.spec(a).hand_written
    js = registry.algorithms_json()
    json.dumps(js)
    assert {a["id"] for a in js} == set(registry.ALGORITHM_IDS)


def test_params_algorithm_switch_resets_algo_params():
    p = validate({"algorithm": "omlsa", "algo_params": {"g_min_db": -30}})
    assert p.algo_params == {"dd_alpha": 0.92, "g_min_db": -30.0, "tracking_window_s": 1.0}
    q = validate({"algorithm": "log_mmse"}, base=p)
    assert q.algo_params == registry.spec("log_mmse").defaults()
    r = validate({"alpha": 3.0}, base=p)             # same algorithm: algo params kept
    assert r.algo_params["g_min_db"] == -30.0


@pytest.mark.parametrize("bad", [
    {"algorithm": "magic"},
    {"algorithm": "omlsa", "algo_params": {"g_min_db": -50}},
    {"algorithm": "omlsa", "algo_params": {"dd_alpha": "0.9"}},
    {"algorithm": "omlsa", "algo_params": {"xi_min_db": -20}},   # belongs to the DD methods
    {"algo_params": {"dd_alpha": 0.9}},                            # SS has no algo params
    {"algo_params": [1]},
    {"atten_limit_db": 2.0}, {"atten_limit_db": 101.0},
])
def test_params_rejects_bad_algorithm_values(bad):
    with pytest.raises(ParamError):
        validate(bad)
