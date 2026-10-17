import numpy as np
import pytest

from app.dsp import metrics, synth

SR = 16000


@pytest.mark.parametrize("kind", synth.NOISE_TYPES)
@pytest.mark.parametrize("snr", [-5.0, 0.0, 10.0, 20.0])
def test_mix_exact_snr(kind, snr):
    d = synth.make_demo(kind, snr, SR, duration=3.0, seed=3)
    assert metrics.snr_db(d["clean"], d["noisy"]) == pytest.approx(snr, abs=1e-6)
    assert np.max(np.abs(d["noisy"])) <= 0.9 + 1e-9
    assert np.all(np.isfinite(d["noisy"]))


def test_deterministic():
    a = synth.make_demo("babble", 5, SR, 2.0, seed=7)
    b = synth.make_demo("babble", 5, SR, 2.0, seed=7)
    c = synth.make_demo("babble", 5, SR, 2.0, seed=8)
    assert np.array_equal(a["noisy"], b["noisy"])
    assert not np.array_equal(a["noisy"], c["noisy"])


def test_clean_has_leading_silence():
    x = synth.clean_voiced(SR, 3.0, seed=0)
    assert np.all(x[: int(0.8 * SR)] == 0)
    assert np.max(np.abs(x)) == pytest.approx(0.5)


def test_hum_has_50hz_peak():
    x = synth.make_noise("hum50", SR * 2, SR, seed=0)
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    assert abs(f[np.argmax(spec)] - 50.0) < 1.0


def test_si_sdr_properties():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 8000)
    assert metrics.si_sdr_db(s, 0.3 * s) > 100          # scale-invariant
    n = rng.normal(0, 1, 8000)
    assert metrics.si_sdr_db(s, s + n) == pytest.approx(0.0, abs=0.5)


def test_segmental_snr_clamped():
    s = synth.clean_voiced(SR, 3.0)
    assert metrics.segmental_snr_db(s, s) == pytest.approx(35.0)
    assert metrics.segmental_snr_db(s, np.zeros_like(s)) == pytest.approx(0.0, abs=1e-6)


def test_vad_and_noise_reduction():
    d = synth.make_demo("white", 10, SR, 4.0, seed=0)
    e = metrics.frame_energy_db(d["noisy"])
    speech = metrics.energy_vad(e)
    assert 0 < speech.mean() < 1
    assert not speech[:20].any()                        # leading 0.8 s is noise-only
    assert metrics.vad_snr_db(e, speech) is not None
    quieter = metrics.frame_energy_db(d["noisy"] * 0.1)
    assert metrics.noise_reduction_db(e, quieter, speech) == pytest.approx(20.0, abs=1e-6)
    assert metrics.vad_snr_db(e, np.ones_like(speech)) is None
