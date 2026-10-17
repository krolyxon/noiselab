import pytest

from app.params import DEFAULTS, PRESETS, SCHEMA, ParamError, validate


def test_defaults_match_spec():
    d = DEFAULTS
    assert (d.n_fft, d.overlap, d.window, d.noise_mode, d.noise_seconds) == (512, 0.5, "hann", "initial", 0.5)
    assert (d.alpha, d.beta, d.exponent, d.method) == (2.0, 0.02, 2, "spectral_subtraction")
    assert (d.gain_smooth, d.freq_smooth_bins, d.output_gain_db) == (0.0, 0, 0.0)


@pytest.mark.parametrize("bad", [
    {"n_fft": 300}, {"n_fft": 512.5}, {"overlap": 0.6}, {"window": "kaiser"},
    {"alpha": 0.5}, {"alpha": 6.1}, {"beta": -0.01}, {"beta": 0.3}, {"exponent": 3},
    {"noise_seconds": 0.05}, {"noise_seconds": 3.5}, {"gain_smooth": 0.99},
    {"freq_smooth_bins": 10}, {"freq_smooth_bins": -1}, {"output_gain_db": 13},
    {"method": "dnn"}, {"noise_mode": "magic"}, {"alpha": "2"}, {"alpha": True},
    {"alpha": float("nan")}, {"unknown_key": 1},
    {"noise_mode": "region", "region_start": 1.0, "region_end": 0.5},
    {"noise_mode": "profile"},
])
def test_rejects_out_of_range(bad):
    with pytest.raises(ParamError):
        validate(bad)


def test_rejects_non_object():
    with pytest.raises(ParamError):
        validate([1, 2])  # type: ignore[arg-type]


def test_accepts_bounds_and_int_for_float():
    p = validate({"alpha": 1, "beta": 0.2, "n_fft": 2048, "freq_smooth_bins": 9, "output_gain_db": -12})
    assert p.alpha == 1.0 and isinstance(p.alpha, float)
    assert p.stft.hop == 1024


def test_presets_are_complete_and_valid():
    assert {"match_simulink_basic", "faithful_light", "balanced", "aggressive", "mains_hum_50", "fan_ac"} <= set(PRESETS)
    for pr in PRESETS.values():
        assert set(pr["params"]) == set(SCHEMA)
        validate(pr["params"])


def test_match_simulink_basic_is_exact():
    p = PRESETS["match_simulink_basic"]["params"]
    assert p["n_fft"] == 512 and p["overlap"] == 0.5 and p["window"] == "hann"
    assert p["noise_mode"] == "initial" and p["noise_seconds"] == 0.5
    assert p["alpha"] == 2.0 and p["beta"] == 0.02 and p["exponent"] == 1
    assert p["method"] == "spectral_subtraction"
    assert p["gain_smooth"] == 0.0 and p["freq_smooth_bins"] == 0 and p["output_gain_db"] == 0.0
