import numpy as np
import pytest

from app.dsp.stft import StftConfig, analysis_window, frame_times, istft, stft


@pytest.mark.parametrize("overlap", [0.5, 0.75])
@pytest.mark.parametrize("window", ["hann", "hamming", "sqrt_hann"])
@pytest.mark.parametrize("n_fft", [256, 512, 1024, 2048])
def test_perfect_reconstruction(n_fft, overlap, window):
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.3, 16000 + 123)          # length not a multiple of the hop
    cfg = StftConfig(n_fft, overlap, window)
    y_hat = istft(stft(y, cfg), len(y), cfg)
    assert y_hat.shape == y.shape
    assert np.max(np.abs(y_hat - y)) < 1e-6


def test_hann_50_overlap_sums_to_one():
    # Plain overlap-add needs no normalisation in this case (Simulink relies on it).
    cfg = StftConfig(512, 0.5, "hann")
    w = analysis_window(cfg)
    assert np.allclose(w[: cfg.hop] + w[cfg.hop :], 1.0)


def test_shapes_and_times():
    cfg = StftConfig(512, 0.5, "hann")
    spec = stft(np.zeros(8000), cfg)
    assert spec.shape[1] == 257
    t = frame_times(spec.shape[0], 16000, cfg)
    assert np.all(np.diff(t) > 0)


def test_rejects_stereo():
    with pytest.raises(ValueError):
        stft(np.zeros((100, 2)), StftConfig())
