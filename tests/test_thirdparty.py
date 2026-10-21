"""Third-party enhancers (CLAUDE.md §14). Each test is skipped when its library or model is missing."""

import numpy as np
import pytest
from scipy.signal import correlate

from app.dsp import metrics, synth
from app.dsp.noise_estimation import estimate_initial
from app.dsp.stft import stft
from app.enhancers import registry
from app.enhancers.base import EnhanceContext
from app.params import validate
from tests.conftest import wav_bytes
from tests.test_api import _upload

pytestmark = pytest.mark.ml

THIRD_PARTY = ("noisereduce", "webrtc_ns", "rnnoise", "nsnet2", "deepfilternet3", "demucs")


def _need(algo: str) -> None:
    ok, reason = registry.availability(algo)
    if not ok:
        pytest.skip(f"{algo}: {reason}")


def _run(algo: str, y: np.ndarray, sr: int, **algo_params) -> np.ndarray:
    p = validate({"algorithm": algo, "algo_params": algo_params})
    spec = stft(y, p.stft)
    mag = np.abs(spec)
    ctx = EnhanceContext(y, sr, p.stft, spec, mag, estimate_initial(mag, len(y), sr, p.stft, 0.5))
    return registry.get(algo).process(ctx, p).cleaned


@pytest.fixture(scope="module", params=[16000, 48000])
def demo(request):
    sr = request.param
    d = synth.make_demo("white", 10.0, sr=sr, duration=5.0, seed=0)
    d["sr"] = sr
    return d


@pytest.mark.parametrize("algo", THIRD_PARTY)
def test_output_is_valid_aligned_and_improves(demo, algo):
    """Same length, finite, no residual delay (resampling + known model delays removed), SI-SDR gain > 2 dB."""
    _need(algo)
    y, c = demo["noisy"], demo["clean"]
    out = _run(algo, y, demo["sr"])
    assert len(out) == len(y) and np.all(np.isfinite(out))
    xc = correlate(out, c, mode="full", method="fft")
    lag = int(np.argmax(xc)) - (len(c) - 1)
    # ≤ 0.1 ms. The synthetic signal is harmonic and narrowband, so its cross-correlation peak is
    # broad; the exact delays (RNNoise 960 @ 48 kHz, WebRTC 96 @ 16 kHz) were set on real speech,
    # where the SI-SDR-optimal lag is exactly 0.
    tol = max(2, round(1e-4 * demo["sr"]))
    assert abs(lag) <= tol, f"{algo} output is misaligned by {lag} samples"
    assert metrics.si_sdr_db(c, out) - metrics.si_sdr_db(c, y) > 2.0


def test_webrtc_phase_compensation_helps_waveform_metrics():
    _need("webrtc_ns")
    d = synth.make_demo("white", 10.0, sr=16000, duration=5.0, seed=0)
    on = _run("webrtc_ns", d["noisy"], 16000, phase_compensation=True)
    off = _run("webrtc_ns", d["noisy"], 16000, phase_compensation=False)
    assert metrics.si_sdr_db(d["clean"], on) > metrics.si_sdr_db(d["clean"], off) + 3.0


def test_webrtc_levels_are_ordered():
    """Higher suppression level → less residual noise in the noise-only lead-in."""
    _need("webrtc_ns")
    d = synth.make_demo("white", 5.0, sr=16000, duration=5.0, seed=0)
    lead = slice(0, 4000)
    e = [np.mean(_run("webrtc_ns", d["noisy"], 16000, level=lv)[lead] ** 2) for lv in (1, 4)]
    assert e[1] < e[0]


def test_rnnoise_vad_gate_attenuates_noise_only_frames():
    _need("rnnoise")
    d = synth.make_demo("pink", 5.0, sr=48000, duration=5.0, seed=0)
    lead = slice(4800, 19200)                  # noise-only lead-in (0.1–0.4 s)
    plain = _run("rnnoise", d["noisy"], 48000)
    gated = _run("rnnoise", d["noisy"], 48000, vad_threshold=0.5, gate_db=30.0)
    assert np.mean(gated[lead] ** 2) < np.mean(plain[lead] ** 2)


def test_deepfilternet_post_filter_variant_runs():
    _need("deepfilternet3")
    d = synth.make_demo("white", 5.0, sr=48000, duration=3.0, seed=0)
    out = _run("deepfilternet3", d["noisy"], 48000, post_filter=True)
    assert np.all(np.isfinite(out)) and len(out) == len(d["noisy"])


def test_api_process_black_box_reports_effective_gain(client):
    _need("nsnet2")
    d = synth.make_demo("white", 5.0, sr=16000, duration=4.0, seed=0)
    r = _upload(client, wav_bytes(d["noisy"]), {"algorithm": "nsnet2"})
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["algorithm"] == {"id": "nsnet2", "name": "NSNet2", "family": "deep", "explicit_gain": False}
    assert j["metrics"]["noise_reduction_db"] > 3.0
    assert j["viz"]["gain_mask"]                 # effective mask |STFT(ŝ)|/|Y|
    f = client.get(f"/api/frame/{j['job_id']}?t=1.0").get_json()
    assert f["explicit_gain"] is False and f["floor_db"] is None



# --- §14 perceptual metrics ------------------------------------------------------------------

def _need_quality(name: str) -> None:
    from app.enhancers import quality
    if not quality.available()[name]:
        pytest.skip(f"{name} not installed")


def test_pesq_stoi_ordering():
    _need_quality("pesq")
    _need_quality("stoi")
    from app.enhancers import quality
    d = synth.make_demo("white", 5.0, sr=16000, duration=5.0, seed=0)
    ident = quality.intrusive(d["clean"], d["clean"], 16000)
    noisy = quality.intrusive(d["clean"], d["noisy"], 16000)
    assert ident["pesq"] > 4.4 and ident["stoi"] > 0.999
    assert noisy["pesq"] < ident["pesq"] and noisy["stoi"] < ident["stoi"] and noisy["estoi"] < noisy["stoi"]


def test_dnsmos_rates_clean_background_higher():
    _need_quality("dnsmos")
    from app.enhancers import quality
    d = synth.make_demo("white", 0.0, sr=16000, duration=4.0, seed=0)
    q_clean, q_noisy = quality.dnsmos(d["clean"], 16000), quality.dnsmos(d["noisy"], 16000)
    assert set(q_clean) == {"sig", "bak", "ovrl"}
    assert q_clean["bak"] > q_noisy["bak"] + 1.0
    assert all(1.0 <= v <= 5.0 for v in (*q_clean.values(), *q_noisy.values()))


def test_api_demo_reports_perceptual_metrics_and_dnsmos(client):
    _need_quality("pesq")
    _need_quality("dnsmos")
    j = client.post("/api/demo", json={"noise_type": "white", "snr_db": 5, "duration": 4,
                                       "params": {"algorithm": "log_mmse"}}).get_json()
    gt = j["metrics"]["ground_truth"]
    assert gt["pesq_out"] > gt["pesq_in"] and gt["stoi_out"] is not None and gt["lsd_out_db"] < gt["lsd_in_db"]
    q = client.post(f"/api/quality/{j['job_id']}")
    assert q.status_code == 200, q.get_json()
    qj = q.get_json()
    assert qj["output"]["bak"] > qj["input"]["bak"] and qj["version"] == j["version"]
