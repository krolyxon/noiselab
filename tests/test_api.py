import base64
import io
import json
import time

import numpy as np
import pytest
import soundfile as sf

from app.dsp import synth
from tests.conftest import wav_bytes

EXPECTED_KEYS = {"job_id", "sr", "duration", "metrics", "viz", "audio", "params", "version"}


def _upload(client, data: bytes, params=None, name="clip.wav"):
    form = {"audio": (io.BytesIO(data), name)}
    if params is not None:
        form["params"] = json.dumps(params)
    return client.post("/api/process", data=form, content_type="multipart/form-data")


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.get_json()
    assert j["status"] == "ok"
    assert {"WAV", "FLAC", "OGG"} <= set(j["formats"])
    assert isinstance(j["mp3_supported"], bool)


def test_presets(client):
    j = client.get("/api/presets").get_json()
    assert any(p["id"] == "match_simulink_basic" for p in j["presets"])
    assert j["schema"]["alpha"]["min"] == 1.0


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and b"extension of the Simulink model" in r.data


def test_process_upload(client, demo_wav):
    r = _upload(client, demo_wav)
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert EXPECTED_KEYS <= set(j)
    assert j["sr"] == 16000 and j["duration"] == pytest.approx(4.0)
    m = j["metrics"]
    assert m["ground_truth"] is None                     # never faked for uploads
    assert m["noise_reduction_db"] > 3
    for k in ("input_rms_dbfs", "output_rms_dbfs", "noise_floor_dbfs", "real_time_factor", "processing_ms"):
        assert isinstance(m[k], float)

    viz = j["viz"]
    assert len(viz["waveform"]["original"]["t"]) <= 2000
    sp = viz["spectrogram"]
    nf, nt = sp["noisy"]["shape"]
    assert nf == len(sp["f"]) == 257 and nt == len(sp["t"]) <= 400
    raw = np.frombuffer(base64.b64decode(sp["noisy"]["b64"]), dtype="<i2")
    assert raw.size == nf * nt
    assert len(viz["spectra"]["noise_profile_db"]) == 257

    for kind, url in j["audio"].items():
        a = client.get(url)
        assert a.status_code == 200 and a.mimetype == "audio/wav", kind
        y, sr = sf.read(io.BytesIO(a.data))
        assert sr == 16000 and len(y) == 64000


def test_cleaned_plus_noise_is_original(client, demo_wav):
    j = _upload(client, demo_wav).get_json()
    get = lambda k: sf.read(io.BytesIO(client.get(j["audio"][k]).data))[0]
    assert np.max(np.abs(get("cleaned") + get("noise") - get("original"))) < 3 / 32768


def test_reprocess_and_frame(client, demo_wav):
    j = _upload(client, demo_wav).get_json()
    jid = j["job_id"]
    r = client.post(f"/api/reprocess/{jid}", json={"params": {"alpha": 4.0, "method": "wiener"}})
    assert r.status_code == 200
    j2 = r.get_json()
    assert j2["version"] == 2 and j2["params"]["alpha"] == 4.0
    assert j2["audio"]["cleaned"].endswith("v=2")

    f = client.get(f"/api/frame/{jid}?t=2.0").get_json()
    assert len(f["noisy_db"]) == len(f["cleaned_db"]) == len(f["alpha_noise_db"]) == 257
    assert abs(f["t"] - 2.0) < 0.02

    assert client.post(f"/api/reprocess/{jid}", json={"params": {"alpha": 99}}).status_code == 400
    assert client.post("/api/reprocess/" + "0" * 32, json={}).status_code == 404
    assert client.post("/api/reprocess/../../etc", json={}).status_code == 404


def test_reprocess_fast_30s(client):
    d = synth.make_demo("pink", 5.0, 16000, duration=20.0, seed=1)
    y = np.concatenate([d["noisy"], d["noisy"][: 10 * 16000]])           # 30 s
    jid = _upload(client, wav_bytes(y)).get_json()["job_id"]
    times = []
    for a in (2.5, 3.0, 3.5):
        t0 = time.perf_counter()
        r = client.post(f"/api/reprocess/{jid}", json={"params": {"alpha": a}})
        times.append(time.perf_counter() - t0)
        assert r.status_code == 200
    print(f"reprocess 30 s clip: {[round(t * 1000) for t in times]} ms")
    assert min(times) < 0.3


def test_noise_modes(client, demo_wav):
    noise = synth.make_noise("white", 32000, 16000, seed=5) * 0.05
    pr = client.post("/api/noise-profile", data={"audio": (io.BytesIO(wav_bytes(noise)), "n.wav")},
                     content_type="multipart/form-data")
    assert pr.status_code == 200
    pid = pr.get_json()["profile_id"]
    for params in ({"noise_mode": "profile", "profile_id": pid},
                   {"noise_mode": "region", "region_start": 0.0, "region_end": 0.7},
                   {"noise_mode": "minstats"}):
        r = _upload(client, demo_wav, params)
        assert r.status_code == 200, (params, r.get_json())
    r = _upload(client, demo_wav, {"noise_mode": "profile", "profile_id": "f" * 32})
    assert r.status_code == 422
    r = _upload(client, demo_wav, {"noise_mode": "region", "region_start": 3.9, "region_end": 3.91})
    assert r.status_code == 422


def test_demo_ground_truth(client):
    r = client.post("/api/demo", json={"noise_type": "white", "snr_db": 5, "duration": 4, "seed": 0})
    assert r.status_code == 200
    gt = r.get_json()["metrics"]["ground_truth"]
    assert gt["snr_in_db"] == pytest.approx(5.0, abs=0.01)
    assert gt["snr_improvement_db"] > 3
    assert gt["si_sdr_out_db"] > gt["si_sdr_in_db"]
    for bad in ({"snr_db": 30}, {"noise_type": "rain"}, {"duration": 500}, {"what": 1},
                {"params": {"alpha": 0}}):
        assert client.post("/api/demo", json=bad).status_code == 400, bad


def test_demo_with_user_clean(client):
    clean = synth.clean_voiced(16000, 3.0, seed=4)
    r = client.post("/api/demo", data={"clean": (io.BytesIO(wav_bytes(clean)), "c.wav"),
                                       "options": json.dumps({"noise_type": "hum50", "snr_db": 0})},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.get_json()["demo"]["source"] == "uploaded clean file"


@pytest.mark.parametrize("fmt,subtype", [("FLAC", "PCM_16"), ("OGG", "VORBIS"), ("WAV", "FLOAT")])
def test_other_formats_and_stereo(client, fmt, subtype):
    d = synth.make_demo("fan", 5.0, 16000, duration=2.0)
    stereo = np.stack([d["noisy"], d["noisy"]], axis=1)
    r = _upload(client, wav_bytes(stereo, fmt=fmt, subtype=subtype), name="x.bin")
    assert r.status_code == 200, r.get_json()


def test_resamples_above_48k(client):
    y = synth.make_demo("white", 5.0, 96000, duration=1.0)["noisy"]
    j = _upload(client, wav_bytes(y, sr=96000)).get_json()
    assert j["sr"] == 48000 and j["duration"] == pytest.approx(1.0)


def test_rejects_bad_uploads(app, client):
    # garbage bytes with a .wav name → unsupported/undecodable, clean JSON error
    r = _upload(client, b"not audio at all" * 100)
    assert r.status_code == 415 and "error" in r.get_json()
    # missing file field
    r = client.post("/api/process", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    # too long (> 120 s) — 121 s of silence at 8 kHz is small enough to upload
    r = _upload(client, wav_bytes(np.zeros(121 * 8000), sr=8000))
    assert r.status_code == 413 and "120" in r.get_json()["error"]
    # too short
    r = _upload(client, wav_bytes(np.zeros(800)))
    assert r.status_code == 400
    # bad params JSON / bad values (checked before decoding)
    assert _upload(client, b"x", params=None).status_code == 415
    form = {"audio": (io.BytesIO(b"x"), "a.wav"), "params": "{bad json"}
    assert client.post("/api/process", data=form, content_type="multipart/form-data").status_code == 400
    assert _upload(client, b"x", {"n_fft": 333}).status_code == 400


def test_rejects_oversize(app):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    c = app.test_client()
    r = _upload(c, b"\0" * 4096)
    assert r.status_code == 413 and "too large" in r.get_json()["error"]


def test_audio_route_validation(client, demo_wav):
    jid = _upload(client, demo_wav).get_json()["job_id"]
    assert client.get(f"/api/audio/{jid}/secret.wav").status_code == 404
    assert client.get(f"/api/audio/{'a' * 32}/cleaned.wav").status_code == 404
    assert client.get("/api/frame/" + jid).status_code == 400


def test_job_expiry(app, client, demo_wav):
    jid = _upload(client, demo_wav).get_json()["job_id"]
    store = app.extensions["jobs"]
    job_dir = store.get(jid).dir
    store.get(jid).accessed -= 31 * 60
    assert client.post(f"/api/reprocess/{jid}", json={}).status_code == 404
    assert not job_dir.exists()


def test_frame_inspector_curves_match_method(client, demo_wav):
    jid = _upload(client, demo_wav, {"alpha": 4.0, "beta": 0.04, "exponent": 2}).get_json()["job_id"]
    f = client.get(f"/api/frame/{jid}?t=2.0").get_json()
    noisy, sub, floor, clean = (np.array(f[k]) for k in ("noisy_db", "alpha_noise_db", "floor_db", "cleaned_db"))
    # spectral subtraction, p = 2: subtracted level is α^(1/2)|N̂| (+6.02 dB over |N̂| for α = 4),
    # floor is β^(1/2)|N̂| (−13.98 dB); both relative to the same |N̂|
    assert np.allclose(sub - floor, 20 * np.log10(2.0 / 0.2), atol=0.15)
    # output never below the floor (up to the 0.1 dB rounding)
    assert np.all(clean >= floor - 0.15)

    client.post(f"/api/reprocess/{jid}", json={"params": {"method": "wiener", "alpha": 4.0, "beta": 0.04}})
    w = client.get(f"/api/frame/{jid}?t=2.0").get_json()
    # Wiener: floor is β·|Y| (−27.96 dB below |Y|), and |S̃| ≥ β·|Y|
    assert np.allclose(np.array(w["noisy_db"]) - np.array(w["floor_db"]), -20 * np.log10(0.04), atol=0.15)
    assert np.all(np.array(w["cleaned_db"]) >= np.array(w["floor_db"]) - 0.15)


def _zip(resp):
    import zipfile
    assert resp.status_code == 200 and resp.mimetype == "application/zip", resp.data[:200]
    return zipfile.ZipFile(io.BytesIO(resp.data))


def test_export_zip_upload_and_demo(client, demo_wav):
    jid = _upload(client, demo_wav, {"alpha": 3.0}).get_json()["job_id"]
    z = _zip(client.get(f"/api/export/{jid}"))
    assert {"noisy_input.wav", "cleaned.wav", "removed_noise.wav", "params.json", "metrics.json",
            "README.txt"} <= set(z.namelist())
    assert "clean_reference.wav" not in z.namelist()
    assert json.loads(z.read("params.json"))["alpha"] == 3.0
    assert json.loads(z.read("metrics.json"))["ground_truth"] is None
    assert b"primary implementation" in z.read("README.txt")

    d = client.post("/api/demo", json={"noise_type": "pink", "snr_db": 5, "duration": 3}).get_json()
    assert "clean" in d["audio"] and client.get(d["audio"]["clean"]).status_code == 200
    z = _zip(client.get(f"/api/export/{d['job_id']}"))
    assert "clean_reference.wav" in z.namelist()
    assert json.loads(z.read("metrics.json"))["ground_truth"]["snr_in_db"] == pytest.approx(5.0, abs=0.01)


def test_export_with_plots_validated(client, demo_wav):
    import base64 as b64
    jid = _upload(client, demo_wav).get_json()["job_id"]
    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
    url = "data:image/png;base64," + b64.b64encode(png).decode()
    z = _zip(client.post(f"/api/export/{jid}", json={"plots": {"../../evil name": url, "waveform": url}}))
    assert "plots/waveform.png" in z.namelist()
    assert all(".." not in n and not n.startswith("/") for n in z.namelist())
    bad = "data:image/png;base64," + b64.b64encode(b"GIF89a....").decode()
    assert client.post(f"/api/export/{jid}", json={"plots": {"x": bad}}).status_code == 400
    assert client.post(f"/api/export/{jid}", json={"plots": {"x": "javascript:alert(1)"}}).status_code == 400
    assert client.get("/api/export/" + "b" * 32).status_code == 404
