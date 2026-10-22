"""HTTP layer: page + /api/*. All heavy lifting is in services/ and dsp/."""

from __future__ import annotations

import base64
import io
import json
import platform
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from typing import Any

import numpy as np
import scipy
import soundfile as sf
from flask import Blueprint, current_app, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from . import __version__, glossary
from .dsp import synth
from .dsp.noise_estimation import estimate_profile
from .dsp.stft import StftConfig
from .enhancers import quality as Q
from .enhancers import registry
from .params import PRESETS, SCHEMA, ParamError, schema_json, validate
from .services import processing
from .services.audio_io import AudioError, load_audio, supported_formats
from .services.jobs import Job, JobStore, ProfileStore, valid_id

bp = Blueprint("main", __name__)
AUDIO_KINDS = ("original", "cleaned", "noise", "clean")      # clean = demo ground truth only
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PLOTS, MAX_PLOT_BYTES = 12, 4 * 1024 * 1024


class APIError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def _jobs() -> JobStore:
    return current_app.extensions["jobs"]


def _profiles() -> ProfileStore:
    return current_app.extensions["profiles"]


def _get_job(job_id: str) -> Job:
    job = _jobs().get(job_id) if valid_id(job_id) else None
    if job is None:
        raise APIError("job not found or expired (jobs are kept for 30 minutes); upload again", 404)
    return job


def _json_field(raw: str | None, name: str) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    if len(raw) > 10_000:
        raise APIError(f"{name} is too large")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise APIError(f"{name} is not valid JSON: {exc.msg}") from exc


def _body_json() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if data is None:
        if request.data:
            raise APIError("request body is not valid JSON")
        return {}
    if not isinstance(data, dict):
        raise APIError("request body must be a JSON object")
    return data


def _read_upload(field: str, max_duration: float) -> tuple[np.ndarray, int, str]:
    f = request.files.get(field)
    if f is None:
        raise APIError(f"missing file field '{field}'")
    name = secure_filename(f.filename or "") or "audio"
    data = f.read()
    cfg = current_app.config
    y, sr = load_audio(data, max_duration=max_duration, min_duration=cfg["MIN_DURATION_S"],
                       max_sr=cfg["MAX_SAMPLE_RATE"])
    return y, sr, name


def _run(job: Job, params: dict[str, Any]) -> dict[str, Any]:
    p = validate(params)
    profile = _profiles().get(p.profile_id) if p.noise_mode == "profile" and valid_id(p.profile_id) else None
    with job.lock:
        return processing.run(job, p, profile)


# --- error handling -----------------------------------------------------------------------

@bp.app_errorhandler(APIError)
def _api_error(e: APIError):
    return jsonify(error=e.message), e.status


@bp.app_errorhandler(ParamError)
def _param_error(e: ParamError):
    return jsonify(error=f"invalid parameter: {e}"), 400


@bp.app_errorhandler(AudioError)
def _audio_error(e: AudioError):
    return jsonify(error=str(e)), e.status


@bp.app_errorhandler(processing.ProcessingError)
def _processing_error(e: processing.ProcessingError):
    return jsonify(error=str(e)), 422


@bp.app_errorhandler(RequestEntityTooLarge)
def _too_large(e):
    mb = current_app.config["MAX_CONTENT_LENGTH"] / 1024 / 1024
    return jsonify(error=f"upload too large (maximum {mb:.0f} MB)"), 413


@bp.app_errorhandler(HTTPException)
def _http_error(e: HTTPException):
    if request.path.startswith("/api/"):
        return jsonify(error=e.description), e.code
    return e


@bp.before_app_request
def _purge():
    _jobs().purge()
    _profiles().purge()


# --- pages ---------------------------------------------------------------------------------

@bp.get("/")
def index():
    return render_template("index.html", version=__version__)


# --- API -----------------------------------------------------------------------------------

@bp.get("/api/health")
def health():
    fmts = supported_formats()
    cfg = current_app.config
    return jsonify(
        status="ok", version=__version__, python=platform.python_version(),
        libraries={"flask": pkg_version("flask"),
                   "numpy": np.__version__, "scipy": scipy.__version__,
                   "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__},
        formats=fmts, mp3_supported="MP3" in fmts,
        limits={"max_upload_mb": cfg["MAX_CONTENT_LENGTH"] // (1024 * 1024),
                "max_duration_s": cfg["MAX_DURATION_S"], "max_sample_rate": cfg["MAX_SAMPLE_RATE"],
                "job_ttl_s": cfg["JOB_TTL_S"]},
        noise_types=list(synth.NOISE_TYPES),
        quality_metrics=Q.available(),
        algorithms={a["id"]: {"available": a["available"], "reason": a["reason"]}
                    for a in registry.algorithms_json()},
    )


@bp.post("/api/compare/<job_id>/<algo>")
def compare(job_id: str, algo: str):
    """One algorithm of the Compare panel (§14). Body: {"params": flat params, "dnsmos": bool}.

    The flat parameters (framing, noise mode, attenuation limit, output gain) come from the
    UI. The algorithm-specific ones are that algorithm's defaults, so every method is compared
    as shipped. The client calls this once per algorithm, which gives it progress for free.
    """
    if algo not in registry.ALGORITHM_IDS:
        raise APIError(f"unknown algorithm {algo!r}", 404)
    job = _get_job(job_id)
    body = _body_json()
    raw = dict(body.get("params", {}))
    if not isinstance(raw, dict):
        raise APIError("params must be a JSON object")
    raw.pop("algo_params", None)
    raw["algorithm"] = algo
    p = validate(raw)
    profile = _profiles().get(p.profile_id) if p.noise_mode == "profile" and valid_id(p.profile_id) else None
    with job.lock:
        return jsonify(processing.compare_one(job, p, profile, dnsmos=bool(body.get("dnsmos", False))))


@bp.post("/api/quality/<job_id>")
def quality(job_id: str):
    """DNSMOS of the input and of the latest output (slow, so it is a separate call)."""
    job = _get_job(job_id)
    with job.lock:
        return jsonify(processing.perceptual_quality(job))


@bp.get("/api/glossary")
def glossary_route():
    """Plain-language explanations of every metric and term (app/glossary.py)."""
    return jsonify(glossary.as_json())


@bp.get("/api/algorithms")
def algorithms():
    return jsonify(algorithms=registry.algorithms_json(), default=registry.DEFAULT_ALGORITHM)


@bp.get("/api/presets")
def presets():
    return jsonify(presets=list(PRESETS.values()), schema=schema_json())


@bp.post("/api/process")
def process():
    params = _json_field(request.form.get("params"), "params")
    validate(params)                                   # fail fast before decoding audio
    y, sr, name = _read_upload("audio", current_app.config["MAX_DURATION_S"])
    job = _jobs().create(y, sr, name=name)
    return jsonify(_run(job, params))


@bp.post("/api/reprocess/<job_id>")
def reprocess(job_id: str):
    job = _get_job(job_id)
    body = _body_json()
    return jsonify(_run(job, body.get("params", {})))


@bp.post("/api/noise-profile")
def noise_profile():
    y, sr, _ = _read_upload("audio", current_app.config["MAX_PROFILE_DURATION_S"])
    prof = _profiles().create(y, sr)
    return jsonify(profile_id=prof.id, sr=sr, duration=round(len(y) / sr, 3))


@bp.get("/api/noise-profile/<profile_id>/spectrum")
def noise_profile_spectrum(profile_id: str):
    """|N̂| of a recorded profile (Eq. 4) for the live preview: Hann, 50 %, given n_fft, at the
    AudioContext's sample rate (the clip is resampled if needed)."""
    prof = _profiles().get(profile_id) if valid_id(profile_id) else None
    if prof is None:
        raise APIError("noise profile not found or expired; record it again", 404)
    n_fft = request.args.get("n_fft", default=512, type=int)
    sr = request.args.get("sr", type=int)
    if n_fft not in SCHEMA["n_fft"].choices:
        raise APIError(f"n_fft must be one of {list(SCHEMA['n_fft'].choices)}")
    if sr is None or not 8000 <= sr <= 192000:
        raise APIError("sr (8000–192000 Hz) is required")
    clip = prof.y
    if prof.sr != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(prof.sr, sr)
        clip = resample_poly(clip, sr // g, prof.sr // g)
    mag = estimate_profile(clip, StftConfig(n_fft, 0.5, "hann"))
    return jsonify(profile_id=prof.id, sr=sr, n_fft=n_fft, noise_mag=mag.tolist())


@bp.post("/api/demo")
def demo():
    """Synthetic clean + noise mix with ground truth. JSON body, or multipart with an
    optional user-supplied 'clean' WAV and an 'options' JSON field."""
    if request.files:
        opts = _json_field(request.form.get("options"), "options")
    else:
        opts = _body_json()
    allowed = {"noise_type", "snr_db", "duration", "seed", "params"}
    if set(opts) - allowed:
        raise APIError(f"unknown demo option(s): {', '.join(sorted(set(opts) - allowed))}")

    noise_type = opts.get("noise_type", "white")
    if noise_type not in synth.NOISE_TYPES:
        raise APIError(f"noise_type must be one of {list(synth.NOISE_TYPES)}")
    snr = _number(opts.get("snr_db", 5.0), "snr_db", -5.0, 20.0)
    max_d = current_app.config["DEMO_MAX_DURATION_S"]
    duration = _number(opts.get("duration", 6.0), "duration", 2.0, max_d)
    seed = int(_number(opts.get("seed", 0), "seed", 0, 2**31 - 1))
    params = opts.get("params", {})
    validate(params)

    sr = 16000
    clean = None
    if "clean" in request.files:
        clean, sr, _ = _read_upload("clean", max_d)
    d = synth.make_demo(noise_type, snr, sr=sr, duration=duration, seed=seed, clean=clean)
    info = {"noise_type": noise_type, "snr_db": snr, "duration": round(len(d["clean"]) / sr, 3),
            "seed": seed, "source": "uploaded clean file" if clean is not None else "synthetic voiced signal"}
    job = _jobs().create(d["noisy"], sr, name=f"demo-{noise_type}-{snr:g}dB", clean=d["clean"], demo=info)
    return jsonify(_run(job, params))


def _number(v: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise APIError(f"{name} must be a number")
    if not lo <= v <= hi:
        raise APIError(f"{name} must be in [{lo:g}, {hi:g}]")
    return float(v)


@bp.post("/api/sweep/<job_id>")
def sweep(job_id: str):
    """α × β grid of ground-truth improvements for a demo job (other params from the body)."""
    job = _get_job(job_id)
    p = validate(_body_json().get("params", {}))
    profile = _profiles().get(p.profile_id) if p.noise_mode == "profile" and valid_id(p.profile_id) else None
    with job.lock:
        return jsonify(processing.sweep(job, p, profile))


@bp.get("/api/audio/<job_id>/<kind>.wav")
def audio(job_id: str, kind: str):
    compared = kind.startswith("cleaned_") and kind[len("cleaned_"):] in registry.ALGORITHM_IDS
    if kind not in AUDIO_KINDS and not compared:
        raise APIError(f"audio kind must be one of {list(AUDIO_KINDS)} or cleaned_<algorithm>", 404)
    job = _get_job(job_id)
    path = job.dir / f"{kind}.wav"
    if not path.is_file():
        raise APIError("audio not ready", 404)
    return send_file(path, mimetype="audio/wav", download_name=f"{job.name}-{kind}.wav", max_age=0)


@bp.get("/api/frame/<job_id>")
def frame(job_id: str):
    job = _get_job(job_id)
    t = request.args.get("t", type=float)
    if t is None or not np.isfinite(t):
        raise APIError("query parameter t (seconds) is required")
    with job.lock:
        return jsonify(processing.inspect_frame(job, t))


@bp.route("/api/export/<job_id>", methods=["GET", "POST"])
def export(job_id: str):
    """ZIP: original/cleaned/removed-noise (and clean, demo only) WAVs, params.json, metrics.json,
    README.txt — plus plot PNGs when POSTed as {"plots": {"name": "data:image/png;base64,..."}}."""
    job = _get_job(job_id)
    plots: dict[str, bytes] = {}
    if request.method == "POST":
        raw = _body_json().get("plots", {})
        if not isinstance(raw, dict) or len(raw) > MAX_PLOTS:
            raise APIError(f"plots must be an object with at most {MAX_PLOTS} entries")
        for name, url in raw.items():
            safe = secure_filename(str(name))[:40]
            if not safe or not isinstance(url, str) or not url.startswith("data:image/png;base64,"):
                raise APIError(f"plot {name!r} must be a PNG data URL")
            try:
                data = base64.b64decode(url.split(",", 1)[1], validate=True)
            except ValueError as exc:
                raise APIError(f"plot {name!r} is not valid base64") from exc
            if not data.startswith(PNG_SIGNATURE) or len(data) > MAX_PLOT_BYTES:
                raise APIError(f"plot {name!r} is not a PNG (or larger than 4 MB)")
            plots[safe] = data

    with job.lock:
        if not job.last:
            raise APIError("job has not been processed yet", 409)
        params = job.last["params"].to_dict()
        metrics = job.last["metrics"]
        files = {k: job.dir / f"{k}.wav" for k in AUDIO_KINDS if (job.dir / f"{k}.wav").is_file()}
        names = {"original": "noisy_input.wav", "cleaned": "cleaned.wav", "noise": "removed_noise.wav",
                 "clean": "clean_reference.wav"}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for k, path in files.items():
                z.write(path, names[k])
            z.writestr("params.json", json.dumps(params, indent=2))
            z.writestr("metrics.json", json.dumps(metrics, indent=2))
            for name, data in plots.items():
                z.writestr(f"plots/{name}.png", data)
            z.writestr("README.txt", _export_readme(job, params, sorted(names[k] for k in files), sorted(plots)))
    buf.seek(0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"noiselab-{job.name}-{stamp}.zip", max_age=0)


def _export_readme(job: Job, params: dict, wavs: list[str], plots: list[str]) -> str:
    lines = [
        f"NoiseLab {__version__} export — {job.name}",
        f"Created {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "This web app is an extension of the Simulink model; the Simulink model is the primary implementation.",
        "Algorithm: spectral subtraction (Boll 1979, Berouti et al. 1979) — see docs/ALGORITHM.md.",
        "",
        f"Sample rate {job.sr} Hz, duration {job.duration:.3f} s.",
        "WAV files (16-bit PCM): " + ", ".join(wavs),
        "  removed_noise.wav = noisy_input − cleaned (before output gain / peak protection).",
        "params.json  — every parameter used for cleaned.wav.",
        "metrics.json — ground-truth metrics are null unless this was a demo with a clean reference.",
    ]
    if plots:
        lines.append("plots/       — " + ", ".join(f"{p}.png" for p in plots))
    if job.demo:
        lines += ["", "Demo: " + json.dumps(job.demo)]
    return "\n".join(lines) + "\n"
