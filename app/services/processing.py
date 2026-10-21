"""Run the algorithm on a job and assemble metrics + plot data. No Flask imports."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from ..dsp import metrics as M
from ..dsp import viz_data as V
from ..dsp.noise_estimation import estimate_initial, estimate_minstats, estimate_profile, estimate_region
from ..dsp.spectral_subtraction import compute_gain, output_stage, subtracted_magnitude
from ..dsp.stft import istft, stft
from ..dsp.stft import StftConfig, analysis_window, frame_times
from ..enhancers import quality as Q
from ..enhancers import registry
from ..enhancers.base import EnhanceContext, EnhanceResult, apply_attenuation_limit, attenuation_limit, warm
from ..params import Params, validate
from .audio_io import write_wav
from .jobs import Job, NoiseProfile


class ProcessingError(ValueError):
    pass


@dataclass(frozen=True)
class _Out:
    cleaned: np.ndarray          # after output stage (Eq. 14)
    cleaned_raw: np.ndarray      # before output stage (used for y − ŝ and all metrics)
    gain: np.ndarray             # explicit or effective G(k, m)


def _round(x: float | None, nd: int = 2) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), nd)


def estimate_noise(job: Job, p: Params, mag: np.ndarray, profile: NoiseProfile | None) -> np.ndarray:
    cfg = p.stft
    n = len(job.y)
    try:
        if p.noise_mode == "initial":
            return estimate_initial(mag, n, job.sr, cfg, p.noise_seconds)
        if p.noise_mode == "region":
            if p.region_end > job.duration + 1e-9:
                raise ProcessingError(f"region_end {p.region_end:.2f} s is beyond the clip ({job.duration:.2f} s)")
            return estimate_region(mag, n, job.sr, cfg, p.region_start, p.region_end)
        if p.noise_mode == "profile":
            if profile is None:
                raise ProcessingError("noise profile not found or expired; record it again")
            clip = profile.y
            if profile.sr != job.sr:                     # match the job's sample rate
                from math import gcd
                g = gcd(profile.sr, job.sr)
                clip = resample_poly(clip, job.sr // g, profile.sr // g)
            return estimate_profile(clip, cfg)
        if p.noise_mode == "minstats":
            return estimate_minstats(mag, p.minstats_percentile)
    except ValueError as exc:
        if isinstance(exc, ProcessingError):
            raise
        raise ProcessingError(str(exc)) from exc
    raise ProcessingError(f"unknown noise mode {p.noise_mode}")


def noise_floor_dbfs(noise_mag: np.ndarray, cfg: StftConfig) -> float:
    """Per-sample noise power implied by |N̂| (Parseval on a windowed frame).

    Uses (mean |Y|)², which underestimates mean power by ~1 dB for Gaussian noise.
    """
    w = analysis_window(cfg)
    p2 = noise_mag**2
    two_sided = p2[0] + p2[-1] + 2.0 * p2[1:-1].sum()
    return float(M.to_db(two_sided / (cfg.n_fft * np.sum(w**2))))


_GAIN_CAP = 10.0      # effective masks are |Ŝ|/|Y|; cap where |Y| ≈ 0 so plots stay readable


def effective_gain(cleaned: np.ndarray, mag: np.ndarray, cfg: StftConfig) -> np.ndarray:
    """|STFT(ŝ)| / |Y| on the analysis grid, for models that do not expose a gain (§14)."""
    return np.minimum(np.abs(stft(cleaned, cfg)) / np.maximum(mag, 1e-12), _GAIN_CAP)


def _cache_key(p: Params) -> str:
    """Everything that changes the enhancer's output (not the attenuation limit / output gain)."""
    d = p.to_dict()
    for k in ("atten_limit_db", "output_gain_db"):
        d.pop(k)
    return repr(sorted(d.items()))


def enhance(job: Job, p: Params, profile: NoiseProfile | None = None
            ) -> tuple[EnhanceResult, np.ndarray, bool]:
    """Run the selected algorithm (cached per job), then the attenuation limit (Eq. 34).

    Returns (result with an explicit or effective gain, noise_mag, gain_is_explicit).
    """
    cfg = p.stft
    spec, mag = job.spectrum(cfg)
    noise_mag = estimate_noise(job, p, mag, profile)
    ok, reason = registry.availability(p.algorithm)
    if not ok:
        raise ProcessingError(f"{registry.spec(p.algorithm).name} is not available: {reason}")
    key = _cache_key(p)
    hit = job.results.get(key)
    if hit is None:
        ctx = EnhanceContext(y=job.y, sr=job.sr, cfg=cfg, spec=spec, mag=mag, noise_mag=noise_mag)
        try:
            warm(registry.get(p.algorithm), p)              # model loading is not part of the timing
            t0 = time.perf_counter()
            res = registry.get(p.algorithm).process(ctx, p)
        except (ValueError, RuntimeError, OSError) as exc:
            raise ProcessingError(f"{registry.spec(p.algorithm).name} failed: {exc}") from exc
        if not np.all(np.isfinite(res.cleaned)) or len(res.cleaned) != len(job.y):
            raise ProcessingError(f"{registry.spec(p.algorithm).name} returned an invalid signal")
        res.extra["proc_s"] = time.perf_counter() - t0     # algorithm only (not STFT / metrics)
        explicit = res.gain is not None
        if not explicit:
            res.gain = effective_gain(res.cleaned, mag, cfg)
        hit = (res, explicit)
        job.remember_result(key, hit)
    res, explicit = hit
    cleaned, gain = apply_attenuation_limit(job.y, res.cleaned, res.gain, p.atten_limit_db)
    out = EnhanceResult(cleaned=cleaned, gain=gain, noise_power=res.noise_power, extra=res.extra)
    return out, noise_mag, explicit


def run(job: Job, p: Params, profile: NoiseProfile | None = None) -> dict[str, Any]:
    t_start = time.perf_counter()
    cfg = p.stft
    spec, mag = job.spectrum(cfg)
    enh, noise_mag, explicit_gain = enhance(job, p, profile)
    cleaned_raw = enh.cleaned
    res = _Out(cleaned=output_stage(cleaned_raw, p.output_gain_db), cleaned_raw=cleaned_raw, gain=enh.gain)
    removed = job.y - res.cleaned_raw                    # what the algorithm took away (before output gain)
    t_dsp = time.perf_counter() - t_start

    # --- metrics -------------------------------------------------------------------------
    frame, hop = cfg.n_fft, cfg.hop
    e_noisy = M.frame_energy_db(job.y, frame, hop)
    e_clean = M.frame_energy_db(res.cleaned_raw, frame, hop)
    speech = M.energy_vad(e_noisy)
    gt = None
    if job.clean is not None:
        c, raw = job.clean, res.cleaned_raw
        snr_in, snr_out = M.snr_db(c, job.y), M.snr_db(c, raw)
        sdr_in, sdr_out = M.si_sdr_db(c, job.y), M.si_sdr_db(c, raw)
        q_in = job.input_quality()
        q_out = Q.intrusive(c, raw, job.sr)
        gt = {
            "pesq_in": _round(q_in["pesq"], 3), "pesq_out": _round(q_out["pesq"], 3),
            "stoi_in": _round(q_in["stoi"], 4), "stoi_out": _round(q_out["stoi"], 4),
            "estoi_in": _round(q_in["estoi"], 4), "estoi_out": _round(q_out["estoi"], 4),
            "lsd_in_db": _round(q_in["lsd_db"]), "lsd_out_db": _round(M.log_spectral_distance_db(c, raw)),
            "snr_in_db": _round(snr_in), "snr_out_db": _round(snr_out),
            "snr_improvement_db": _round(snr_out - snr_in),
            "si_sdr_in_db": _round(sdr_in), "si_sdr_out_db": _round(sdr_out),
            "si_sdr_improvement_db": _round(sdr_out - sdr_in),
            "seg_snr_in_db": _round(M.segmental_snr_db(c, job.y)),
            "seg_snr_out_db": _round(M.segmental_snr_db(c, raw)),
            "note": "computed on the output before output gain / peak protection",
        }
    vad_in, vad_out = M.vad_snr_db(e_noisy, speech), M.vad_snr_db(e_clean, speech)

    # --- viz -----------------------------------------------------------------------------
    times = frame_times(spec.shape[0], job.sr, cfg)
    mag_clean = res.gain * mag
    mag_removed = np.abs(spec - res.gain * spec)
    e_times = (np.arange(len(e_noisy)) * hop + frame / 2) / job.sr
    viz = {
        "waveform": {"original": V.envelope(job.y, job.sr), "cleaned": V.envelope(res.cleaned, job.sr),
                     "removed": V.envelope(removed, job.sr)},
        "spectrogram": {
            "t": V.axis_times(times), "f": V.axis_freqs(cfg, job.sr),
            "noisy": V.spectrogram(mag, cfg), "cleaned": V.spectrogram(mag_clean, cfg),
            "removed": V.spectrogram(mag_removed, cfg), "units": "dBFS",
        },
        "gain_mask": V.gain_mask(res.gain),
        "spectra": {
            "f": np.round(np.fft.rfftfreq(cfg.n_fft, 1 / job.sr), 2).tolist(),
            "noise_profile_db": np.round(V.mag_db(noise_mag, cfg), 1).tolist(),
            "alpha_noise_db": np.round(V.mag_db(_subtracted_level(p) * noise_mag, cfg), 1).tolist(),
            "noise_tracked_db": (None if enh.noise_power is None else
                                 np.round(V.mag_db(np.sqrt(np.pi / 4 * enh.noise_power.mean(axis=0)), cfg),
                                          1).tolist()),
            "noisy_db": np.round(V.mean_spectrum_db(mag, cfg), 1).tolist(),
            "cleaned_db": np.round(V.mean_spectrum_db(mag_clean, cfg), 1).tolist(),
        },
        "energy": V.timeline(e_times, {"noisy_db": e_noisy, "cleaned_db": e_clean,
                                       "speech": speech.astype(float)}),
    }

    # --- files ---------------------------------------------------------------------------
    job.version += 1
    if job.version == 1:
        write_wav(job.dir / "original.wav", job.y, job.sr)
    write_wav(job.dir / "cleaned.wav", res.cleaned, job.sr)
    write_wav(job.dir / "noise.wav", removed, job.sr)
    if job.version == 1 and job.clean is not None:
        write_wav(job.dir / "clean.wav", job.clean, job.sr)
    job.last = {"params": p, "noise_mag": noise_mag, "gain": res.gain, "explicit_gain": explicit_gain,
                "noise_power": enh.noise_power, "cleaned_raw": res.cleaned_raw}

    elapsed = time.perf_counter() - t_start
    m = {
        "duration_s": _round(job.duration, 3), "sr": job.sr,
        "input_rms_dbfs": _round(M.rms_dbfs(job.y)), "output_rms_dbfs": _round(M.rms_dbfs(res.cleaned)),
        "input_peak_dbfs": _round(M.peak_dbfs(job.y)), "output_peak_dbfs": _round(M.peak_dbfs(res.cleaned)),
        "noise_floor_dbfs": _round(noise_floor_dbfs(noise_mag, cfg)),
        "noise_reduction_db": _round(M.noise_reduction_db(e_noisy, e_clean, speech)),
        "speech_attenuation_db": _round(M.speech_attenuation_db(e_noisy, e_clean, speech)),
        "vad_snr_in_db": _round(vad_in), "vad_snr_out_db": _round(vad_out),
        "speech_fraction": _round(float(speech.mean()), 3),
        "dsp_ms": _round(t_dsp * 1000, 1), "processing_ms": _round(elapsed * 1000, 1),
        "real_time_factor": _round(elapsed / job.duration, 4),
        "ground_truth": gt,
    }
    job.last["metrics"] = m
    base = f"/api/audio/{job.id}"
    audio = {k: f"{base}/{k}.wav?v={job.version}" for k in ("original", "cleaned", "noise")}
    if job.clean is not None:
        audio["clean"] = f"{base}/clean.wav?v=1"
    return {
        "job_id": job.id, "name": job.name, "sr": job.sr, "duration": _round(job.duration, 3),
        "version": job.version, "params": p.to_dict(), "metrics": m, "viz": viz,
        "algorithm": {"id": p.algorithm, "name": registry.spec(p.algorithm).name,
                      "family": registry.spec(p.algorithm).family, "explicit_gain": explicit_gain},
        "demo": job.demo,
        "audio": audio,
    }


def _subtracted_level(p: Params) -> float:
    """Magnitude-domain equivalent of what is subtracted, for plotting next to |Y|.

    Spectral subtraction removes α·|N̂|^p, i.e. a magnitude of α^(1/p)·|N̂| (Eq. 7).
    The Wiener gain (Eq. 9) reaches zero where |Y|² = α·|N̂|², i.e. |Y| = √α·|N̂|.
    Other algorithms do not subtract a scaled estimate, so |N̂| itself is shown.
    """
    if p.algorithm != "spectral_subtraction":
        return 1.0
    return p.alpha ** 0.5 if p.method == "wiener" else p.alpha ** (1.0 / p.exponent)


def _inspector_floor(p: Params, y: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray | None, str]:
    """The lowest |S̃| the algorithm can output in this frame, and its label."""
    lim = attenuation_limit(p.atten_limit_db)
    if p.algorithm == "spectral_subtraction":
        if p.method == "wiener":
            return p.beta * y, "floor β·|Y|"                    # Eq. (9): G ≥ β  ⇒  |S̃| ≥ β·|Y|
        floor = subtracted_magnitude(np.zeros((1, len(noise))), noise, 0.0, p.beta, p.exponent)[0]
        return floor, "floor β^(1/p)·|N̂|"                       # Eq. (7)
    a = p.algo_params
    if "gain_floor_db" in a or "g_min_db" in a:
        g_db = a.get("gain_floor_db", a.get("g_min_db"))
        g = max(10.0 ** (g_db / 20.0), lim)
        return g * y, f"floor G_min·|Y| ({20 * np.log10(g):.0f} dB)"
    if lim > 0:
        return lim * y, f"attenuation limit ({p.atten_limit_db:.0f} dB)"
    return None, ""


def inspect_frame(job: Job, t: float) -> dict[str, Any]:
    """STFT frame inspector: |Y|, the noise estimate and |S̃| for the frame nearest to time t (dBFS)."""
    if not job.last:
        raise ProcessingError("job has not been processed yet")
    p: Params = job.last["params"]
    cfg = p.stft
    _, mag = job.spectrum(cfg)
    times = frame_times(mag.shape[0], job.sr, cfg)
    m = int(np.argmin(np.abs(times - t)))
    y = mag[m]
    noise = job.last["noise_mag"]
    if job.last.get("noise_power") is not None:              # tracked noise (OM-LSA / IMCRA)
        noise = np.sqrt(np.pi / 4.0 * job.last["noise_power"][m])   # Eq. (22) inverted
    gain = job.last["gain"][m]
    floor, floor_label = _inspector_floor(p, y, noise)
    if p.algorithm == "spectral_subtraction":
        noise_label = ("√α·|N̂| (G = 0 below)" if p.method == "wiener"
                       else f"α^(1/p)·|N̂| subtracted (p = {p.exponent})")
    elif job.last.get("noise_power") is not None:
        noise_label = "|N̂| tracked (IMCRA), this frame"
    else:
        noise_label = "|N̂| estimate (reference)"
    return {
        "frame": m, "t": _round(float(times[m]), 4), "n_frames": int(mag.shape[0]),
        "f": np.round(np.fft.rfftfreq(cfg.n_fft, 1 / job.sr), 2).tolist(),
        "noisy_db": np.round(V.mag_db(y, cfg), 1).tolist(),
        "alpha_noise_db": np.round(V.mag_db(_subtracted_level(p) * noise, cfg), 1).tolist(),
        "floor_db": None if floor is None else np.round(V.mag_db(floor, cfg), 1).tolist(),
        "cleaned_db": np.round(V.mag_db(gain * y, cfg), 1).tolist(),
        "gain": np.round(np.minimum(gain, 10.0), 4).tolist(),
        "method": p.method, "exponent": p.exponent, "alpha": p.alpha, "beta": p.beta,
        "algorithm": p.algorithm, "explicit_gain": bool(job.last.get("explicit_gain", True)),
        "noise_label": noise_label, "floor_label": floor_label,
    }


def compare_one(job: Job, p: Params, profile: NoiseProfile | None = None, dnsmos: bool = False
                ) -> dict[str, Any]:
    """Run one algorithm for the Compare panel (§14): metrics only, no plots.

    Shares the job's result cache with run(), so selecting a compared algorithm afterwards
    is instant. Writes cleaned_<algorithm>.wav for listening.
    """
    enh, _, explicit = enhance(job, p, profile)
    raw = enh.cleaned
    frame, hop = p.stft.n_fft, p.stft.hop
    e_noisy = M.frame_energy_db(job.y, frame, hop)
    e_clean = M.frame_energy_db(raw, frame, hop)
    speech = M.energy_vad(e_noisy)
    row: dict[str, Any] = {
        "algorithm": p.algorithm, "name": registry.spec(p.algorithm).name,
        "family": registry.spec(p.algorithm).family, "explicit_gain": explicit,
        "algo_params": p.algo_params,
        "noise_reduction_db": _round(M.noise_reduction_db(e_noisy, e_clean, speech)),
        "speech_attenuation_db": _round(M.speech_attenuation_db(e_noisy, e_clean, speech)),
        "proc_ms": _round(enh.extra.get("proc_s", math.nan) * 1000, 1),
        "real_time_factor": _round(enh.extra.get("proc_s", math.nan) / job.duration, 4),
        "ground_truth": None, "dnsmos": None,
    }
    if job.clean is not None:
        c = job.clean
        q = Q.intrusive(c, raw, job.sr)
        q_in = job.input_quality()
        row["ground_truth"] = {
            "si_sdr_improvement_db": _round(M.si_sdr_db(c, raw) - M.si_sdr_db(c, job.y)),
            "snr_improvement_db": _round(M.snr_db(c, raw) - M.snr_db(c, job.y)),
            "pesq_out": _round(q["pesq"], 3), "pesq_in": _round(q_in["pesq"], 3),
            "estoi_out": _round(q["estoi"], 4), "estoi_in": _round(q_in["estoi"], 4),
            "lsd_out_db": _round(M.log_spectral_distance_db(c, raw)),
        }
    if dnsmos and Q.available()["dnsmos"]:
        if "dnsmos" not in job.quality_cache:
            job.quality_cache["dnsmos"] = Q.dnsmos(job.y, job.sr)
        d_in, d_out = job.quality_cache["dnsmos"], Q.dnsmos(raw, job.sr)
        row["dnsmos"] = {"input": d_in and {k: round(v, 3) for k, v in d_in.items()},
                         "output": d_out and {k: round(v, 3) for k, v in d_out.items()}}
    write_wav(job.dir / f"cleaned_{p.algorithm}.wav", output_stage(raw, p.output_gain_db), job.sr)
    row["audio"] = f"/api/audio/{job.id}/cleaned_{p.algorithm}.wav?v={job.version}.{len(job.results)}"
    return row


def perceptual_quality(job: Job) -> dict[str, Any]:
    """DNSMOS P.835 of the input (cached per job) and of the latest output (§14).

    Kept out of run() because it costs ~0.1 s per 9 s of audio, which is too slow for
    slider-driven reprocessing.
    """
    if not job.last:
        raise ProcessingError("job has not been processed yet")
    if not Q.available()["dnsmos"]:
        raise ProcessingError("DNSMOS is not available: install requirements-ml.txt and run "
                              "scripts/fetch_models.py")
    t0 = time.perf_counter()
    if "dnsmos" not in job.quality_cache:
        job.quality_cache["dnsmos"] = Q.dnsmos(job.y, job.sr)
    out = Q.dnsmos(job.last["cleaned_raw"], job.sr)
    r = lambda d: None if d is None else {k: round(v, 3) for k, v in d.items()}  # noqa: E731
    return {"version": job.version, "input": r(job.quality_cache["dnsmos"]), "output": r(out),
            "model": "DNSMOS P.835 (Microsoft), predicted MOS 1–5: SIG speech, BAK background, OVRL overall",
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


SWEEP_ALPHAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
SWEEP_BETAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def sweep(job: Job, p: Params, profile: NoiseProfile | None = None,
          alphas: tuple[float, ...] = SWEEP_ALPHAS, betas: tuple[float, ...] = SWEEP_BETAS) -> dict[str, Any]:
    """Ground-truth SNR / SI-SDR improvement over an α × β grid (demo jobs only).

    Every other parameter stays as in `p`. Reuses the cached STFT and noise estimate, so
    each grid point costs only one gain computation + one ISTFT (no output stage).
    """
    if job.clean is None:
        raise ProcessingError("the parameter sweep needs a clean reference — use demo mode")
    if p.algorithm != "spectral_subtraction":
        raise ProcessingError("the α × β sweep applies to spectral subtraction only")
    t0 = time.perf_counter()
    cfg = p.stft
    spec, mag = job.spectrum(cfg)
    noise_mag = estimate_noise(job, p, mag, profile)
    c = job.clean
    snr_in, sdr_in = M.snr_db(c, job.y), M.si_sdr_db(c, job.y)
    snr = np.zeros((len(betas), len(alphas)))
    sdr = np.zeros_like(snr)
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            q = validate({"alpha": float(a), "beta": float(b)}, base=p)   # also range-checks
            y = istft(compute_gain(mag, noise_mag, q.subtraction) * spec, len(job.y), cfg)
            snr[i, j] = M.snr_db(c, y) - snr_in
            sdr[i, j] = M.si_sdr_db(c, y) - sdr_in
    bi, bj = np.unravel_index(np.argmax(sdr), sdr.shape)
    return {
        "alphas": list(alphas), "betas": list(betas),
        "snr_improvement_db": np.round(snr, 2).tolist(),
        "si_sdr_improvement_db": np.round(sdr, 2).tolist(),
        "best": {"alpha": alphas[bj], "beta": betas[bi], "si_sdr_improvement_db": round(float(sdr[bi, bj]), 2),
                 "snr_improvement_db": round(float(snr[bi, bj]), 2)},
        "current": {"alpha": p.alpha, "beta": p.beta},
        "fixed": {k: v for k, v in p.to_dict().items() if k not in ("alpha", "beta", "profile_id")},
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }
