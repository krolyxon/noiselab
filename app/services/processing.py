"""Run the algorithm on a job and assemble metrics + plot data. No Flask imports."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from ..dsp import metrics as M
from ..dsp import viz_data as V
from ..dsp.noise_estimation import estimate_initial, estimate_minstats, estimate_profile, estimate_region
from ..dsp.spectral_subtraction import denoise, subtracted_magnitude
from ..dsp.stft import StftConfig, analysis_window, frame_times
from ..params import Params
from .audio_io import write_wav
from .jobs import Job, NoiseProfile


class ProcessingError(ValueError):
    pass


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


def run(job: Job, p: Params, profile: NoiseProfile | None = None) -> dict[str, Any]:
    t_start = time.perf_counter()
    cfg = p.stft
    spec, mag = job.spectrum(cfg)
    noise_mag = estimate_noise(job, p, mag, profile)
    res = denoise(job.y, noise_mag, cfg, p.subtraction, p.output_gain_db, spec=spec)
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
        gt = {
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
            "alpha_noise_db": np.round(V.mag_db(p.alpha ** (1 / p.exponent) * noise_mag, cfg), 1).tolist(),
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
    job.last = {"params": p, "noise_mag": noise_mag, "gain": res.gain}

    elapsed = time.perf_counter() - t_start
    m = {
        "duration_s": _round(job.duration, 3), "sr": job.sr,
        "input_rms_dbfs": _round(M.rms_dbfs(job.y)), "output_rms_dbfs": _round(M.rms_dbfs(res.cleaned)),
        "input_peak_dbfs": _round(M.peak_dbfs(job.y)), "output_peak_dbfs": _round(M.peak_dbfs(res.cleaned)),
        "noise_floor_dbfs": _round(noise_floor_dbfs(noise_mag, cfg)),
        "noise_reduction_db": _round(M.noise_reduction_db(e_noisy, e_clean, speech)),
        "vad_snr_in_db": _round(vad_in), "vad_snr_out_db": _round(vad_out),
        "speech_fraction": _round(float(speech.mean()), 3),
        "dsp_ms": _round(t_dsp * 1000, 1), "processing_ms": _round(elapsed * 1000, 1),
        "real_time_factor": _round(elapsed / job.duration, 4),
        "ground_truth": gt,
    }
    base = f"/api/audio/{job.id}"
    return {
        "job_id": job.id, "name": job.name, "sr": job.sr, "duration": _round(job.duration, 3),
        "version": job.version, "params": p.to_dict(), "metrics": m, "viz": viz,
        "demo": job.demo,
        "audio": {k: f"{base}/{k}.wav?v={job.version}" for k in ("original", "cleaned", "noise")},
    }


def inspect_frame(job: Job, t: float) -> dict[str, Any]:
    """STFT frame inspector: |Y|, α|N̂| and |S̃| for the frame nearest to time t (all in dBFS)."""
    if not job.last:
        raise ProcessingError("job has not been processed yet")
    p: Params = job.last["params"]
    cfg = p.stft
    _, mag = job.spectrum(cfg)
    times = frame_times(mag.shape[0], job.sr, cfg)
    m = int(np.argmin(np.abs(times - t)))
    y = mag[m]
    noise = job.last["noise_mag"]
    gain = job.last["gain"][m]
    floor = subtracted_magnitude(np.zeros((1, len(noise))), noise, 0.0, p.beta, p.exponent)[0]
    return {
        "frame": m, "t": _round(float(times[m]), 4), "n_frames": int(mag.shape[0]),
        "f": np.round(np.fft.rfftfreq(cfg.n_fft, 1 / job.sr), 2).tolist(),
        "noisy_db": np.round(V.mag_db(y, cfg), 1).tolist(),
        "alpha_noise_db": np.round(V.mag_db(p.alpha ** (1 / p.exponent) * noise, cfg), 1).tolist(),
        "floor_db": np.round(V.mag_db(floor, cfg), 1).tolist(),
        "cleaned_db": np.round(V.mag_db(gain * y, cfg), 1).tolist(),
        "gain": np.round(np.minimum(gain, 10.0), 4).tolist(),
        "method": p.method, "exponent": p.exponent, "alpha": p.alpha, "beta": p.beta,
    }
