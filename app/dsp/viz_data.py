"""Server-side downsampling of signals and spectra for the browser plots.

Large matrices are sent as base64 int16 holding round(value × 10), i.e. dB values
rounded to 0.1 dB. That is the same precision as JSON with one decimal, but much
faster to encode and decode (an Int16Array in JS), which keeps reprocessing quick.
"""

from __future__ import annotations

import base64

import numpy as np

from .stft import StftConfig, analysis_window

DB_FLOOR = -120.0
_EPS = 1e-12


def _pool(x: np.ndarray, n_out: int, axis: int, how: str) -> np.ndarray:
    """Reduce `axis` to at most n_out by max/mean pooling over equal groups."""
    n = x.shape[axis]
    if n <= n_out:
        return x
    edges = np.linspace(0, n, n_out + 1).astype(int)
    fn = np.maximum.reduceat if how == "max" else np.add.reduceat
    out = fn(x, edges[:-1], axis=axis)
    if how == "mean":
        shape = [1] * x.ndim
        shape[axis] = n_out
        out = out / np.diff(edges).reshape(shape)
    return out


def encode_matrix(z: np.ndarray, scale: float = 10.0) -> dict:
    q = np.clip(np.round(z * scale), -32768, 32767).astype("<i2")
    return {"shape": list(q.shape), "scale": scale, "b64": base64.b64encode(q.tobytes()).decode("ascii")}


def envelope(x: np.ndarray, sr: int, max_points: int = 2000) -> dict:
    """Min/max envelope with ≤ max_points buckets (preserves peaks, unlike decimation)."""
    n = len(x)
    k = min(max_points, n)
    edges = np.linspace(0, n, k + 1).astype(int)
    lo = np.minimum.reduceat(x, edges[:-1])
    hi = np.maximum.reduceat(x, edges[:-1])
    t = (edges[:-1] + np.diff(edges) / 2) / sr
    return {"t": np.round(t, 4).tolist(), "min": np.round(lo, 4).tolist(), "max": np.round(hi, 4).tolist()}


def magnitude_ref(cfg: StftConfig) -> float:
    """|Y| of a full-scale sine at a bin centre is Σw/2, so dividing by this gives dBFS."""
    return float(np.sum(analysis_window(cfg)) / 2.0)


def mag_db(mag: np.ndarray, cfg: StftConfig) -> np.ndarray:
    return np.maximum(20.0 * np.log10(mag / magnitude_ref(cfg) + _EPS), DB_FLOOR)


def spectrogram(mag: np.ndarray, cfg: StftConfig, max_frames: int = 400, max_bins: int = 257) -> dict:
    """dB spectrogram, frequency-major (z[k][m]) as Plotly's heatmap expects."""
    pooled = _pool(_pool(mag, max_frames, 0, "max"), max_bins, 1, "max")
    return encode_matrix(mag_db(pooled, cfg).T)


def gain_mask(gain: np.ndarray, max_frames: int = 400, max_bins: int = 257) -> dict:
    """Linear gain G(k, m), mean-pooled, capped at 2 for display (floor-fill bins can exceed 1)."""
    pooled = _pool(_pool(np.minimum(gain, 2.0), max_frames, 0, "mean"), max_bins, 1, "mean")
    return encode_matrix(pooled.T, scale=1000.0)


def axis_times(times: np.ndarray, max_frames: int = 400) -> list[float]:
    return np.round(_pool(times[:, None], max_frames, 0, "mean")[:, 0]
                    if len(times) > max_frames else times, 4).tolist()


def axis_freqs(cfg: StftConfig, sr: int, max_bins: int = 257) -> list[float]:
    f = np.fft.rfftfreq(cfg.n_fft, 1.0 / sr)
    return np.round(_pool(f[:, None], max_bins, 0, "mean")[:, 0] if len(f) > max_bins else f, 2).tolist()


def mean_spectrum_db(mag: np.ndarray, cfg: StftConfig) -> np.ndarray:
    """Mean power over frames, in dBFS."""
    p = np.mean(mag**2, axis=0)
    return np.maximum(10.0 * np.log10(p / magnitude_ref(cfg) ** 2 + _EPS), DB_FLOOR)


def timeline(t: np.ndarray, series: dict[str, np.ndarray], max_points: int = 2000) -> dict:
    """Downsample frame-rate series (energy dB, VAD flags) to ≤ max_points by max-pooling."""
    out = {"t": np.round(_pool(t[:, None], max_points, 0, "mean")[:, 0] if len(t) > max_points else t, 4).tolist()}
    for name, v in series.items():
        v = np.asarray(v, dtype=np.float64)
        pooled = _pool(v[:, None], max_points, 0, "max")[:, 0] if len(v) > max_points else v
        out[name] = np.round(pooled, 2).tolist()
    return out
