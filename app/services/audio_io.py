"""Safe audio loading / saving.

* The file type is detected from the *content* (libsndfile), never from the
  client's file name or MIME type.
* Duration is checked from the header before decoding the whole file.
* Output is mono float64 at the original rate (resampled only if > MAX_SAMPLE_RATE).
"""

from __future__ import annotations

import io
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

BASE_FORMATS = ("WAV", "FLAC", "OGG")


class AudioError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def supported_formats() -> list[str]:
    """WAV/FLAC/OGG always; MP3 only if this libsndfile build has it (≥ 1.1)."""
    available = sf.available_formats()
    fmts = [f for f in BASE_FORMATS if f in available]
    if "MP3" in available:
        fmts.append("MP3")
    return fmts


def load_audio(data: bytes, *, max_duration: float, min_duration: float = 0.0,
               max_sr: int = 48000) -> tuple[np.ndarray, int]:
    """Decode bytes → (mono float64 signal, sample rate). Raises AudioError on any problem."""
    if not data:
        raise AudioError("empty file")
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            if f.format not in supported_formats():
                raise AudioError(f"unsupported format {f.format}; allowed: {', '.join(supported_formats())}",
                                 415)
            sr = int(f.samplerate)
            if sr < 4000:
                raise AudioError(f"sample rate {sr} Hz is too low (minimum 4000 Hz)")
            if f.frames > 0:  # some OGG/MP3 headers report 0 frames; checked after decode
                duration = f.frames / sr
                if duration > max_duration:
                    raise AudioError(f"clip is {duration:.1f} s; maximum is {max_duration:.0f} s", 413)
            x = f.read(dtype="float64", always_2d=True)
    except AudioError:
        raise
    except (sf.LibsndfileError, RuntimeError, TypeError, ValueError) as exc:
        raise AudioError("could not decode audio (supported: "
                         f"{', '.join(supported_formats())}): {exc}", 415) from exc

    duration = x.shape[0] / sr
    if duration > max_duration:
        raise AudioError(f"clip is {duration:.1f} s; maximum is {max_duration:.0f} s", 413)
    if duration < min_duration:
        raise AudioError(f"clip is {duration:.2f} s; minimum is {min_duration:.2f} s")

    y = x.mean(axis=1)                                   # mono downmix
    if not np.all(np.isfinite(y)):
        raise AudioError("audio contains NaN/Inf samples")
    if sr > max_sr:
        g = gcd(sr, max_sr)
        y = resample_poly(y, max_sr // g, sr // g)
        sr = max_sr
    return y, sr


def write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    """16-bit PCM WAV (clipped to [−1, 1] so the integer conversion cannot wrap)."""
    sf.write(str(path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16", format="WAV")


def wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1.0, 1.0), sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()
