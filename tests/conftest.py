import io

import numpy as np
import pytest
import soundfile as sf

from app import create_app
from app.dsp import synth


@pytest.fixture()
def app(tmp_path):
    return create_app({"TESTING": True, "JOB_ROOT": tmp_path / "jobs"})


@pytest.fixture()
def client(app):
    return app.test_client()


def wav_bytes(y: np.ndarray, sr: int = 16000, fmt: str = "WAV", subtype: str = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


@pytest.fixture(scope="session")
def demo_wav() -> bytes:
    d = synth.make_demo("white", 5.0, 16000, duration=4.0, seed=0)
    return wav_bytes(d["noisy"])
