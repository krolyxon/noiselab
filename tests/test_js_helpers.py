"""Run the browser helpers (wav.js, util.js) under Node and check them against Python.

Skipped when `node` is not installed — the web app itself never needs Node.
"""

import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.dsp import viz_data

JS = Path(__file__).resolve().parents[1] / "app" / "static" / "js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def _node(script: str) -> str:
    r = subprocess.run([NODE, "--input-type=module"], input=script, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_encode_wav_roundtrip(tmp_path):
    out = tmp_path / "js.wav"
    _node(f"""
      import {{ encodeWav }} from "{(JS / 'wav.js').as_uri()}";
      import {{ writeFileSync }} from "node:fs";
      const n = 22050, x = new Float32Array(n);
      for (let i = 0; i < n; i++) x[i] = 0.5 * Math.sin(2 * Math.PI * 440 * i / 22050);
      x[10] = 1.5; x[11] = -1.5;   // must be clipped, not wrapped
      const blob = encodeWav(x, 22050);
      writeFileSync("{out}", Buffer.from(await blob.arrayBuffer()));
    """)
    y, sr = sf.read(out)
    info = sf.info(out)
    assert sr == 22050 and info.subtype == "PCM_16" and info.channels == 1 and len(y) == 22050
    ref = 0.5 * np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)
    ref[10], ref[11] = 1.0, -1.0
    assert np.max(np.abs(y - ref)) < 2 / 32768


def test_decode_matrix_matches_server_encoding():
    z = np.round(np.random.default_rng(0).uniform(-120, 0, (5, 7)), 1)
    enc = viz_data.encode_matrix(z)
    got = json.loads(_node(f"""
      globalThis.document = undefined;
      import {{ decodeMatrix }} from "{(JS / 'util.js').as_uri()}";
      console.log(JSON.stringify(decodeMatrix({json.dumps(enc)})));
    """))
    assert np.allclose(np.array(got), z, atol=1e-9)


def test_js_envelope_matches_python():
    x = np.sin(np.linspace(0, 40, 9001)) * np.linspace(0, 1, 9001)
    got = json.loads(_node(f"""
      import {{ envelope }} from "{(JS / 'wav.js').as_uri()}";
      const x = Float64Array.from({json.dumps(x.tolist())});
      console.log(JSON.stringify(envelope(x, 8000, 500)));
    """))
    ref = viz_data.envelope(x, 8000, 500)
    assert np.allclose(got["min"], ref["min"], atol=1e-4) and np.allclose(got["max"], ref["max"], atol=1e-4)
    assert np.allclose(got["t"], ref["t"], atol=1e-3)
