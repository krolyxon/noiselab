"""JS ⇄ Python parity for the live-preview core (app/static/js/dsp/ss_core.js).

The AudioWorklet runs ss_core.js. Here the same module runs under Node on a fixed test
vector and must match the Python pipeline (shifted by its one-hop latency).
Skipped when Node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.dsp import synth
from app.dsp.noise_estimation import estimate_initial
from app.dsp.spectral_subtraction import SubtractionParams, denoise
from app.dsp.stft import StftConfig, stft

CORE = (Path(__file__).resolve().parents[1] / "app" / "static" / "js" / "dsp" / "ss_core.js").as_uri()
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")
SR = 16000


def _run_js(tmp_path, x, *, N, alpha, beta, exponent, live_learning, noise_seconds=0.5):
    inp, out = tmp_path / "x.f64", tmp_path / "y.f64"
    x.astype("<f8").tofile(inp)
    script = f"""
      import {{ SpectralSubtractor, computeNoiseMag }} from "{CORE}";
      import {{ readFileSync, writeFileSync }} from "node:fs";
      const b = readFileSync("{inp}");
      const x = new Float64Array(b.buffer, b.byteOffset, b.length / 8);
      const core = new SpectralSubtractor({{ N: {N}, alpha: {alpha}, beta: {beta}, exponent: {exponent} }});
      if ({json.dumps(live_learning)}) core.startLearning({noise_seconds}, {SR});
      else core.setNoise(computeNoiseMag(x, {N}, {noise_seconds}, {SR}));
      const H = {N} / 2, steps = Math.ceil(x.length / H) + 1, y = new Float64Array(steps * H);
      const hop = new Float64Array(H);
      for (let k = 0; k < steps; k++) {{
        hop.fill(0);
        hop.set(x.subarray(k * H, Math.min((k + 1) * H, x.length)));
        y.set(core.processHop(hop), k * H);
      }}
      writeFileSync("{out}", Buffer.from(y.buffer));
      console.log(JSON.stringify({{ learning: core.learning }}));
    """
    r = subprocess.run([NODE, "--input-type=module"], input=script, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return np.fromfile(out, dtype="<f8"), json.loads(r.stdout)


def _python(x, *, N, alpha, beta, exponent):
    cfg = StftConfig(N, 0.5, "hann")
    spec = stft(x, cfg)
    noise = estimate_initial(np.abs(spec), len(x), SR, cfg, 0.5)
    return denoise(x, noise, cfg, SubtractionParams(alpha, beta, exponent), spec=spec).cleaned_raw


@pytest.mark.parametrize("N,alpha,beta,exponent", [(512, 2.0, 0.02, 1), (512, 2.0, 0.02, 2), (1024, 4.0, 0.01, 2),
                                                   (256, 3.0, 0.05, 1)])
def test_core_matches_python_offline_noise(tmp_path, N, alpha, beta, exponent):
    x = synth.make_demo("pink", 5.0, SR, duration=3.0, seed=7)["noisy"]
    y_js, _ = _run_js(tmp_path, x, N=N, alpha=alpha, beta=beta, exponent=exponent, live_learning=False)
    y_py = _python(x, N=N, alpha=alpha, beta=beta, exponent=exponent)
    H = N // 2
    assert np.max(np.abs(y_js[H:H + len(x)] - y_py)) < 1e-9


def test_core_live_learning_matches_offline_estimate(tmp_path):
    """Learning |N̂| from the first 0.5 s of the stream gives the same output as the offline
    estimate, from the moment learning finishes (before that the core passes audio through)."""
    x = synth.make_demo("fan", 5.0, SR, duration=3.0, seed=8)["noisy"]
    N, H = 512, 256
    y_js, info = _run_js(tmp_path, x, N=N, alpha=2.0, beta=0.02, exponent=1, live_learning=True)
    assert info["learning"] is False
    y_py = _python(x, N=N, alpha=2.0, beta=0.02, exponent=1)
    sim = y_js[H:H + len(x)]
    # The last learning frame is the last one (start a multiple of H) that ends by 0.5 s.
    # It already gets the gain, so: before its start → pass-through; from start + H → identical.
    last_start = ((int(0.5 * SR) - N) // H) * H                  # 7424 for N = 512 at 16 kHz
    assert np.max(np.abs(sim[last_start + H:] - y_py[last_start + H:])) < 1e-9
    # while learning, the core is a pass-through (Hann at 50 % sums to 1): output = input
    assert np.max(np.abs(sim[:last_start] - x[:last_start])) < 1e-9
