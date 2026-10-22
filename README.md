# NoiseLab: Speech Noise Suppression by Spectral Subtraction

**B.Tech ENTC (5th semester) · Model-Based Design course project**

Single-channel speech noise suppression with **spectral subtraction** (Boll, 1979), including Berouti's over-subtraction factor α and spectral floor β.

> ### The Simulink model is the primary implementation; the web app is an extension.
> The assessed deliverable is the Simulink model in [`simulink/`](simulink/README.md). **NoiseLab**, the web app in this repo, re-implements *the same algorithm* in Python: same window, frame size, overlap, subtraction rule and noisy-phase reconstruction. It is for interactive exploration, live recording and comparison with Simulink. It does not replace the model.

| Part | What it is | Testing status |
|---|---|---|
| `simulink/` | Block table, model builder, run/export scripts, plain-MATLAB reference | **UNTESTED in MATLAB** (written without MATLAB access). The block *design* is verified by a Python emulation (`tests/test_simulink_chain.py`): it matches the Python pipeline to 3 × 10⁻¹⁶ |
| `app/dsp/` | Hand-written DSP in NumPy/SciPy: STFT, noise estimation, subtraction, metrics, test signals | Unit-tested (`pytest`) |
| `app/` | Flask API + browser UI (vanilla JS, Plotly, KaTeX, all vendored) | API tests + headless-browser checks |
| `scripts/compare_simulink_python.py` | Aligns and compares Simulink output with Python | Tested on emulated Simulink output |
| `docs/` | [`ALGORITHM.md`](docs/ALGORITHM.md) (math, numbered equations cited in code), [`GLOSSARY.md`](docs/GLOSSARY.md) (every term in plain words), [`VIVA.md`](docs/VIVA.md), [`REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md) | — |

Everything in `app/dsp/` is hand-written and can be explained line by line. That covers spectral subtraction and, since v2, four statistical estimators. Prebuilt denoisers (noisereduce, WebRTC NS, RNNoise, NSNet2, DeepFilterNet3, Demucs) appear **only** as optional comparison baselines in `app/enhancers/thirdparty/`. See [Comparison bench](#comparison-bench-extension) below.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
flask --app app run --debug                          # open http://127.0.0.1:5000
pytest -q                                            # ~175 tests, ~6 s (ML tests skipped)
```

Requires Python 3.10+ (developed on 3.14). Dependencies: Flask, NumPy, SciPy, soundfile, pytest, plus matplotlib, which only the comparison script uses. Plotly.js 2.35.2, KaTeX 0.16.11 and the Inter / JetBrains Mono fonts are vendored in `app/static/vendor/` with their licences. The app works offline and on lab networks that block CDNs.

### Compare with Simulink

```matlab
% MATLAB, repo root (see simulink/README.md first; the scripts are UNTESTED)
addpath simulink
run_and_export('samples/noisy.wav', 'out/cleaned_simulink.wav')
```

```bash
python scripts/compare_simulink_python.py samples/noisy.wav out/cleaned_simulink.wav
```

The script prints the alignment lag, RMS error, correlation and the SNR between the two outputs (> 40 dB means "MATCH"), and saves a comparison figure. `samples/README.md` shows how to generate `samples/noisy.wav`; no audio is committed.

---

## Using NoiseLab

Layout, desktop (≥ 1280 px): **left** = source, demo mode, parameters; **centre** = player and plots; **right** = metrics and noise spectrum. It collapses to two columns, then one on narrow screens.

1. **Load audio.** Drag and drop or pick a WAV/FLAC/OGG file (MP3 if your libsndfile supports it; see `/api/health`). Limits: 25 MB, 120 s. The browser draws the waveform immediately, then the server processes the file. Formats the server can't read but the browser can (e.g. M4A, WebM) are decoded in the browser and uploaded as 16-bit WAV.
2. **Or record.** **Record** (`R`) shows a level meter, timer and live spectrum, and the clip is processed when you stop. Leave about 1 s of silence first: the default noise estimate uses the first 0.5 s. **Capture noise profile (2 s)** records room noise only and switches to the "Noise profile clip" mode.
3. **Or run a demo.** Choose a noise type (white, pink, 50 Hz mains hum, fan/AC, babble-like, clicks) and an input SNR from −5 to 20 dB. The clean reference is known, so true **SNR, SI-SDR and segmental SNR** improvements are shown, and "Clean ref" becomes playable. You can also supply your own clean WAV.
4. **Tune.** Every slider reprocesses after 250 ms, reusing the cached STFT (about 50 ms for a 30 s clip). **Match Simulink basic** sets the exact Simulink parameters. Other presets: Faithful (light), Balanced, Aggressive, Mains hum 50 Hz, Fan/AC. Options marked *extended* (smoothing, Wiener gain, min-statistics) are not in the basic Simulink model.
5. **Listen.** The A/B player switches Original / Cleaned / Removed (/ Clean ref) on one shared playhead (`Space` play, `A` switch).
6. **Inspect.** Plots: waveform; noisy/cleaned/removed spectrograms on one colour scale; frame energy with VAD shading; gain mask G(k, m); noise estimate with mean spectra. **Click any time plot** to open the **STFT frame inspector**, which shows that frame's |Y|, the subtracted level α|N̂|, the floor and the result |S̃|. `,` and `.` step frames. Drag to zoom (all time plots stay in sync); double-click to reset.
7. **Live denoise (preview).** Hear the basic algorithm in real time on your microphone, through an AudioWorklet with ≈ N samples of algorithmic latency (10.7 ms at N = 512, 48 kHz). The Denoised/Bypass toggle gives a direct A/B. α, β and p apply instantly. **Use headphones.** The noise estimate comes from your recorded profile, or is learned from the first T s (stay quiet). The live path implements only the basic algorithm (Hann, 50 %, subtraction rule); its JS core is parity-tested against Python (< 10⁻⁹).
8. **Parameter sweep (demo mode).** A heatmap of ground-truth ΔSI-SDR or ΔSNR over 11 α × 7 β values, with every other parameter as currently set. ★ marks the best cell and an outline marks the current one. Click a cell to apply it.
9. **Export.** The ZIP contains the noisy, cleaned and removed-noise WAVs (and the clean reference in demo mode), `params.json`, `metrics.json` and every plot as a titled PNG.
10. **Method.** The header **Method** dialog gives the equations (KaTeX), what α and β do, limitations and references. `?` lists all shortcuts.

**Honest metrics.** SNR, SI-SDR and segmental SNR need the clean signal, so for uploads and recordings they show **"n/a (no clean reference)"**. They are never estimated. Reference-free indicators are shown instead: noise reduction over VAD-detected noise frames, and a rough VAD-based SNR estimate.

### Microphone recording needs `localhost` or HTTPS

Browsers allow microphone access only in a secure context:

- ✅ `http://127.0.0.1:5000` / `http://localhost:5000` on the same machine
- ❌ `http://192.168.x.x:5000` from another device. The Record buttons are disabled and the app says why. Use HTTPS (e.g. a reverse proxy with a certificate) for remote recording.

The app asks the browser to **turn off its own noise suppression, echo cancellation and auto-gain** while recording, so the browser doesn't denoise the signal first; some browsers or OS audio stacks may ignore this. Recordings pass through `MediaRecorder` (usually Opus, which is lossy). For exact comparisons with Simulink, use uploaded WAV files.

---

## Results on synthetic test mixtures

SI-SDR improvement in dB at 5 dB input SNR, mean of 5 seeds (6 s synthetic voiced signal; `app/dsp/synth.py`). Higher is better; bold = best per row. Measured with the pipeline in this repo.

| Noise (5 dB input SNR) | Defaults (p=2) | Match Simulink basic | Balanced | Aggressive | Mains hum 50 Hz | Fan / AC |
|---|---|---|---|---|---|---|
| white | +6.1 | **+11.3** | +8.6 | +10.6 | +9.0 | +11.0 |
| pink | +5.3 | +8.0 | +6.3 | +7.3 | +8.0 | **+8.7** |
| hum50 | +7.6 | +5.2 | +6.6 | +5.7 | **+12.3** | +9.7 |
| fan | +3.9 | +4.2 | +4.5 | +4.4 | **+7.1** | +7.1 |
| babble | +4.8 | **+9.7** | +7.3 | +9.3 | +7.6 | +9.1 |
| clicks | +0.7 | +2.1 | +0.8 | +1.4 | **+4.6** | +3.1 |

How to read this:
- **Magnitude subtraction (p = 1, the Simulink preset) with α = 2 is more aggressive than power subtraction (p = 2) with α = 2.** That is why it scores higher on broadband noise.
- **Long frames (N = 2048, the hum preset) resolve the 50 Hz harmonic lines**, which is why that preset wins on hum and on narrow-band noise.
- **Clicks are impulsive and non-stationary.** Spectral subtraction assumes stationary noise, so it barely helps there. This is an expected limitation (see `docs/VIVA.md`).
- SI-SDR does not measure **musical noise**. Higher α / lower β often scores well but sounds "watery". Listen with the A/B player.
- The synthetic "babble" is far more stationary than real babble. Real multi-talker babble is much harder.

---

## Comparison bench (extension)

The project guide asked which noise-suppression algorithms work best under which conditions. NoiseLab v2 runs 11 algorithms through the same pipeline and the same metrics. **The Simulink model still implements spectral subtraction only, and remains the primary deliverable.**

| Family | Algorithms | Code |
|---|---|---|
| Classical | Spectral subtraction (the Simulink algorithm) | `app/dsp/spectral_subtraction.py` |
| Statistical, hand-written | Wiener (decision-directed), MMSE-STSA, log-MMSE, OM-LSA + IMCRA noise tracking | `app/dsp/statistical.py`, Eq. 22–33 |
| DSP libraries | noisereduce (spectral gating), WebRTC NS (the browser / Google Meet suppressor) | `app/enhancers/thirdparty/` |
| Deep learning | RNNoise (Jitsi, OBS), NSNet2 (Microsoft DNS Challenge), DeepFilterNet3, Demucs | `app/enhancers/thirdparty/` |

- **In the app:**
  - The *Algorithm* menu switches methods, and every method exposes its own parameters.
  - *Max attenuation* (Eq. 34) works the same way for all methods.
  - The *Algorithm comparison* panel runs every available method on the current clip, with a sortable table, a bar chart and per-method playback.
  - Demo mode adds PESQ, ESTOI and log-spectral distance.
  - *Score with DNSMOS* gives a reference-free predicted MOS for any recording.
- **For the report:**
  - `scripts/benchmark.py` runs every method on real speech (LibriSpeech) × 11 noises (6 synthetic + 5 recorded DEMAND environments) × 6 SNRs.
  - `scripts/report_figures.py` turns the results into 11 figures and `results/summary.md` (means ± 95 % CI, per-condition winners, Wilcoxon tests).

### Install (optional, separate venv)

The neural models need PyTorch, and DeepFilterNet needs numpy < 2, so they get their own Python 3.12 environment ([uv](https://docs.astral.sh/uv/) makes this one command each). The basic app above does not need any of it.

```bash
uv venv -p 3.12 .venv-ml
VIRTUAL_ENV=.venv-ml uv pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=.venv-ml uv pip install -r requirements-ml.txt
.venv-ml/bin/python scripts/fetch_models.py       # NSNet2 + DNSMOS ONNX (SHA-256 checked), warms DFN3/Demucs weights
.venv-ml/bin/flask --app app run --debug           # same app, all 11 algorithms available
.venv-ml/bin/python -m pytest -q                   # ~195 tests incl. the ml-marked ones
```

Missing models are greyed out in the UI, with the reason, and listed in `/api/health`.

### Benchmark

```bash
.venv-ml/bin/python scripts/fetch_datasets.py      # ~870 MB download, ~50 MB kept in data/ (gitignored)
.venv-ml/bin/python scripts/benchmark.py --quick   # smoke test, ~1 min
.venv-ml/bin/python scripts/benchmark.py           # 14 520 runs → results/bench.csv (~1.5 h on 12 cores)
.venv-ml/bin/python scripts/benchmark.py --sweep   # parameter sensitivity → results/sweep.csv
.venv-ml/bin/python scripts/report_figures.py      # results/figures/*.png + results/summary.md
```

The benchmark is resumable, deterministic (seeded) and fair: same mixtures for all methods, measured model delays removed. See VIVA Q41. `benchmark.py --timing` measures real-time factors serially on an idle machine. RTFs logged during the parallel grid are inflated by contention, unevenly across models (Demucs 1.08 vs 0.14 serial), so figure 5 uses the serial pass.

### Results (20 LibriSpeech utterances × 11 noises × 6 SNRs, 14 520 runs; generated 2026-10-22)

Means over all conditions (± 95 % CI ≈ 0.2–0.3 dB for ΔSI-SDR, ≈ 0.03 for ΔPESQ). Full tables, per-noise winners and Wilcoxon tests are in `results/summary.md`.

| Algorithm | Speech clarity gain ↑<br>(ΔSI-SDR, dB) | Sound-quality improvement ↑<br>(ΔPESQ) | Intelligibility improvement ↑<br>(ΔESTOI) | Speech distortion ↓<br>(LSD, dB) | Noise removed in pauses ↑<br>(NR, dB) | CPU cost ↓<br>(RTF, 1 core) |
|---|---|---|---|---|---|---|
| SS (Simulink settings) | +5.84 | +0.40 | +0.018 | 7.23 | 24.2 | 0.0006 |
| Wiener-DD | +6.42 | +0.36 | +0.019 | 7.28 | 27.5 | 0.0020 |
| MMSE-STSA | +5.78 | +0.47 | +0.020 | 6.60 | 14.5 | 0.0043 |
| Log-MMSE | +6.25 | +0.49 | +0.021 | 6.52 | 17.9 | 0.0040 |
| OM-LSA + IMCRA | +4.65 | +0.40 | +0.030 | 6.81 | 12.4 | 0.0111 |
| noisereduce | +2.72 | +0.45 | +0.026 | 6.75 | 31.0 | 0.0032 |
| WebRTC NS | +4.64 | +0.38 | +0.007 | 7.31 | 15.1 | 0.0016 |
| RNNoise | +4.31 | +0.15 | +0.066 | 6.73 | 62.8 | 0.0232 |
| NSNet2 | +5.47 | +0.50 | +0.066 | 6.13 | 40.6 | 0.0083 |
| DeepFilterNet3 | +8.45 | +0.75 | +0.098 | 6.41 | 47.2 | 0.0185 |
| Demucs (dns64) | **+10.05** | +0.64 | **+0.111** | **5.84** | 48.0 | 0.1433 |

Every term in plain words: [`docs/GLOSSARY.md`](docs/GLOSSARY.md), the **Glossary** button in the app, and `results/figures/fig0_glossary.png` for the report. Each figure starts with a "How to read" line.

What the numbers show (details and the remaining checks for the report: `docs/REPORT_OUTLINE.md` §7A):
- **ΔSI-SDR:** Demucs is best on 10 of 11 noise types, and DeepFilterNet3 on office noise. All but the car case are significant (paired Wilcoxon, p < 0.05; car p = 0.24).
- **ΔPESQ:** DeepFilterNet3 is best overall (+0.75). Hand-written statistical methods win the PESQ cells for 50 Hz hum at 10–20 dB and for office noise at 5–20 dB.
- **High SNR:** at 20 dB input SNR the best ΔSI-SDR is log-MMSE (+3.80), ahead of every neural model, and the neural models' ΔSI-SDR falls fastest as the SNR rises (fig. 1).
- **Simulink SS vs the statistical estimators:** the statistical estimators match SS on ΔSI-SDR (Wiener-DD +6.4, log-MMSE +6.3 vs +5.8). Log-MMSE reaches higher ΔPESQ (+0.49 vs +0.40) and lower spectral distortion (LSD 6.5 vs 7.2 dB) at about 7× SS's cost, still 0.4 % of one core.
- **Cost:** DeepFilterNet3 gives the largest ΔPESQ at 1.9 % of one CPU core. Demucs costs 8× more for a slightly *lower* ΔPESQ (fig. 5).
- **Worse than doing nothing:** noisereduce made SI-SDR worse on average for fan (−0.2 dB) and clicks (−2.1 dB).
- **RNNoise:** it removes the most noise in pauses (62.8 dB) but also attenuates speech frames the most (12.6 dB), and has the lowest ΔPESQ (fig. 4).
- **Parameter sweeps (5 dB):** SS peaks at α ≈ 1.5–2, which supports the Simulink default α = 2. Log-MMSE peaks at α_dd = 0.98. A DeepFilterNet3/Demucs attenuation limit below 18 dB costs > 1 dB SI-SDR.
- **DNSMOS vs PESQ:** DNSMOS OVRL agrees with PESQ with Spearman ρ = 0.79 over all outputs.
- **Speech material:** LibriSpeech read speech. Results may differ for conversational speech, other languages and microphones.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Versions, supported formats (MP3 detected at startup), limits |
| GET | `/api/presets` | Presets + parameter schema (ranges, defaults) |
| POST | `/api/process` | multipart `audio` + `params` (JSON) → `job_id`, `sr`, `duration`, `metrics`, `viz`, `audio` URLs |
| POST | `/api/reprocess/<job_id>` | `{"params": {...}}`; reuses the cached STFT |
| POST | `/api/noise-profile` | multipart `audio` (noise only) → `profile_id` for `noise_mode=profile` |
| POST | `/api/demo` | `{noise_type, snr_db, duration, seed, params}` (or multipart with a `clean` file) → as process + ground truth |
| GET | `/api/frame/<job_id>?t=` | STFT frame inspector data |
| POST | `/api/sweep/<job_id>` | α × β grid of ground-truth improvements (demo jobs only) |
| GET | `/api/noise-profile/<id>/spectrum?sr=&n_fft=` | Profile \|N̂\| for the live preview |
| GET | `/api/audio/<job_id>/<original\|cleaned\|noise\|clean>.wav` | `noise` = removed component y − ŝ; `clean` = demo only |
| GET/POST | `/api/export/<job_id>` | ZIP (POST adds plot PNGs) |
| GET | `/api/algorithms` | Every algorithm: family, description, reference, parameter schema, availability (+ reason) |
| POST | `/api/compare/<job_id>/<algorithm>` | `{"params", "dnsmos"}` → metrics row + `cleaned_<algorithm>.wav` URL (Compare panel) |
| POST | `/api/quality/<job_id>` | DNSMOS P.835 of the input and the latest output (slow, so on demand) |

Parameters are validated against the table in `app/params.py`. Out-of-range values are **rejected with a 400**, not silently clamped. Uploads are identified by content, not file name or MIME type. Jobs live in memory plus a per-job temp folder with random IDs, and are deleted after 30 minutes idle.

---

## Repository layout

```
app/
  __init__.py            create_app()
  config.py              limits, paths
  params.py              parameter schema, validation, presets
  routes.py              page + /api/*
  dsp/                   pure NumPy/SciPy, no Flask: stft, noise_estimation, spectral_subtraction,
                         statistical (MMSE family, OM-LSA/IMCRA), metrics, synth, viz_data
  enhancers/             algorithm registry (§14): base interface, classical adapters, runner,
                         quality (PESQ/STOI/DNSMOS), model_files; thirdparty/ = optional baselines
  services/              audio_io (safe load/save), jobs (LRU+TTL cache), processing (run + metrics + viz)
  templates/index.html   the UI
  static/{css,js,vendor} vanilla ES modules, no build step
simulink/                model builder, run/export, helpers, reference loop, README (block table)
scripts/                compare_simulink_python.py, fetch_models.py, fetch_datasets.py, benchmark.py,
                         report_figures.py
requirements-ml.txt      optional comparison-bench environment (Python 3.12)
tests/                   pytest: DSP, params, API, Simulink-chain emulation, JS helpers (via Node, optional)
docs/                    ALGORITHM.md, VIVA.md, REPORT_OUTLINE.md
samples/                 no audio committed; see samples/README.md
```

## Development notes

- `pytest -q` runs everything. `tests/test_js_helpers.py` (browser WAV encoder/decoders) and `tests/test_js_parity.py` (live-preview DSP core vs Python) run under Node, and are skipped if Node isn't installed.
- DSP functions are pure and typed, and their comments cite equation numbers in `docs/ALGORITHM.md`.
- The UI was checked in headless Chrome at 1366×768, 1920×1080 and 390 px wide (upload, demo, sliders, player, recording and live preview with a fake microphone, sweep, export, themes). The AudioWorklet was also rendered offline and compared with Python (93.8 dB SNR between outputs). Real microphone input and audible playback were not testable headlessly.
- Minimum statistics (`minstats`) is bias-compensated (Eq. 6). A raw 10th percentile underestimates the noise by 8.7 dB, and correcting it raised the mean demo SI-SDR gain from +1.5 to +4.0 dB.

## References

1. S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," *IEEE Trans. ASSP*, 27(2), 113–120, 1979. [doi:10.1109/TASSP.1979.1163209](https://doi.org/10.1109/TASSP.1979.1163209)
2. M. Berouti, R. Schwartz, J. Makhoul, "Enhancement of speech corrupted by acoustic noise," *Proc. ICASSP*, 208–211, 1979. [doi:10.1109/ICASSP.1979.1170788](https://doi.org/10.1109/ICASSP.1979.1170788)
3. J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR — half-baked or well done?," *Proc. ICASSP*, 2019.
4. R. Martin, "Noise power spectral density estimation based on optimal smoothing and minimum statistics," *IEEE Trans. Speech Audio Process.*, 9(5), 2001 (background for the simplified `minstats` mode).
