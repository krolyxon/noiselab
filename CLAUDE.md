# CLAUDE.md — NoiseLab: Spectral Subtraction in Simulink + Python/Flask

Project guide for Claude Code. Read this fully before writing any code.

## 1. What this project is

A B.Tech ENTC (5th sem) **Model-Based Design** course project: single-channel speech noise suppression using **spectral subtraction** (Boll, 1979).

It has two deliverables:

1. **Simulink model (the core, assessed deliverable).** A basic, clearly explainable spectral subtraction chain built from Simulink blocks.
2. **NoiseLab web app (extension).** A Flask web app with a polished, scientific-instrument-style UI. It implements the *same algorithm* in Python so results can be compared with Simulink.

### Non-negotiable principles

- **The Simulink model is the project.** The web app is an extension and must be described as one in every doc and in the UI "About" panel. Never present the web app as a replacement for the Simulink work.
- **Algorithm parity.** The Python DSP must implement the same steps as the Simulink model (same window, frame size, overlap, subtraction rule, phase handling). A `scripts/compare_simulink_python.py` script must demonstrate this on the same input file.
- **The core algorithm stays hand-written.** Everything in `app/dsp/` (spectral subtraction and the statistical estimators of §14) is written by hand with NumPy/SciPy, so the author can explain every line in a viva. Prebuilt denoisers (`noisereduce`, WebRTC NS, RNNoise, NSNet2, DeepFilterNet, Demucs) are allowed **only** as optional comparison baselines in `app/enhancers/thirdparty/` (§14). They must never be imported by `app/dsp/` or presented as the project's own algorithm.
- **Honesty about what was tested.** Claude cannot run MATLAB. Any `.m` file or `.slx` guidance must be labelled "untested — run and fix in MATLAB". Never claim the Simulink model works.
- **Explainability over cleverness.** Prefer readable code with comments linking each step to an equation in `docs/ALGORITHM.md`.

## 2. Tech stack

- Python 3.10+, Flask 3, NumPy, SciPy, soundfile, pytest.
- Frontend: vanilla JS (ES modules), **no build step**, no React/Node toolchain.
- Plotting: Plotly.js (spectrograms, spectra, waveforms). Equations: KaTeX.
- Audio capture: browser MediaRecorder / Web Audio API. Decode in the browser with `AudioContext.decodeAudioData`, then upload **16-bit PCM WAV** so the server never needs ffmpeg.
- **Vendor all JS/CSS/fonts into `app/static/vendor/`** (Plotly, KaTeX, fonts). College labs often block CDNs. CDN links may exist only as a commented fallback.
- Mic access requires `localhost` or HTTPS. Document this in README.
- **Optional comparison-bench stack (§14):** a separate Python 3.12 venv `.venv-ml` from `requirements-ml.txt` (PyTorch CPU, onnxruntime, the third-party enhancers, `pesq`, `pystoi`). The base app must keep working, and its tests passing, without it.

## 3. Repository layout

```
noiselab/
├─ CLAUDE.md
├─ README.md
├─ requirements.txt
├─ app/
│  ├─ __init__.py              # create_app()
│  ├─ config.py                # limits, paths, defaults
│  ├─ routes.py                # page + /api/*
│  ├─ dsp/
│  │  ├─ stft.py               # framing, windowing, STFT / ISTFT (overlap-add)
│  │  ├─ noise_estimation.py   # initial-segment, region, profile, min-statistics
│  │  ├─ spectral_subtraction.py  # core algorithm + Wiener variant
│  │  ├─ metrics.py            # RMS, SNR, SI-SDR, segmental SNR, noise reduction
│  │  ├─ synth.py              # test signals + noise generators + mixer
│  │  └─ viz_data.py           # downsampling for plots
│  ├─ services/
│  │  ├─ audio_io.py           # safe load / save / resample / mono
│  │  └─ jobs.py               # in-memory LRU cache of STFTs per job
│  ├─ templates/index.html
│  └─ static/{css,js,vendor}/
├─ simulink/
│  ├─ README.md                # how to build/run the model, block table
│  ├─ build_spectral_subtraction_model.m   # programmatic model builder (UNTESTED)
│  └─ run_and_export.m         # runs model, writes cleaned WAV for comparison
├─ scripts/compare_simulink_python.py
├─ tests/
├─ docs/
│  ├─ ALGORITHM.md             # math, with equation numbers referenced in code
│  ├─ VIVA.md                  # likely examiner questions + short answers
│  └─ REPORT_OUTLINE.md
└─ samples/                    # only CC0 / self-recorded audio; note licences
```

## 4. Algorithm specification (single source of truth)

Let `y[n]` be the noisy mono signal. Steps (both Simulink and Python):

1. **Frame + window:** frame length `N` (default 512), hop `H = N·(1 − overlap)`, default overlap 50% (Python allows 75%). Window `w` = Hann (options: Hamming, sqrt-Hann).
2. **FFT:** `Y_k(m) = FFT{ w · y_m }`, magnitude `|Y|`, phase `∠Y`.
3. **Noise estimate** `|N̂_k|` (per bin), computed from noise-only frames (see modes below).
4. **Subtract** (Berouti-style over-subtraction with spectral floor), with exponent `p` (1 = magnitude, 2 = power):

   `|S̃|^p = max( |Y|^p − α·|N̂|^p ,  β·|N̂|^p )`

   Equivalent gain form: `G = (|S̃|/|Y|)`, so `Ŝ = G · Y`.
5. **Optional refinements (Python only, off by default; must be switchable and labelled "extended"):**
   - Gain smoothing over time (`gain_smooth`, 0–0.95) and over frequency (`freq_smooth_bins`, 0–9) to reduce musical noise.
   - Wiener-style gain: `G = max(1 − α·(|N̂|²/|Y|²), β)` as an alternative method.
6. **Reconstruct:** use the **noisy phase**: `Ŝ = |S̃| · e^{j∠Y}`; IFFT; overlap-add with proper window normalisation.
7. **Output:** avoid clipping (soft-limit or scale only if peak > 0.99); optional output gain in dB.

### Noise estimation modes

| Mode | Description |
|---|---|
| `initial` (default; **the only mode in the basic Simulink model**) | Mean magnitude of the first `T` seconds assumed noise-only (default 0.5 s) |
| `region` | User drags a region on the waveform; mean magnitude over it |
| `profile` | A separately recorded noise-only clip ("capture noise profile" button) |
| `minstats` | Per-bin low percentile (default 10th) of power across frames; no silence needed (advanced) |

### Parameter table (defaults and UI ranges)

| Param | Default | Range | Notes |
|---|---|---|---|
| `n_fft` | 512 | 256, 512, 1024, 2048 | Tooltip: time vs frequency resolution trade-off |
| `overlap` | 0.5 | 0.5, 0.75 | |
| `window` | hann | hann, hamming, sqrt_hann | |
| `noise_mode` | initial | see above | |
| `noise_seconds` | 0.5 | 0.1–3.0 | for `initial` |
| `alpha` | 2.0 | 1.0–6.0 | over-subtraction factor |
| `beta` | 0.02 | 0.0–0.2 | spectral floor |
| `exponent` | 2 | 1, 2 | magnitude vs power subtraction |
| `method` | spectral_subtraction | spectral_subtraction, wiener | |
| `gain_smooth` | 0.0 | 0–0.95 | extended |
| `freq_smooth_bins` | 0 | 0–9 | extended |
| `output_gain_db` | 0 | −12–+12 | |

Provide named **presets** (`/api/presets`): "Faithful (light)", "Balanced", "Aggressive", "Mains hum 50 Hz", "Fan / AC noise", "Match Simulink basic". The last preset must exactly reproduce the Simulink model's settings.

## 5. Backend requirements

### Limits and safety
- Max upload 25 MB, max duration 120 s (reject longer with a clear error).
- Allowed: WAV, FLAC, OGG (MP3 only if the installed `soundfile` supports it; detect at startup and report in `/api/health`).
- Use `werkzeug.utils.secure_filename`; never trust client file names or MIME types; store jobs under a temp dir with random IDs; clean up after TTL (30 min).
- Convert to mono float32; keep the original sample rate (resample only if > 48 kHz, to 48 kHz). No `pickle`, no `eval`.
- Validate every parameter against the table above (server-side clamp or 400 error).

### Endpoints
- `GET /` — the app.
- `GET /api/health` — versions, supported formats.
- `GET /api/presets`
- `POST /api/process` — multipart: `audio` file + `params` (JSON string). Returns JSON: `job_id`, `sr`, `duration`, `metrics`, `viz`, `audio` URLs.
- `POST /api/reprocess/<job_id>` — new params, reuses cached STFT so slider tweaks are fast (<300 ms for a 30 s clip).
- `POST /api/noise-profile` — upload a noise-only clip, returns `profile_id` usable in `noise_mode=profile`.
- `POST /api/demo` — generate a synthetic test case (see §6) and process it, including ground-truth metrics.
- `GET /api/audio/<job_id>/<original|cleaned|noise>.wav` — `noise` is the removed component (`y − ŝ`).
- `GET /api/export/<job_id>` — ZIP: cleaned WAV, removed-noise WAV, `params.json`, `metrics.json`, PNGs of plots (optional).

### Metrics (return all that apply)
- Input/output RMS (dBFS), peak, duration, processing time, real-time factor.
- Estimated noise floor (dBFS) and **noise reduction (dB)** = mean attenuation over noise frames.
- Simple energy-based VAD → speech-frame/noise-frame SNR estimate, before and after.
- **When ground truth exists (demo mode):** true SNR in/out, SNR improvement, **SI-SDR**, segmental SNR. Never fake these for uploaded audio; show "n/a (no clean reference)".

### Viz payload (server-side downsampled)
- Waveforms: min/max envelope, ≤ 2000 points.
- Spectrograms (noisy, cleaned, removed): dB matrices capped at ~400 time × 257 freq bins, rounded to 1 decimal.
- Noise profile spectrum, mean spectra (noisy vs cleaned), gain mask `G(k, m)`, per-frame energy with VAD labels.

## 6. Synthetic test signals (`dsp/synth.py`)

- Clean source: a harmonic "voiced-like" signal with a moving pitch + formant-like filtering + syllable envelope (so the repo has no licence issues), plus support for user-supplied clean WAV.
- Noise generators: white, pink, **50 Hz mains hum + harmonics** (India), fan/AC (low-passed noise), babble-ish (summed band-limited noise bursts), clicks.
- `mix(clean, noise, snr_db)` with exact SNR scaling. Deterministic via a seed.

## 7. Frontend / UI requirements

**Goal: looks and feels like a lab instrument made by someone who knows signal processing.** Not a generic upload form.

### Design system
- Dark theme default (optional light theme toggle). Tokens in CSS variables:
  - bg `#0b0f14`, panel `#11171e`, border `#1f2a36`, text `#d7e0ea`, muted `#7d8b99`
  - Semantic colours, used consistently everywhere: **noisy = amber `#ffb454`**, **cleaned = cyan `#4cc9f0`**, **estimated noise = magenta `#e56bd5`**, OK/positive = green `#6ee7a8`.
- Fonts: Inter (UI) + JetBrains Mono (numbers, readouts). Vendored.
- Spectrograms use a perceptually uniform colormap (inferno or viridis), colour bar labelled in dB, axes labelled with units (Hz, s, dBFS).
- Dense but organised: 3-column layout on desktop (≥1280 px): left = controls, centre = plots, right = metrics + noise profile. Must remain usable at 1366×768 and degrade to a single column on mobile.

### Features (in priority order)
1. **Upload** (drag & drop + file picker) with instant waveform preview.
2. **Live record**: Record / Stop button, input-level meter, elapsed timer, and a live spectrum/waveform view from an `AnalyserNode` while recording. A "Capture noise profile (2 s)" button records room noise only. After stopping, the clip is processed automatically.
3. **Parameter panel**: sliders with numeric readouts, preset dropdown, reset, "Match Simulink basic" button. Changing a slider calls `/api/reprocess` (debounced 250 ms) and updates all plots.
4. **A/B player**: original / cleaned / removed-noise toggle with one shared playhead and keyboard shortcut; playhead drawn across the waveform and spectrograms.
5. **Plots** (Plotly, synchronised x-axis zoom between waveform and spectrograms):
   - Waveform (noisy vs cleaned overlay)
   - Spectrogram: noisy | cleaned | removed (side by side, shared colour scale)
   - Gain-mask heatmap `G(k, m)`
   - Noise-profile spectrum with the mean noisy and cleaned spectra overlaid
   - Frame energy timeline with VAD shading
   - **STFT frame inspector**: click a time on the spectrogram → show that frame's `|Y|`, `α|N̂|`, and `|S̃|` as three curves (this is the "science people" feature; it visually explains the algorithm)
6. **Metrics cards**: monospace numbers, units, delta arrows.
7. **Demo mode**: pick noise type + SNR (−5 to 20 dB) → generates a synthetic case with ground-truth metrics and shows SI-SDR / SNR improvement.
8. **Parameter sweep** (stretch): heatmap of SNR-improvement vs `alpha × beta` for demo mode.
9. **Export**: download cleaned WAV, removed-noise WAV, params/metrics JSON, plot PNGs.
10. **About / Method** panel: KaTeX equations from §4, link to Boll (1979), and the sentence: "This web app is an extension of the Simulink model; the Simulink model is the primary implementation."
11. **Stretch — live denoise preview:** an `AudioWorklet` port of the basic algorithm (`initial`/`profile` noise modes only) for real-time listening. If done, add a parity test comparing JS output to Python on a fixed vector. Skip if time is short.

### Frontend rules
- Keyboard shortcuts: Space play/pause, `A` cycle original/cleaned/noise, `R` record, `?` shortcut help.
- Loading states, clear error toasts, empty states with guidance. No layout shift when plots update.
- Don't re-render all plots on playhead movement; only update shapes.
- All numbers formatted with fixed decimals and units.

## 8. Simulink deliverable (`simulink/`)

### Basic model (keep it SIMPLE — the author must explain every block)

| Stage | Blocks |
|---|---|
| Input | **From Multimedia File** (file path set via a MATLAB variable, mono) |
| Framing | **Buffer** (frame `N`, overlap `N/2`) |
| Windowing | **Window Function** (Hann) or Product with a Constant window vector |
| Transform | **FFT** |
| Mag/Phase | **Complex to Magnitude-Angle** |
| Noise estimate | Constant block `noiseMag` (vector, length `N`) computed in the model's **InitFcn** from the first `T` seconds of the file |
| Subtract | Gain `α` on `noiseMag`, **Subtract**, then **Math Function / MinMax** with floor `β·noiseMag` |
| Recombine | **Magnitude-Angle to Complex** using the *original* phase |
| Inverse | **IFFT** (real output) |
| Reconstruct | **Overlap-add** (Buffer with negative overlap or Unbuffer + delay/sum, whichever matches the chosen overlap) |
| Output | **To Audio Device** and **To Multimedia File** (`cleaned_simulink.wav`) |
| Analysis | **Spectrum Analyzer** or **Spectrogram** blocks for noisy vs cleaned |

Notes:
- Required toolbox: **DSP System Toolbox**. State this in `simulink/README.md`.
- Stretch (only if time): running noise estimate inside the model (Unit Delay + moving average) instead of InitFcn.
- Provide `build_spectral_subtraction_model.m` that builds the model with `new_system / add_block / add_line` and sets parameters from a config struct. **Mark it untested** and write it defensively (comments on every `add_block` path, because block library paths change across releases). The author may instead build the model by hand using the block table above.
- Provide `run_and_export.m`: runs the model on a given WAV and writes `cleaned_simulink.wav` at the same sample rate.
- `simulink/README.md` must include: block table, parameter values matching the "Match Simulink basic" preset, screenshot placeholders, and a "how to run" section.

### Comparison script
`scripts/compare_simulink_python.py <noisy.wav> <cleaned_simulink.wav>`:
- Runs the Python pipeline with the "Match Simulink basic" preset.
- Aligns outputs (cross-correlation lag), prints RMS error, correlation, SNR between the two outputs, and saves a comparison figure (waveforms + spectrograms + difference).
- Expect small differences (windowing normalisation, edge frames). Document them in `docs/ALGORITHM.md` rather than hiding them.

## 9. Testing (pytest) — required before declaring anything done

- `stft`: perfect reconstruction (STFT→ISTFT) error < 1e-6 for Hann 50% and 75%.
- `spectral_subtraction`: with demo mix at 5 dB SNR (white noise, fixed seed), SNR improvement > 3 dB; output has no NaN/Inf; length equals input length.
- `beta=1, alpha=0` returns the input (identity check).
- Parameter validation rejects out-of-range values.
- API tests with Flask's test client: upload a synthetic WAV → 200 with expected JSON keys; oversize/invalid file → clean 4xx.
- Parity test (if the worklet is built): JS vs Python within tolerance.

## 10. Commands

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
flask --app app run --debug                          # http://127.0.0.1:5000
pytest -q
python scripts/compare_simulink_python.py samples/noisy.wav out/cleaned_simulink.wav
```

`requirements.txt` (pin versions after first successful install): `flask`, `numpy`, `scipy`, `soundfile`, `pytest`.

## 11. Build order (do in this order, one milestone at a time, commit after each)

1. **Skeleton + DSP core:** `stft.py`, `spectral_subtraction.py`, `noise_estimation.py(initial)`, `synth.py`, `metrics.py`, with tests passing. Write `docs/ALGORITHM.md` alongside.
2. **API:** `/api/process`, `/api/audio`, `/api/health`, `/api/presets`, validation, limits.
3. **UI v1:** layout, design system, upload, waveform + spectrogram plots, sliders → reprocess, A/B player.
4. **Recording:** mic capture, level meter, live spectrum, noise-profile capture, WAV encoding in the browser.
5. **Science views:** gain mask, spectra overlay, VAD timeline, STFT frame inspector, demo mode with ground-truth metrics.
6. **Simulink assets:** block table, README, MATLAB builder script, export script, comparison script. Update presets so "Match Simulink basic" is exact.
7. **Polish:** presets, export ZIP, About/Method panel with KaTeX, keyboard shortcuts, empty/error states, responsive layout.
8. **Docs:** `README.md`, `docs/VIVA.md` (at least 20 likely questions with 2–3 line answers: why Hann, why overlap, what is musical noise, why noisy phase, what does α/β do, limitations of the method, etc.), `docs/REPORT_OUTLINE.md`.
9. **Stretch:** AudioWorklet live preview, parameter sweep heatmap, min-statistics tuning.

## 12. Working agreements for Claude

- Work milestone by milestone; run `pytest -q` and start the server to verify before saying a milestone is done. If browser tools are available, take screenshots to check the UI actually renders.
- Ask before adding any new dependency beyond §2 / `requirements-ml.txt`. Denoising libraries go only in `app/enhancers/thirdparty/` (§14).
- Keep functions pure and typed; DSP modules must not import Flask.
- Comment DSP code with equation numbers from `docs/ALGORITHM.md`.
- Don't leave placeholder or fake data in the UI. If a metric can't be computed honestly, show "n/a".
- Keep the repo small: no large audio files, no committed `.venv`, add `.gitignore`.
- When uncertain about MATLAB/Simulink behaviour, say so in the file and in your reply; do not guess silently.
- At the end of each milestone, summarise in 2–3 sentences what works, what is untested, and what is next.

## 13. Definition of done

- `pytest -q` passes; app runs from a fresh clone with the commands in §10.
- Upload and live-record flows both produce cleaned audio; sliders update plots without page reload.
- Demo mode shows ground-truth SNR improvement and SI-SDR.
- Simulink README, block table and scripts exist and are honestly labelled as to what was tested.
- `docs/VIVA.md` and `docs/REPORT_OUTLINE.md` exist.
- The UI and README both state plainly: **the Simulink model is the primary implementation; the web app is an extension.**

## 14. v2 — Algorithm comparison bench (extension)

Goal: answer **"which noise-suppression algorithm works best under which conditions?"** with graphs, statistics and conclusions for the report, as requested by the project guide. The Simulink model stays spectral subtraction only and remains the primary implementation. Everything in this section is a further extension of the web app.

### Algorithms (registry in `app/enhancers/`)

| id | Family | Implementation | Native rate |
|---|---|---|---|
| `spectral_subtraction`, `wiener` | classical | hand-written, `app/dsp/spectral_subtraction.py` (Simulink parity) | any |
| `wiener_dd` | statistical | hand-written, decision-directed Wiener (Scalart & Vieira Filho 1996) | any |
| `mmse_stsa` | statistical | hand-written, Ephraim & Malah 1984 | any |
| `log_mmse` | statistical | hand-written, Ephraim & Malah 1985 | any |
| `omlsa` | statistical, adaptive noise | hand-written, OM-LSA + IMCRA (Cohen 2002/2003) | any |
| `noisereduce` | spectral gating (library) | `noisereduce` | any |
| `webrtc_ns` | production DSP (WebRTC / Meet) | `webrtc-noise-gain` | 16 kHz |
| `rnnoise` | hybrid DSP + RNN (Xiph; Jitsi, OBS) | `pyrnnoise` C library | 48 kHz |
| `nsnet2` | DNN (Microsoft DNS-Challenge baseline) | ONNX via `onnxruntime` | 16 kHz |
| `deepfilternet3` | DNN (Schröter et al. 2023) | `deepfilternet` (torch) | 48 kHz |
| `demucs_dns64` | DNN, waveform (Défossez et al. 2020) | `denoiser` (torch) | 16 kHz |

Rules:
- Each third-party model is optional: availability is detected by try-import, reported in `/api/health` and `/api/algorithms`, and the model is greyed out in the UI with the reason. Tests that need it use `@pytest.mark.ml` and are skipped when it is missing.
- Model files (`models/`), datasets (`data/`) and benchmark output (`results/`) are downloaded or generated and are gitignored. `scripts/fetch_models.py` and `scripts/fetch_datasets.py` verify SHA-256 sums.
- Every algorithm exposes tweakable parameters through the same reject-don't-clamp schema as §4. A generic attenuation limit and a wet/dry mix apply identically to all of them, for fair comparisons.
- For models without an explicit gain, plots use the **effective** mask `|STFT(ŝ)| / |Y|` and label it as such.
- Metrics: §5, plus PESQ (WB) and STOI/ESTOI when a clean reference exists, plus DNSMOS P.835 (SIG/BAK/OVRL) for any audio. DNSMOS is labelled a *model-predicted* MOS, never ground truth.
- Benchmark: `scripts/benchmark.py` runs algorithms × noise types × SNRs × utterances at 16 kHz with fixed seeds, writing `results/bench.csv`. `scripts/report_figures.py` writes the figures and `results/summary.md`. Conclusions are written from those numbers, never in advance.
- Known environment quirks: DeepFilterNet 0.5.6 imports the removed `torchaudio.backend`, so the wrapper installs a stub module before importing `df`. `hydra-core` 0.11 (pulled in by `denoiser`) ships a top-level `tests` package, which is why `tests/__init__.py` exists.

### Commands

```bash
uv venv -p 3.12 .venv-ml
VIRTUAL_ENV=.venv-ml uv pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=.venv-ml uv pip install -r requirements-ml.txt
.venv-ml/bin/python scripts/fetch_models.py
.venv-ml/bin/python scripts/fetch_datasets.py
.venv-ml/bin/python scripts/benchmark.py --quick        # then without --quick for the report
.venv-ml/bin/python scripts/report_figures.py
.venv-ml/bin/flask --app app run --debug
```
