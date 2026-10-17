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
| `docs/` | [`ALGORITHM.md`](docs/ALGORITHM.md) (math, numbered equations cited in code), [`VIVA.md`](docs/VIVA.md), [`REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md) | — |

No black-box denoisers are used (`noisereduce`, RNNoise, DeepFilterNet, Demucs, …). Every step is in `app/dsp/` and can be explained line by line.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
flask --app app run --debug                          # open http://127.0.0.1:5000
pytest -q                                            # ~140 tests, ~5 s
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
                         metrics, synth, viz_data
  services/              audio_io (safe load/save), jobs (LRU+TTL cache), processing (run + metrics + viz)
  templates/index.html   the UI
  static/{css,js,vendor} vanilla ES modules, no build step
simulink/                model builder, run/export, helpers, reference loop, README (block table)
scripts/compare_simulink_python.py
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
