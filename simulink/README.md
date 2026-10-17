# Simulink model: basic spectral subtraction (primary implementation)

**This Simulink model is the primary deliverable of the project.** The NoiseLab web app is an extension that implements the same algorithm in Python for interactive exploration.

> ⚠️ **Testing status — read this first.**
> All `.m` files in this folder were written **without access to MATLAB/Simulink** and are **UNTESTED. Run them and fix them in MATLAB.** Library block paths and parameter names change between releases. The builder script is written defensively: it tries several paths and prints the real parameter names when one is wrong. Expect to correct a few lines anyway.
>
> What *has* been tested is the **design**. `tests/test_simulink_chain.py` emulates this exact block chain frame by frame in Python (Buffer → window → FFT → subtract/max → noisy phase → IFFT → split/delay/add). Its output equals the Python STFT pipeline to within **3 × 10⁻¹⁶**, apart from the expected one-hop delay of 256 samples. If your model is wired as in the table below, it should match Python closely.

Required: **MATLAB + Simulink + DSP System Toolbox** (which includes the Signal Processing Toolbox). No other toolboxes are needed. The Hann window is computed explicitly in `nl_hann_periodic.m`.

---

## 1. Block table

N = 512, H = N/2 = 256. Equation numbers refer to `docs/ALGORITHM.md`.

| # | Stage | Block (library) | Key settings | Eq. |
|---|---|---|---|---|
| 1 | Input | **From Multimedia File** (DSP System Toolbox › Sources) | File: `nl_cfg.inputFile` (mono WAV), samples per frame = `nl_H` (256), output double. *Fallback:* **Signal From Workspace**, `X = nl_x`, `Ts = 1/nl_fs`, samples per frame `nl_H`, then output zeros | — |
| 2 | Framing | **Buffer** (DSP › Signal Management › Buffers) | Output buffer size `nl_N` (512), overlap `nl_N - nl_H` (256), initial conditions 0 | (1) |
| 3 | Window | **Constant** `nl_hannWin` + **Product** (element-wise) | `nl_hannWin = nl_hann_periodic(512)`, i.e. *periodic* Hann | (2) |
| 4 | Transform | **FFT** (DSP › Transforms) | FFT length = input length (512) | (2) |
| 5 | Mag / phase | **Complex to Magnitude-Angle** | Output: Magnitude and angle | (3) |
| 6 | Noise estimate | **Constant** `noiseMag` (512 × 1) | Computed in the model **InitFcn** from the first T = 0.5 s (`nl_compute_noise_mag.m`) | (4) |
| 7 | Over-subtraction | **Gain** `nl_alpha` on `noiseMag` → **Subtract** (`|Y| − α·noiseMag`) | α = 2.0 | (7) |
| 8 | Spectral floor | **Gain** `nl_beta` on `noiseMag` → **MinMax** (`max`, 2 inputs) | β = 0.02 → `max(|Y| − α|N̂|, β|N̂|)` | (7) |
| 9 | Recombine | **Magnitude-Angle to Complex** | Angle input = the **original noisy phase** from block 5 | (12) |
| 10 | Inverse | **IFFT** → **Complex to Real-Imag** (Real) | The spectrum is conjugate-symmetric; taking Re removes rounding-level imaginary parts | (12) |
| 11 | Overlap-add | **Selector** (rows 1:256) + **Selector** (rows 257:512) → **Unit Delay** (IC `zeros(256,1)`) → **Add** | out(k) = first half of frame k + second half of frame k−1. A periodic Hann at 50 % sums to 1, so no normalisation is needed | (13) |
| 12 | Output | **To Multimedia File** (`cleaned_simulink.wav`), **To Workspace** (`nl_cleanedFrames`, 2-D array), **Audio Device Writer** / **To Audio Device** | `run_and_export.m` uses the To Workspace data | (14) |
| 13 | Analysis | **Spectrum Analyzer** (2 inputs: noisy, cleaned) | Switch *View → Spectrogram* for a spectrogram | — |

Why the overlap-add is done this way: with 50 % overlap, each 512-sample IFFT frame overlaps only the previous and the next frame. Output hop k is therefore *first half of frame k* + *second half of frame k − 1*, which is one Selector pair, one Unit Delay and one Add. This avoids the less obvious "Buffer with negative overlap" trick, and the Unbuffer block isn't needed because the sinks accept 256-sample frames directly.

**Latency:** the output at step k is input hop k − 1, so the model delays the signal by H = 256 samples (16 ms at 16 kHz). `run_and_export.m` removes this. The comparison script also aligns by cross-correlation, in case your release's Buffer adds more delay.

## 2. Parameters (= "Match Simulink basic" preset in the web app)

| Parameter | Value | Where |
|---|---|---|
| Frame length N | 512 | `nl_default_config.m` → `cfg.N` |
| Overlap | 50 % (H = 256) | `cfg.overlap` |
| Window | Hann, **periodic** | `nl_hann_periodic.m` |
| Noise estimate | mean \|FFT\| of the frames in the first **0.5 s** (frames start at 0, H, 2H, …) | `cfg.noiseSeconds`, `nl_compute_noise_mag.m` |
| α (over-subtraction) | 2.0 | `cfg.alpha` |
| β (spectral floor) | 0.02 | `cfg.beta` |
| Exponent | 1: **magnitude** subtraction | wiring of blocks 7–8 |
| Phase | noisy phase | block 9 |
| Output | scaled down only if peak > 0.99 | `run_and_export.m` |

`tests/test_simulink_chain.py::test_preset_matches_matlab_config` fails if `nl_default_config.m` and the web app's preset ever disagree.

## 3. How to run

In MATLAB, from the repository root:

```matlab
addpath simulink                                   % helpers + InitFcn must be on the path
cfg = nl_default_config();
cfg.inputFile = fullfile(pwd, 'samples', 'noisy.wav');   % any mono or stereo WAV
build_spectral_subtraction_model(cfg);             % creates simulink/spectral_subtraction_basic.slx
run_and_export(cfg.inputFile, fullfile(pwd, 'out', 'cleaned_simulink.wav'));
```

Then, in a terminal (Python environment from the main README):

```bash
python scripts/compare_simulink_python.py samples/noisy.wav out/cleaned_simulink.wav
```

This prints the alignment lag, RMS error, correlation and the SNR between the two outputs, then writes `out/compare/compare_simulink_python.png` and `compare_report.json`. A correct model gives **SNR between outputs > 40 dB** ("MATCH"). If you see "MISMATCH", check, in order: periodic Hann, α/β, magnitude (not power) subtraction, the noise frames, and the overlap-add delay. `spectral_subtraction_reference.m` is a plain-MATLAB loop with one line per block, so you can compare each stage.

**No sample audio is committed** (licensing; see `samples/README.md`). To make a test input with a known clean reference, use the web app's demo mode (Export → noisy WAV), or run:

```bash
python -c "import soundfile as sf; from app.dsp import synth; d = synth.make_demo('fan', 5, 16000, 6, seed=1); sf.write('samples/noisy.wav', d['noisy'], 16000, subtype='PCM_16')"
```

### Building by hand instead

If the builder script fights your MATLAB release, build the model by hand from the block table in §1. Set the model's **InitFcn** (Model Properties › Callbacks) to:

```matlab
if ~exist('nl_cfg','var'), nl_cfg = nl_default_config(); end; nl_init_model(nl_cfg);
```

Set the solver to **Fixed-step / discrete** and the stop time to `nl_stopTime`.

## 4. Files

| File | Purpose |
|---|---|
| `build_spectral_subtraction_model.m` | Builds and saves the model with `new_system / add_block / add_line` (UNTESTED) |
| `run_and_export.m` | Runs the model on a WAV, removes the latency, writes `cleaned_simulink.wav` (UNTESTED) |
| `nl_default_config.m` | All parameters, identical to the "Match Simulink basic" preset |
| `nl_init_model.m` | InitFcn: loads the file and computes `noiseMag`, the window and the stop time |
| `nl_compute_noise_mag.m` | Eq. (4): noise magnitude from the first T seconds |
| `nl_hann_periodic.m` | Periodic Hann window |
| `spectral_subtraction_reference.m` | The same chain as a plain MATLAB loop (for debugging and the viva) (UNTESTED) |

## 5. Screenshots (add yours)

> Replace these placeholders with screenshots from your own MATLAB session for the report.

- `docs/img/simulink_model.png`: the full block diagram *(placeholder)*
- `docs/img/simulink_spectrum.png`: Spectrum Analyzer, noisy vs cleaned *(placeholder)*
- `docs/img/compare_simulink_python.png`: output of the comparison script *(placeholder)*

## 6. Stretch: running noise estimate inside the model (not implemented)

Instead of the InitFcn constant, |N̂| could be updated inside the model during speech pauses. Feed |Y| through a gated recursive average, `N(k) = λ·N(k−1) + (1−λ)·|Y(k)|` (Unit Delay + Gain + Add), gated by an energy threshold. This is left as an extension. It would no longer match the "initial" mode exactly.
