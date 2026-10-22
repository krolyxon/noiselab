# Project report outline

*Model-Based Design course project: speech noise suppression using spectral subtraction.*
Suggested length: 20–30 pages plus appendices. Items marked 📷 need a screenshot or figure from your own run. Items marked ✍️ need your own measured numbers. Do not copy numbers that you did not reproduce.

---

### Title page, certificate, acknowledgement, abstract (≈ 200 words)
Problem, method (spectral subtraction with over-subtraction and spectral floor, implemented as a Simulink model), the web-app extension, the key result (✍️ e.g. "SI-SDR improvement of X dB on white noise at 5 dB SNR"), and one limitation.

### 1. Introduction (2 pages)
- Noise in speech communication: phones, hearing aids, lecture recordings; mains hum and fans in Indian labs and classrooms.
- Single-channel vs multi-channel enhancement; why single-channel is harder.
- Objectives: (1) a clearly explainable Simulink model, (2) a quantitative evaluation, (3) an interactive extension to explore parameters.
- Scope: basic model = `initial` noise estimate, magnitude subtraction, noisy phase. Extensions are listed separately.

### 2. Background and literature (3 pages)
- Additive noise model, short-time Fourier analysis, COLA windows.
- Boll (1979): magnitude subtraction. Berouti et al. (1979): over-subtraction α and floor β.
- Musical noise and the remedies for it. Wiener filtering as a gain rule. MMSE-STSA (Ephraim–Malah) and minimum statistics (Martin), for context only.
- Evaluation metrics: SNR, segmental SNR, SI-SDR (Le Roux et al., 2019).

### 3. Algorithm (4 pages)
Use `docs/ALGORITHM.md` and keep its equation numbers.
- Framing and windowing (Eq. 1–2), noise estimation (Eq. 4), subtraction rule (Eq. 7–8), noisy-phase reconstruction (Eq. 12), overlap-add (Eq. 13), output stage (Eq. 14).
- Block diagram of the signal flow (draw it; it maps one-to-one to the Simulink model).
- Parameter table, with justification for N = 512, 50 % Hann, T = 0.5 s, α = 2, β = 0.02, p = 1.

### 4. Simulink implementation: primary deliverable (5–6 pages)
- 📷 Full model screenshot, plus close-ups of: framing (Buffer + window), subtraction (Gain, Subtract, MinMax), overlap-add (Selectors, Unit Delay, Add).
- Block table (from `simulink/README.md`) with the setting of every block.
- InitFcn and workspace variables. Why the noise estimate is a Constant computed before the simulation.
- Latency (one hop) and how it is removed.
- Toolboxes used (Simulink, DSP System Toolbox) and the MATLAB release (✍️).
- **Testing status:** state honestly which scripts you ran and what you changed (✍️). The provided builder was untested when you received it.
- 📷 Spectrum Analyzer, noisy vs cleaned.

### 5. Verification: Simulink vs Python (2 pages)
- Why a second implementation: an independent check plus fast exploration.
- Method: same input WAV → `run_and_export.m` → `compare_simulink_python.py`.
- ✍️ Table: lag, RMS error, correlation, SNR between outputs (full / excluding edge frames).
- 📷 The comparison figure (waveforms, spectrograms, difference).
- Discussion of any residual differences (`docs/ALGORITHM.md` §10).
- Mention the design-level verification: block-chain emulation = Python to 3 × 10⁻¹⁶ (`tests/test_simulink_chain.py`).

### 6. NoiseLab web app: extension (3 pages)
- State clearly: *an extension of the Simulink model; the Simulink model is the primary implementation.*
- Architecture: Flask API, pure-NumPy DSP module, browser UI (no build step, vendored libraries).
- Features used in the evaluation: demo mode with ground truth, parameter presets, A/B listening, STFT frame inspector, gain mask, export.
- 📷 Main screen, frame inspector, Method dialog.
- Safety and limits: upload validation, duration limits, job expiry.
- Software testing: number of automated tests (✍️ run `pytest -q`), headless UI checks.

### 7. Results (4 pages)
- Test material: synthetic voiced signal + six noise types (white, pink, 50 Hz hum, fan, babble-like, clicks), SNR −5 … 20 dB, fixed seeds. Plus at least one real recording of your own (✍️).
- ✍️ Table: SI-SDR / SNR improvement per noise type for the Simulink settings (start from the README table and reproduce it).
- ✍️ Plot: improvement vs input SNR (−5, 0, 5, 10, 20 dB) for white and fan noise.
- ✍️ Effect of α and β: a short sweep showing the trade-off. Describe the musical noise you hear and point to the gain mask.
- ✍️ Effect of N: 256 / 512 / 1024 / 2048 on hum vs white noise.
- Real recording: no ground truth, so report noise reduction (dB) and describe the listening impression. Do **not** report SNR or SI-SDR for it.

### 7A. Comparative evaluation of noise-suppression algorithms (5–6 pages), *asked for by the guide*
State first that this is a further extension: the Simulink model remains spectral subtraction and is the primary deliverable.
- **Algorithms** (table): family, idea in one line, native rate, latency, who uses it:
  - SS (Simulink settings)
  - Wiener-DD, MMSE-STSA, log-MMSE, OM-LSA/IMCRA (hand-written, Eq. 22–33)
  - noisereduce, WebRTC NS
  - RNNoise, NSNet2, DeepFilterNet3, Demucs (pretrained, third-party)
  - Cite each paper (`docs/ALGORITHM.md` §12, VIVA Q34–Q50).
- **Test material:**
  - Speech: 20 LibriSpeech test-clean utterances (10 F / 10 M).
  - Noise: 6 synthetic types + 5 DEMAND environments.
  - SNR: −5 … 20 dB.
  - A 1 s noise lead-in. Fixed seeds. `scripts/fetch_datasets.py`, `scripts/benchmark.py`.
- **Fairness:** same mixtures, time alignment (measured model delays), single-thread RTF, defaults plus a separate parameter sweep (VIVA Q41).
- **Metrics:** ΔSI-SDR, ΔSNR, PESQ, ESTOI, LSD, noise reduction / speech attenuation, DNSMOS (model-predicted, reference-free).
- **Make it readable:** put fig. 0 (`fig0_glossary.png`, every term in plain words + the 11 methods in one line each) right before the results. Under each figure, copy its "How to read" line and add one sentence on what it shows. Use the plain names in the text ("speech clarity gain") and the technical name once in brackets ("ΔSI-SDR"). [`docs/GLOSSARY.md`](GLOSSARY.md) has the wording; VIVA Q51–Q55 have simple explanations.
- **Results:** figures from `scripts/report_figures.py`, with numbers from `results/summary.md`:
  - Fig. 1: ΔSI-SDR vs input SNR, by noise class.
  - Fig. 2: PESQ / ESTOI vs SNR, with the unprocessed baseline.
  - Fig. 3: best algorithm per (noise, SNR). *This is the "which algorithm when" answer.*
  - Fig. 9 / 10: algorithm × noise heatmaps.
  - Fig. 4: noise reduction vs speech distortion trade-off.
  - Fig. 5: quality vs real-time factor (Pareto front).
  - Fig. 6: stationary vs non-stationary noise.
  - Fig. 7: parameter sensitivity (SS α, log-MMSE α_dd, OM-LSA G_min, WebRTC level, DFN/Demucs attenuation limit).
  - Fig. 8: spectrograms of one utterance through every method.
  - Fig. 11: DNSMOS vs PESQ agreement.
- ✍️ **Conclusions:** write only what the numbers show, with 95 % CIs and Wilcoxon p-values from `summary.md`. Suggested structure:
  - (a) Which family wins overall, and by how much.
  - (b) Where classical methods are competitive or better (stationary noise, hum, high SNR?).
  - (c) Where neural models clearly win (non-stationary / real noise?).
  - (d) Cost: how much quality each step up in compute buys.
  - (e) The noise-removal vs speech-distortion trade-off.
- ✍️ A real recording of your own through every method (Compare panel in the app): DNSMOS + listening notes only, no SI-SDR.

### 8. Discussion (2 pages)
- Where spectral subtraction works (stationary broadband noise, hum with long frames) and where it fails (clicks, babble, low SNR).
- Musical noise vs residual noise: the α/β trade-off.
- Importance of the noise estimate. What happens when speech leaks into the first 0.5 s.
- Limits of the evaluation: synthetic speech is not real speech, and SI-SDR does not measure perceived quality.

### 9. Conclusion and future work (1 page)
- Summary of achievements against the objectives.
- Future work:
  - Port the best hand-written statistical estimator (log-MMSE or OM-LSA) into the Simulink model, as a MATLAB Function block on the same FFT chain.
  - Noise tracking inside the model (IMCRA is already written in Python).
  - Real-time implementation on hardware, e.g. an embedded target via Simulink code generation.
  - A small listening test, to check the DNSMOS/PESQ rankings.

### References
Boll 1979; Berouti et al. 1979; Le Roux et al. 2019; Martin 2001; MathWorks DSP System Toolbox documentation (blocks used).
Comparison extension:
- Ephraim & Malah 1984, 1985; Scalart & Vieira Filho 1996; Cohen & Berdugo 2001; Cohen 2003; Loizou 2013 (textbook).
- Valin 2018 (RNNoise); Braun & Tashev 2020 (NSNet2); Schröter et al. 2023 (DeepFilterNet); Défossez et al. 2020 (Demucs); Sainburg et al. 2020 (noisereduce); WebRTC.
- Metrics: ITU-T P.862.2 (PESQ), Taal et al. 2011 / Jensen & Taal 2016 (STOI / ESTOI), Reddy et al. 2022 (DNSMOS P.835).
- Data: Panayotov et al. 2015 (LibriSpeech, CC BY 4.0), Thiemann et al. 2013 (DEMAND, CC BY 4.0).

### Appendices
- A. Full block parameter listing (export from the model).
- B. MATLAB scripts (`simulink/*.m`).
- C. Key Python functions (`app/dsp/spectral_subtraction.py`, `stft.py`).
- D. Viva-style Q&A (`docs/VIVA.md`), optional.
