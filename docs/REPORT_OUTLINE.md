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

### 8. Discussion (2 pages)
- Where spectral subtraction works (stationary broadband noise, hum with long frames) and where it fails (clicks, babble, low SNR).
- Musical noise vs residual noise: the α/β trade-off.
- Importance of the noise estimate. What happens when speech leaks into the first 0.5 s.
- Limits of the evaluation: synthetic speech is not real speech, and SI-SDR does not measure perceived quality.

### 9. Conclusion and future work (1 page)
- Summary of achievements against the objectives.
- Future work: noise tracking inside the model (VAD-gated recursive average, minimum statistics), Wiener / MMSE-STSA gain, real-time implementation on hardware (e.g. an embedded target via Simulink code generation), perceptual metrics (PESQ/STOI) if licensing allows.

### References
Boll 1979; Berouti et al. 1979; Le Roux et al. 2019; Martin 2001; MathWorks DSP System Toolbox documentation (blocks used); any datasets or recordings (with licences).

### Appendices
- A. Full block parameter listing (export from the model).
- B. MATLAB scripts (`simulink/*.m`).
- C. Key Python functions (`app/dsp/spectral_subtraction.py`, `stft.py`).
- D. Viva-style Q&A (`docs/VIVA.md`), optional.
