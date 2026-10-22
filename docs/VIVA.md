# Viva preparation: likely questions and short answers

Answers are 2–3 lines, phrased the way you might say them. Equation numbers refer to `docs/ALGORITHM.md`.

## A. The method

**1. What is spectral subtraction, in one sentence?**
We estimate the average noise spectrum from a noise-only part of the signal and subtract it from the spectrum of every short frame. Then we rebuild the waveform using the original phase (Boll, 1979).

**2. What assumption does it rely on?**
That the noise is **additive**, **uncorrelated with speech** and **stationary** (its spectrum doesn't change much). Under those assumptions |Y|² ≈ |S|² + |D|² on average, so subtracting a noise estimate recovers |S|.

**3. Why work in short frames instead of one big FFT?**
Speech is non-stationary. Its spectrum changes every few tens of milliseconds, but over 20–40 ms it is roughly stationary. N = 512 at 16 kHz is 32 ms, so each frame sees one "sound".

**4. Why use a window at all, and why Hann?**
Cutting a frame with a rectangular window causes strong spectral leakage, with sidelobes at −13 dB. Hann has much lower sidelobes (−31 dB, falling fast), so noise and speech components stay in their own bins. Also, a **periodic** Hann at 50 % overlap sums to exactly 1, so plain overlap-add reconstructs the signal perfectly.

**5. Why 50 % overlap? What changes at 75 %?**
Without overlap, the window tapers would make the signal "pump" at the frame rate. With 50 % overlap, Hann windows add up to a constant (COLA). At 75 % the time resolution is finer and the gain changes more smoothly between frames, which slightly reduces musical noise, but it costs twice the computation.

**6. What is the trade-off in choosing N?**
Time vs frequency resolution: Δf = f_s/N. A long N (2048) resolves closely spaced tones, like the 50 Hz hum harmonics, but smears speech onsets in time. A short N (256) follows fast changes but blurs frequency. The results table in the README shows the 2048-point preset winning on hum.

**7. Why do we keep the noisy phase?**
Estimating the clean phase is much harder, and at moderate SNR the ear is fairly insensitive to phase errors. Since Ŝ = G·Y with a real, non-negative G, only the magnitude changes, and the phase of Y is reused (Eq. 12).

**8. What do α and β do?**
α (over-subtraction, > 1) removes *more* than the average noise, so the random peaks of the noise above its mean are also removed. β (spectral floor) keeps every bin at least at β·|N̂|, so nothing drops to zero (Eq. 7). Larger α: less noise, more distortion. Larger β: less musical noise, more residual hiss.

**9. What is "musical noise" and why does it occur?**
After subtraction, a few bins where the noise happened to be above its average survive as isolated, short-lived peaks. They are heard as random "tinkling" tones. Over-subtraction (α), the spectral floor (β) and gain smoothing (extended options) all reduce it. In the frame inspector it shows up as amber |Y| poking above the dashed α|N̂| line at random bins.

**10. Magnitude (p = 1) vs power (p = 2) subtraction?**
Power subtraction follows from |Y|² ≈ |S|² + |D|²; magnitude subtraction is Boll's original form. With the same α, p = 1 removes more noise, because α|N̂| in magnitude corresponds to α²|N̂|² in power. That's why the Simulink preset (p = 1, α = 2) scores higher than the Python default (p = 2, α = 2) in our table.

**11. How is the noise estimated in your model?**
The mean magnitude spectrum of the frames inside the first 0.5 s, assumed speech-free (Eq. 4). It is computed once in the model's InitFcn and fed in as a Constant block. The web app also offers a selected region, a separately recorded noise profile, and minimum statistics (a low percentile per bin).

**12. What happens if speech starts in the first 0.5 s?**
Speech energy enters the noise estimate. The algorithm then subtracts part of the speech itself, and the result sounds muffled and distorted. That's why the recording hint says "wait about 1 s before speaking", and why the region and profile modes exist.

**13. What is the Wiener-style option and how does it differ?**
G = max(1 − α·|N̂|²/|Y|², β) (Eq. 9). It is a gain-domain rule, so the floor is relative to |Y| rather than to the noise. It is smoother and usually gives less musical noise. It is an *extended* option and not in the basic Simulink model.

## B. Implementation (Simulink)

**14. Walk me through the Simulink model.**
From Multimedia File (256-sample frames) → Buffer (512, overlap 256) → × Hann → FFT → Complex to Magnitude-Angle. Then |Y| − α·noiseMag → MinMax(max) with β·noiseMag → Magnitude-Angle to Complex, using the noisy phase. Then IFFT → real part → overlap-add → To Multimedia File / audio device. Each block maps to one equation (`simulink/README.md`).

**15. How did you do the overlap-add in Simulink?**
With 50 % overlap, output hop k = first half of IFFT frame k + second half of frame k − 1. So I select the two halves with two Selector blocks, delay the second half by one frame with a Unit Delay, and add them. No normalisation is needed, because the periodic Hann windows sum to 1.

**16. Why does the Simulink output lag the input?**
The Unit Delay in the overlap-add delays the output by one hop: 256 samples, 16 ms at 16 kHz. `run_and_export.m` removes it. The comparison script also finds any lag by cross-correlation and reports it.

**17. Why a Constant + Product for the window instead of the Window Function block?**
To avoid any ambiguity between the symmetric and periodic Hann. Only the periodic one sums exactly to 1 at 50 % overlap. The vector comes from `nl_hann_periodic.m`, w[n] = 0.5 − 0.5cos(2πn/N).

**18. How do you know the Python app implements the same algorithm?**
Three ways:
- The "Match Simulink basic" preset uses identical parameters, and a test fails if `nl_default_config.m` and the preset ever differ.
- `tests/test_simulink_chain.py` emulates the block chain frame by frame and matches the Python pipeline to 3 × 10⁻¹⁶, after the one-hop shift.
- `scripts/compare_simulink_python.py` compares the real Simulink output with Python.

**19. What differences would you expect between Simulink and Python, and why?**
A latency of whole hops (removed by alignment), possible edge effects in the first and last frame from the Buffer's initial conditions, and tiny floating-point differences. If both read the same WAV and the model is wired correctly, the SNR between the outputs should be well above 40 dB (`docs/ALGORITHM.md` §10).

## C. Evaluation

**20. How do you measure performance?**
With a clean reference (synthetic demo mixtures): SNR (Eq. 16), SI-SDR (Eq. 17) and segmental SNR (Eq. 18), before and after. Without a reference: noise reduction over noise-only frames (Eq. 19) and a rough VAD-based SNR estimate. Plus listening, because no single number captures musical noise.

**21. Why SI-SDR and not just SNR?**
Plain SNR punishes a simple change of loudness. SI-SDR first finds the best scaling of the clean signal, then measures the error, so it judges *distortion + residual noise*, not volume. It's the standard modern metric (Le Roux et al., 2019).

**22. Why is segmental SNR different from global SNR?**
Global SNR is dominated by the loud frames. Segmental SNR averages the per-frame SNR (clamped to [−10, 35] dB), so quiet speech frames count equally, which is closer to perception. We skip frames where the clean signal is near-silent.

**23. Why don't you show SNR for an uploaded recording?**
SNR needs the clean signal, and for a real recording we don't have it. Showing an estimate as if it were true SNR would be misleading, so the UI says "n/a (no clean reference)" and offers reference-free indicators instead.

**24. How does your VAD work and how good is it?**
Energy-based: a frame is "speech" if its energy is more than 3 dB above the 20th-percentile frame energy, with a 3-frame hangover (Eq. 20). On our synthetic mixtures it is 74–83 % accurate per frame at 0 dB SNR and 72–89 % at 10 dB; impulsive clicks are the worst case. It's only used for the reference-free indicators.

**25. How did you make the tests reproducible?**
All synthetic signals are generated from a seed (`app/dsp/synth.py`). The mixer scales the noise to the exact requested SNR (Eq. 21). The test suite checks perfect STFT reconstruction (< 10⁻⁶), SNR improvement > 3 dB on white noise at 5 dB, and an identity case (α = 0, β = 1, zero noise ⇒ output = input).

## D. Limitations and extensions

**26. What are the main limitations of spectral subtraction?**
- It assumes stationary noise, so babble, music, clicks and passing cars are poorly handled.
- It produces musical noise and speech distortion, both worse at low SNR.
- It needs a good noise estimate, from a silent lead-in or tracking.
- It uses the noisy phase, which limits quality at very low SNR.
Our clicks result (+0.7 to +4.6 dB) shows the non-stationary case.

**27. How could you track changing noise?**
Update |N̂| during speech pauses (a VAD-gated recursive average), or use minimum statistics (Martin, 2001), which tracks the minimum of the smoothed power per bin. NoiseLab has a simplified percentile version (`minstats`). A running estimate inside Simulink is listed as a stretch goal.

**28. What would modern systems do differently?**
Statistical estimators (Ephraim–Malah MMSE-STSA, log-MMSE) with a decision-directed a-priori SNR, or neural networks (e.g. RNNoise, DeepFilterNet) that learn a mask. The core project (Simulink + the basic Python chain) stays spectral subtraction, because it is explainable block by block. The comparison extension (section E) then puts SS next to both families: the statistical estimators hand-written, the neural ones as third-party baselines.

**29. Why is β·|N̂| sometimes *above* |Y| (G > 1)?**
The floor in Eq. (7) is absolute. In a bin where |Y| is already below β^{1/p}|N̂|, the rule raises it to the floor. That fills spectral holes and is part of what masks musical noise. When smoothing is enabled, the gain is capped at 1 first, otherwise the smoothing would spread those raised gains to neighbouring bins (`docs/ALGORITHM.md` §4).

**30. What was the most surprising bug you found?**
During development, gain smoothing *lowered* SNR to −4 dB with a Hamming window at 75 % overlap. The cause was the >1 floor-fill gains described in Q29 being smeared into neighbouring bins. Capping G at 1 before smoothing fixed it, and a test now covers it. (Another: the energy VAD with a 6 dB margin found only 8–20 % speech at 0 dB SNR, so the margin was lowered to 3 dB after measuring accuracy.)

**31. Why build a web app at all if Simulink is the deliverable?**
It's an extension for exploring the algorithm: change α or β and hear and see the effect immediately, inspect individual STFT frames, record live, and compare quantitatively with ground truth. It implements the same steps, so insights carry over to the model.

**32. Can it run in real time? What is the latency?**
Yes. The live preview runs the basic algorithm in an AudioWorklet, one 512-point FFT per 256-sample hop. The algorithmic latency is N samples: one hop to fill the buffer plus one hop of overlap-add delay, i.e. 10.7 ms at 48 kHz, plus the sound card's I/O latency. Longer frames add latency, so N = 2048 gives 43 ms.

**33. Why is the plain percentile ("minimum statistics") estimate biased, and how did you fix it?**
In noise-only bins |Y|² is roughly exponentially distributed, so its q-quantile is −ln(1−q)·σ², only about 0.105·σ² for q = 10 %, an 8.7 dB underestimate. Dividing by −ln(1−q), and scaling by π/4 to get a mean magnitude, removes the bias (Eq. 6). On the demo set this raised the SI-SDR gain from +1.5 dB to +4.0 dB.

## E. Comparison extension (statistical and neural methods)

**34. What did you add beyond spectral subtraction, and why?**
A comparison bench, because the project guide asked which algorithm works best under which conditions. It has four hand-written statistical estimators (Wiener-DD, MMSE-STSA, log-MMSE, OM-LSA/IMCRA), one library method (noisereduce), one production DSP method (WebRTC NS), and four neural networks (RNNoise, NSNet2, DeepFilterNet3, Demucs). All run on the same STFT/plotting/metrics pipeline and the same benchmark. The Simulink model is unchanged and still the primary deliverable.

**35. What is the a-priori SNR, and what does "decision-directed" mean?**
γ = |Y|²/λ_d (a-posteriori SNR) is what we measure. ξ = E|S|²/λ_d (a-priori SNR) is what the gain really needs, but the clean speech is unknown. The decision-directed rule (Eq. 24) estimates ξ as a weighted mix of the *previous frame's clean estimate* (G²γ of frame m−1) and the current max(γ−1, 0). With α_dd = 0.98 the first term dominates.

**36. Why do MMSE methods produce less musical noise than spectral subtraction?**
In SS, a random noise peak that exceeds α|N̂| passes straight through as an isolated tone. In the decision-directed methods, ξ in noise-only regions is dominated by the previous, already small estimate, so a single peak raises ξ only slightly and its gain stays low. The gain varies smoothly in time (Cappé 1994 analysed this). ξ_min sets the residual-noise floor that masks what is left.

**37. Difference between MMSE-STSA and log-MMSE?**
MMSE-STSA minimises the error of the amplitude, E(|S|−|Ŝ|)² (Eq. 26). Log-MMSE minimises the error of the log-amplitude (Eq. 27), which matches loudness perception better and suppresses low-SNR bins slightly more, giving less residual noise for similar distortion. Both tend to the Wiener gain at high SNR (unit test).

**38. What does OM-LSA add, and what is IMCRA?**
OM-LSA weights the log-MMSE gain by the speech-presence probability p: G = G_LSA^p · G_min^(1−p) (Eq. 28). Where speech is absent, the gain falls smoothly to G_min. IMCRA is its noise tracker: it smooths |Y|², tracks the minimum over about 1 s, uses that to estimate where speech is absent, and updates the noise estimate quickly there and not at all where speech is present (Eq. 30–33). So OM-LSA does not need a silent lead-in, and it follows noise whose level changes. The unit test steps the noise by +10 dB, and the estimate re-tracks to within 2 dB.

**39. How does a neural denoiser differ from these?**
It learns the mapping noisy → clean from thousands of hours of data, instead of assuming Gaussian speech and noise. RNNoise and NSNet2 predict a gain mask (like ours, but learned). DeepFilterNet3 also predicts complex filter taps that restore harmonics. Demucs outputs the waveform directly, with no STFT and no noisy phase. They need no noise estimate from the user.

**40. Why did you not write the neural networks yourself?**
Training needs large datasets and GPU time, which is outside the scope of an MBD course project. The pretrained models are used only as comparison baselines, in their own folder (`app/enhancers/thirdparty/`), and are clearly labelled as third-party. Everything the project claims as its own is hand-written and documented with equations.

**41. How did you make the comparison fair?**
- Every method gets the same mixtures: same utterances, noise segments, SNRs and seeds.
- Metrics are computed by the same code.
- Outputs are time-aligned: model delays were measured and removed (RNNoise 960 samples at 48 kHz, WebRTC 96 samples at 16 kHz), and resampling is zero-phase.
- Every method can share one attenuation limit (Eq. 34).
- Runtime is measured on one CPU thread for all of them.
- Default parameters are used, and sensitivity is shown separately in the sweep.

**42. Why did one sample of delay matter so much?**
SI-SDR and SNR compare waveforms sample by sample. A 1-sample shift at 16 kHz is a phase error of 2π·f/16000, already 45° at 2 kHz. During development RNNoise lost up to 9 dB of SI-SDR from a 1-sample error. Perceptual metrics such as PESQ align internally, which is why they barely changed. Lesson: always check alignment before trusting waveform metrics.

**43. Why does WebRTC have a "phase compensation" switch?**
WebRTC always applies a high-pass filter before its noise suppressor. Its phase shift at 100–200 Hz is inaudible, but it makes SI-SDR drop by several dB even with suppression off. We identify the filter once (white noise in, cross-spectrum out), undo its *phase only*, and keep its magnitude, which really is part of WebRTC. The switch lets you show the effect in the viva.

**44. Which metrics did you use, and what does each measure?**
- SNR and SI-SDR: waveform error, which needs the clean reference.
- Segmental SNR: frame-wise SNR.
- PESQ (P.862.2): perceived quality, MOS-like 1–4.6.
- STOI/ESTOI: intelligibility, 0–1.
- LSD: spectral distortion of the speech.
- Noise reduction and speech attenuation: reference-free, from the VAD.
- DNSMOS: a neural prediction of listener ratings, also reference-free.

A good method has high noise reduction *and* low speech attenuation or LSD, and these pull against each other.

**45. What is DNSMOS and why should it not be over-trusted?**
A CNN trained on crowd-sourced P.835 ratings predicts SIG (speech), BAK (background) and OVRL (overall) on a 1–5 scale. It needs no clean reference, so it works on real recordings. But it is a model: it is biased toward the kind of audio it was trained on (it rates our synthetic voiced signal poorly even when clean), and it can be fooled. We report how well it agrees with PESQ (Spearman ρ in `results/summary.md`) and use it for ranking only.

**46. Why does the benchmark use real speech instead of your synthetic signal?**
The neural models learned the statistics of real speech. On a synthetic harmonic signal they can misbehave, and DNSMOS scores it as poor speech. LibriSpeech test-clean (40 speakers, CC BY 4.0) gives real speech. The DEMAND noises (car, street, cafeteria, office, living room) give real noise. The synthetic noises are kept because they isolate effects such as hum, clicks and stationarity.

**47. What is the trade-off between latency, compute and quality?**
Latency: frame length plus look-ahead (WebRTC 10 ms frames, RNNoise 20 ms, NSNet2 20 ms window, DeepFilterNet3 about 40 ms). Compute: the real-time factor on one CPU core, from about 0.001 for SS to about 0.3 for Demucs, a 300× spread (fig. 5). A video-call app needs RTF ≪ 1 and latency < 40 ms, which is why products use RNNoise/DeepFilterNet-sized models rather than the biggest network.

**48. Where do neural models fail?**
On noise unlike their training data: steady tones and hum, clicks, music. When the input is already clean they can damage it (speech attenuation, "underwater" artefacts). At very low SNR they may also remove speech or hallucinate. Check `results/summary.md` ("cases where a configuration made SI-SDR worse") and fig. 3 for the cells where a classical method won.

**49. What is a paired Wilcoxon test and why use it?**
Each mixture is processed by both algorithms, so the results are *paired*. The Wilcoxon signed-rank test asks whether the median of the per-mixture differences is zero, without assuming normality. Metric differences are often skewed, which is why we use it rather than a t-test. A p < 0.05 means the winner's advantage is unlikely to be chance for that noise type.

**50. If you had to ship one algorithm in a phone app, which and why?**
Answer from your own results (fig. 3, fig. 5, summary.md). The usual reasoning: the best quality per unit of compute at real-time latency, which is typically a small neural model such as DeepFilterNet3 or RNNoise. A statistical method (log-MMSE / OM-LSA) is the choice when there is no budget for a neural network or the noise is stationary. Spectral subtraction is the right *teaching* algorithm, and the easiest to implement in Simulink or on a DSP chip.

