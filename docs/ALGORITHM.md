# Spectral Subtraction — Algorithm Reference

This is the single mathematical reference for the project. Both the **Simulink model (the primary implementation)** and the Python code in the NoiseLab web app (an extension) implement the steps below. Python source comments cite the equation numbers used here, e.g. `# Eq. (7)`.

Main references:
- S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," *IEEE Trans. ASSP*, 27(2), 113–120, 1979.
- M. Berouti, R. Schwartz, J. Makhoul, "Enhancement of speech corrupted by acoustic noise," *Proc. ICASSP*, 1979 (over-subtraction α and spectral floor β).

---

## 0. Signal model

The observed signal is clean speech plus additive noise that is uncorrelated with it:

$$y[n] = s[n] + d[n] \tag{0}$$

In the short-time Fourier domain, |Y|² ≈ |S|² + |D|² on average, because the cross terms average out. So if we can estimate the noise spectrum, we can subtract it.

## 1. Framing

The signal is cut into frames of length N, starting every H samples:

$$y_m[n] = y[mH + n],\quad 0 \le n < N,\qquad H = N\,(1-\text{overlap}) \tag{1}$$

Default N = 512 and overlap 50 %, so H = 256. At 16 kHz that is a 32 ms frame and a 16 ms hop. Speech is roughly stationary over 20–40 ms, which is why these values are used.

*Implementation detail (Python):* the signal is zero-padded by N − H samples at the start and end. Frame 0 then begins at original sample −(N − H), and every original sample is covered by the same number of frames. Reconstruction is then exact right up to the first and last sample (see §6).

## 2. Window and FFT

$$Y_k(m) = \sum_{n=0}^{N-1} w[n]\,y_m[n]\,e^{-j2\pi kn/N} \tag{2}$$

$$|Y_k(m)|,\quad \angle Y_k(m) \tag{3}$$

The window w is **periodic** Hann by default (MATLAB `hann(N,'periodic')`, SciPy `get_window('hann', N, fftbins=True)`). The periodic form matters because at 50 % and 75 % overlap it sums to an exact constant (constant-overlap-add, COLA):

$$\sum_m w[n - mH] = \begin{cases}1 & \text{Hann, 50 \%}\\ 2 & \text{Hann, 75 \%}\end{cases}$$

Other options are Hamming (also COLA, with constant 1.08 at 50 %) and sqrt-Hann (applied twice, at analysis and at synthesis, so the product is Hann). Python stores only the one-sided spectrum k = 0 … N/2, which is N/2 + 1 bins. The other half is the complex-conjugate mirror.

## 3. Noise estimate

All modes produce one magnitude value per frequency bin, |N̂_k|.

**`initial`** (the default, and the only mode in the basic Simulink model): average over the M₀ complete frames inside the first T seconds, which are assumed to contain no speech:

$$|\hat N_k| = \frac{1}{M_0}\sum_{m \in \mathcal{M}_0} |Y_k(m)| \tag{4}$$

𝓜₀ contains the frames that start at samples 0, H, 2H, … and end at or before T·f_s. The Simulink `InitFcn` uses exactly these frames. If T is shorter than one frame, the single frame starting at 0 is used.

**`region`**: Eq. (4) over the frames inside a user-selected interval [t₀, t₁]:

$$|\hat N_k| = \frac{1}{|\mathcal{M}_R|}\sum_{m \in \mathcal{M}_R} |Y_k(m)|,\quad \mathcal{M}_R = \{m : t_0 f_s \le mH - (N-H),\ mH + H \le t_1 f_s\} \tag{5}$$

**`profile`**: Eq. (4) over all complete frames of a separately recorded noise-only clip.

**`minstats`** (simplified minimum statistics, bias-compensated): a low percentile q (default 20 %) of the power in each bin over all frames. It is corrected for the bias of taking a low quantile:

$$|\hat N_k| = \sqrt{\frac{\pi}{4}\cdot\frac{\operatorname{P}_q\{|Y_k(m)|^2\}_m}{-\ln(1-q)}} \tag{6}$$

Why the correction: in noise-only frames |Y_k|² is approximately exponentially distributed with mean σ², so its q-quantile is −ln(1−q)·σ². Without the division, the 10th percentile underestimates the noise power by 8.7 dB: measured −8.70 dB on white noise, theory −8.72 dB. The factor π/4 converts σ² into the *mean magnitude* (E|Y| = √(π/4)·σ for a Rayleigh magnitude), so the result is comparable with Eq. (4). On the demo set (6 noise types × 0/5/10 dB × 3 seeds) the correction raised the mean SI-SDR improvement from +1.5 dB to +4.0 dB (q = 20 %), against +4.6 dB for the `initial` mode. q = 30 % scored slightly higher (+4.2 dB), but 20 % is the safer default when speech has few pauses.

This mode needs no silent segment. It fails if speech never pauses, because then even the low percentile contains speech.

## 4. Subtraction (Berouti over-subtraction with spectral floor)

With exponent p (1 = magnitude subtraction, 2 = power subtraction):

$$|\tilde S_k(m)|^p = \max\!\big(|Y_k(m)|^p - \alpha\,|\hat N_k|^p,\;\; \beta\,|\hat N_k|^p\big) \tag{7}$$

- **α (over-subtraction, default 2.0):** removes more than the mean noise, so that noise peaks above the mean are suppressed too. A larger α gives less residual noise but more speech distortion.
- **β (spectral floor, default 0.02):** never let a bin go below a small fraction of the noise. This replaces the isolated random peaks of "musical noise" with a smooth low-level background.

The same operation written as a real gain:

$$G_k(m) = \frac{|\tilde S_k(m)|}{|Y_k(m)|} \tag{8}$$

*Note:* the floor is absolute (β|N̂|^p), so in bins where |Y| < β^{1/p}|N̂| the gain is **greater than 1**. Such bins are raised up to the floor. That is exactly what Eq. (7) says, and the Simulink model does the same.

### Wiener-style variant (Python only, `method = wiener`)

$$G_k(m) = \max\!\Big(1 - \alpha\,\frac{|\hat N_k|^2}{|Y_k(m)|^2},\;\beta\Big) \tag{9}$$

### Extended refinements (Python only, off by default)

Recursive time smoothing of the gain, with λ = `gain_smooth`:

$$\bar G_k(m) = \lambda\,\bar G_k(m-1) + (1-\lambda)\,G_k(m) \tag{10}$$

Moving average over `freq_smooth_bins` neighbouring bins:

$$\bar G_k(m) = \frac{1}{B}\sum_{|i| \le B/2} \bar G_{k+i}(m) \tag{11}$$

Both reduce musical noise, at the cost of temporal or spectral smearing. **Before smoothing, G is capped at 1.** Without the cap, the large >1 "floor-fill" gains described above would be spread into neighbouring bins and frames and amplify them. During development this lowered SNR from +12 dB to −4 dB on the white-noise demo with Hamming at 75 % overlap. The cap is applied only when smoothing is on, so the basic path still matches Simulink.

## 5. Reconstruction with the noisy phase

$$\hat S_k(m) = G_k(m)\,Y_k(m) = |\tilde S_k(m)|\,e^{j\angle Y_k(m)} \tag{12}$$

The phase is not estimated. The phase of the noisy signal is reused. The ear is relatively insensitive to phase at the SNRs where spectral subtraction is used, and estimating the clean phase is a much harder problem. Multiplying by a real, non-negative G changes only the magnitude, so the two forms in Eq. (12) are identical.

## 6. Inverse FFT and overlap-add

$$\hat s[n] = \frac{\sum_m v[n-mH]\;\mathrm{IFFT}\{\hat S(m)\}[n-mH]}{\sum_m w[n-mH]\,v[n-mH]} \tag{13}$$

Here v is the synthesis window: v = 1 for Hann and Hamming, v = w for sqrt-Hann. For **periodic Hann at 50 % overlap the denominator is exactly 1**, so Eq. (13) is plain overlap-add. That is what the Simulink model does: it adds the first half of each IFFT frame to the second half of the previous frame. Python divides by the window envelope so that 75 % overlap and the other windows also reconstruct exactly. The unit test checks STFT → ISTFT error < 10⁻⁶ for every window, N and overlap.

## 7. Output stage

$$x_\text{out} = g\,\hat s,\quad g = 10^{G_\text{dB}/20};\qquad x_\text{out} \leftarrow 0.99\,\frac{x_\text{out}}{\max|x_\text{out}|}\ \text{ only if }\max|x_\text{out}| > 0.99 \tag{14}$$

The "removed noise" signal is computed as y − ŝ, using ŝ *before* the output stage. That way it shows what the algorithm removed, unaffected by the output gain.

---

## 8. Metrics

**RMS level**

$$L_\text{RMS} = 10\log_{10}\!\Big(\tfrac{1}{K}\textstyle\sum_{n=0}^{K-1} x[n]^2\Big)\ \text{dBFS} \tag{15}$$

**SNR** (needs the clean reference s; demo mode only)

$$\mathrm{SNR} = 10\log_{10}\frac{\sum s^2}{\sum (s-\hat s)^2} \tag{16}$$

**SI-SDR** (Le Roux et al., 2019). This is scale-invariant, so an overall gain change does not affect it. Both signals are first made zero-mean.

$$\mathrm{SI\text{-}SDR} = 10\log_{10}\frac{\|\gamma s\|^2}{\|\gamma s - \hat s\|^2},\quad \gamma = \frac{\langle \hat s, s\rangle}{\|s\|^2} \tag{17}$$

**Segmental SNR**: the mean of the per-frame SNR (512-sample frames, 256 hop), each value clamped to [−10, 35] dB. Frames more than 40 dB below the loudest clean frame are skipped, because the synthetic source contains exact digital silence.

$$\mathrm{SNR_{seg}} = \frac{1}{|\mathcal{F}|}\sum_{f\in\mathcal F} \operatorname{clip}_{[-10,35]}\Big(10\log_{10}\frac{\sum_{n\in f}s^2}{\sum_{n\in f}(s-\hat s)^2}\Big) \tag{18}$$

**Noise reduction** (no reference needed): the mean attenuation over the frames the VAD labels as noise.

$$\mathrm{NR} = \frac{1}{|\mathcal N|}\sum_{m\in\mathcal N}\big(E_y(m) - E_{\hat s}(m)\big)\ \text{dB} \tag{19}$$

**Energy VAD**: frame m counts as speech if

$$E(m) > \operatorname{P}_{20}\{E\} + 3\ \text{dB} \tag{20}$$

with a 3-frame hangover after each active frame. The 3 dB margin was chosen by measuring frame accuracy against the clean reference on the six synthetic noise types: at 0 dB SNR it scores 0.74–0.83, against 0.55–0.74 for a 6 dB margin; at 10 dB SNR both score about 0.72–0.89. Impulsive noise (clicks) is the weakest case. The reference-free SNR estimate is 10 log₁₀((P̄_speech − P̄_noise)/P̄_noise). It is a rough indicator only. For uploaded audio, SNR, SI-SDR and segmental SNR are reported as **n/a (no clean reference)** and never estimated.

## 9. Synthetic test mixtures

The noise is scaled so that the SNR is exact:

$$d' = d\sqrt{\frac{P_s}{P_d\,10^{\mathrm{SNR}/10}}},\qquad y = s + d' \tag{21}$$

If the mixture's peak exceeds 0.9, clean, noise and mixture are all scaled by the same factor, which preserves the SNR.

---

## 10. Simulink vs Python: expected differences

> The Simulink model has **not** been run by the author of this document (no MATLAB on the development machine). The points below are *expected* differences, to be confirmed with `scripts/compare_simulink_python.py`.
>
> **What was measured:** `tests/test_simulink_chain.py` emulates the Simulink block chain frame by frame in Python (the same steps as `simulink/spectral_subtraction_reference.m`). Its output equals the Python STFT pipeline with the "Match Simulink basic" preset to within **3 × 10⁻¹⁶**, after a delay of exactly one hop (256 samples). The block design is therefore algorithmically identical. Remaining differences can only come from the MATLAB implementation details below.
>
> **Pitfall seen while testing the script:** if the two implementations get *different* inputs, for example one reads the 16-bit WAV and the other uses the original float signal, the noise estimate shifts slightly (up to 0.2 % in near-silent bins). That produces differences around 10⁻⁴ in bins where subtraction is active. Always give both implementations the same WAV file.

1. **Latency.** The overlap-add (Selector + Unit Delay + Add) delays the output by exactly one hop, H = 256 samples. `run_and_export.m` removes it. If your release's Buffer block adds further delay, the comparison script finds it by cross-correlation and reports it in samples and hops.
2. **Edge frames.** Simulink's Buffer starts from zero initial conditions, so its first frames are partly zero. Python zero-pads by N − H. The two should match after alignment, except for at most one frame at each end.
3. **Precision.** If the model runs in single precision, expect differences around 10⁻⁷ relative. In double precision they should be negligible.
4. **Noise-estimate frames.** Both average the complete frames starting at 0, H, 2H, … inside the first T s (Eq. 4). Any difference here changes |N̂| and is visible everywhere. Check it first if the outputs disagree.
5. **Exponent.** The basic Simulink chain does magnitude subtraction (p = 1). The "Match Simulink basic" preset therefore uses `exponent = 1`, while the Python default is p = 2 (power subtraction).

---

## 11. Live preview (AudioWorklet, stretch goal)

`app/static/js/dsp/ss_core.js` runs the **basic** algorithm hop by hop in the browser: periodic Hann, 50 % overlap, Eq. (7) with p = 1 or 2, noisy phase, overlap-add with a one-frame delay. It is the same block chain as the Simulink model. `live-worklet.js` wraps it in an AudioWorklet: 128-sample render quanta are collected into hops of H = N/2, and the output FIFO is primed with H zeros.

- **Latency:** H (FIFO priming) + H (overlap-add) = N samples, i.e. 10.7 ms for N = 512 at 48 kHz, plus the audio device's own I/O latency.
- **Noise estimate:** either the recorded noise profile (Eq. 4, computed by the server at the AudioContext's sample rate), or learned from the first T seconds of the live input. Learning uses the same frames as Eq. (4). Until it finishes, the output is a pass-through, because the Hann windows sum to 1.
- **Not implemented live:** the Wiener rule, smoothing, `region` / `minstats` modes, and 75 % overlap.

Verification:
- `tests/test_js_parity.py` runs `ss_core.js` under Node. It matches the Python pipeline to < 10⁻⁹ for N = 256, 512, 1024 and p = 1, 2, after the one-hop latency, including the live-learning path.
- A headless-Chrome check rendered a demo clip through the real `live-worklet.js` in an `OfflineAudioContext`. It matched Python at **93.8 dB SNR between outputs** (max difference 9.6 × 10⁻⁶, float32 audio buffers), with a measured latency of exactly N samples.

---

## 12. Statistical estimators (comparison bench, §14 of CLAUDE.md)

These four methods are **hand-written extensions** in `app/dsp/statistical.py`. They are not part of the Simulink model. They keep every block of the chain above (framing, window, FFT, noisy phase, overlap-add, output stage) and change **only the gain rule** of Eq. (8). Spectral subtraction treats the noise estimate as a fixed amount to remove. These methods treat speech and noise as zero-mean complex Gaussian random variables and pick the gain that minimises an expected error.

References:
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator," *IEEE Trans. ASSP*, 32(6), 1984.
- Y. Ephraim, D. Malah, "… log-spectral amplitude estimator," *IEEE Trans. ASSP*, 33(2), 1985.
- P. Scalart, J. Vieira Filho, "Speech enhancement based on a priori signal to noise estimation," *Proc. ICASSP*, 1996.
- I. Cohen, B. Berdugo, "Speech enhancement for non-stationary noise environments," *Signal Processing*, 81(11), 2001 (OM-LSA).
- I. Cohen, "Noise spectrum estimation in adverse environments: improved minima controlled recursive averaging," *IEEE Trans. SAP*, 11(5), 2003 (IMCRA).
- P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013, chapters 7–9 (textbook treatment of all of the above).

### 12.1 Noise power and the two SNRs

The noise modes of §3 return a mean **magnitude** |N̂_k|. The estimators need the noise **variance** λ_d. A complex-Gaussian noise bin has a Rayleigh magnitude, so:

$$\lambda_{d,k} = \frac{4}{\pi}\,|\hat N_k|^2 \tag{22}$$

Then, per bin k and frame m:

$$\gamma_k(m) = \frac{|Y_k(m)|^2}{\lambda_{d,k}}\qquad\text{(a-posteriori SNR)} \tag{23}$$

$$\xi_k(m) = \max\!\Big(\alpha_{dd}\,G_k^2(m-1)\,\gamma_k(m-1) + (1-\alpha_{dd})\max\big(\gamma_k(m)-1,\,0\big),\;\xi_\min\Big)\qquad\text{(a-priori SNR, decision-directed)} \tag{24}$$

The first term of Eq. (24) is the previous frame's clean-speech estimate |Ŝ(m−1)|²/λ_d. With α_dd ≈ 0.98 it dominates, so ξ changes slowly in noise-only regions. A random peak in |Y| does not create an isolated high-gain bin. **This is the main reason these methods produce much less musical noise than Eq. (7)**, where an isolated peak passes straight through. ξ_min (default −25 dB) limits how far ξ can fall. A smaller ξ_min removes more noise but brings back artefacts. In the first frame there is no previous estimate, so ξ = max(γ − 1, ξ_min).

### 12.2 Gain rules

With v = ξγ / (1 + ξ):

$$G_{\text{Wiener}} = \frac{\xi}{1+\xi} \tag{25}$$

$$G_{\text{STSA}} = \frac{\sqrt{\pi}}{2}\,\frac{\sqrt{v}}{\gamma}\,e^{-v/2}\Big[(1+v)\,I_0\!\big(\tfrac v2\big) + v\,I_1\!\big(\tfrac v2\big)\Big] \tag{26}$$

$$G_{\text{LSA}} = \frac{\xi}{1+\xi}\,\exp\!\Big(\tfrac12\,E_1(v)\Big),\qquad E_1(v)=\int_v^\infty \frac{e^{-t}}{t}\,dt \tag{27}$$

- I₀, I₁ are modified Bessel functions. The code uses the scaled forms `i0e`/`i1e` (= e^{−x}·Iₙ(x)), so the product e^{−v/2}·Iₙ(v/2) never overflows.
- Eq. (26) minimises E[(|S| − |Ŝ|)²]. Eq. (27) minimises E[(log|S| − log|Ŝ|)²], which is closer to how loudness is perceived and attenuates low-SNR bins slightly more.
- At high SNR (v → ∞) both tend to the Wiener gain (unit test).
- Every gain is clipped to [G_min, 1], with G_min = `gain_floor_db` (default −40 dB).

### 12.3 OM-LSA with IMCRA noise tracking

The methods above use a **fixed** noise estimate. OM-LSA also estimates, per bin and frame, the probability p that speech is present, and it updates the noise estimate whenever speech is probably absent.

$$G_{\text{OM-LSA}} = G_{\text{LSA}}^{\,p}\;G_\min^{\,1-p} \tag{28}$$

$$p = \Big[1 + \frac{q}{1-q}\,(1+\xi)\,e^{-v}\Big]^{-1} \tag{29}$$

Here q is the a-priori speech-*absence* probability, estimated by IMCRA:

1. **Smoothing and minimum tracking.** S_f(k) = Σᵢ b(i)·|Y(k−i)|², with b = [¼, ½, ¼]. Then S(m) = α_s S(m−1) + (1−α_s) S_f(m), α_s = 0.9. S_min is the minimum of S over a window of U = 8 sub-windows of V frames (`tracking_window_s` ≈ 1 s):

$$S_k(m) = \alpha_s S_k(m-1) + (1-\alpha_s) \sum_i b(i)\,|Y_{k-i}(m)|^2 \tag{30}$$

2. **Rough VAD.** A bin is treated as noise if γ_min = |Y|²/(B_min S_min) < γ₀ = 4.6 and ζ = S/(B_min S_min) < ζ₀ = 1.67, with B_min = 1.66:

$$I_k(m) = \big[\gamma_{\min,k}(m) < \gamma_0\big]\wedge\big[\zeta_k(m) < \zeta_0\big] \tag{31}$$

3. **Second iteration.** The same smoothing and minimum search are repeated using only the bins with I = 1, giving S̃_min. This keeps strong speech out of the minimum. Then:

$$q_k(m) = \begin{cases}1, & \tilde\gamma_{\min} \le 1,\ \tilde\zeta < \zeta_0\\[2pt] \dfrac{\gamma_1 - \tilde\gamma_{\min}}{\gamma_1 - 1}, & 1 < \tilde\gamma_{\min} < \gamma_1,\ \tilde\zeta < \zeta_0\\[4pt] 0, & \text{otherwise}\end{cases}\qquad \gamma_1 = 3 \tag{32}$$

   The code caps q at 0.99 so that p > 0 and the noise update in Eq. (33) never stops completely.

4. **Noise update.** The update is fast where speech is absent and frozen where it is present:

$$\tilde\alpha_d = \alpha_d + (1-\alpha_d)\,p,\qquad \tilde\lambda_d(m+1) = \tilde\alpha_d\,\tilde\lambda_d(m) + (1-\tilde\alpha_d)\,|Y(m)|^2,\qquad \lambda_d = \beta\,\tilde\lambda_d \tag{33}$$

   Here α_d = 0.85 and β = 1.47, a bias correction. The selected noise mode (§3) only initialises λ_d through Eq. (22). The unit test steps the noise level by +10 dB mid-clip, and IMCRA re-tracks it to within ±2 dB.

Constants are Cohen's (2003, Table I). Only V is derived from the hop, so the search window stays about `tracking_window_s` long at any N and overlap. The trade-off is in the window length. A short window follows changing noise faster, but it can mistake a long sustained vowel for noise.

### 12.4 Attenuation limit (all algorithms)

$$\hat s'[n] = \ell\,y[n] + (1-\ell)\,\hat s[n],\qquad \ell = 10^{-L/20} \tag{34}$$

No bin can be attenuated by more than L dB. For gain-based methods this is exactly G' = ℓ + (1 − ℓ)G, because the STFT is linear. DeepFilterNet's `atten_lim_db` uses the same rule. Applying it to **every** algorithm makes it possible to compare methods at equal maximum suppression. L = 100 dB means "off" (ℓ = 0), and the default reproduces each method unchanged.

### 12.5 First check (6 s synthetic demo, N = 512, 50 % overlap, seed 0)

| ΔSI-SDR (dB) | SS (α=2, β=0.02, p=2) | Wiener-DD | MMSE-STSA | Log-MMSE | OM-LSA |
|---|---|---|---|---|---|
| white, 5 dB | +6.1 | +11.2 | +10.3 | +11.0 | +10.9 |
| pink, 5 dB | +4.2 | +7.4 | +6.6 | +7.2 | +4.8 |
| babble, 5 dB | +4.7 | +9.3 | +8.8 | +9.2 | +8.4 |
| fan, 5 dB | +4.0 | +3.7 | +4.7 | +4.2 | +5.4 |

These numbers come from one seed on the synthetic voiced signal. They are a sanity check, not the evaluation. The report should use the benchmark in `scripts/benchmark.py`: real speech, many seeds, confidence intervals.

## 13. Comparison-bench metrics (§14)

In addition to Eq. (15)–(20):

$$\mathrm{LSD} = \frac{1}{|\mathcal F|}\sum_{m\in\mathcal F}\sqrt{\frac1K\sum_k\big(L_k(m)-\hat L_k(m)\big)^2},\qquad L = \max\!\big(10\log_{10}|S|^2,\ \max_k 10\log_{10}|S_k(m)|^2 - 50\big) \tag{35}$$

Log-spectral distance is a measure of **speech distortion**: 0 dB means the spectrum is unchanged. Speech frames (𝓕) are selected as in Eq. (18). The 50 dB range limit stops empty bins (between harmonics, above the speech band) from dominating. LSD is level-sensitive on purpose: removing noise *is* a level change in noise-dominated bins.

$$\mathrm{SA} = \frac{1}{|\mathcal S|}\sum_{m\in\mathcal S}\big(E_y(m) - E_{\hat s}(m)\big)\ \text{dB} \tag{36}$$

Speech attenuation is the reference-free twin of Eq. (19), averaged over VAD speech frames 𝒮. Plotting noise reduction (Eq. 19) against speech attenuation (Eq. 36) shows each algorithm's trade-off on real recordings, where no clean reference exists.

Third-party metrics (`app/enhancers/quality.py`, optional):
- **PESQ**, ITU-T P.862.2 wide-band (`pesq`): both signals resampled to 16 kHz. MOS-LQO scale, about 1.0–4.64.
- **STOI / ESTOI** (`pystoi`): short-time objective intelligibility, 0–1. ESTOI is stricter for modulated noise.
- **DNSMOS P.835** (Microsoft, ONNX): a CNN trained on listener ratings predicts SIG (speech distortion), BAK (background intrusiveness) and OVRL (overall), each 1–5. It is **non-intrusive**: it needs no clean reference, so it scores uploaded and recorded audio too. It is a *prediction* of a MOS. Use it to rank systems, and never quote it as a listening-test result. The synthetic voiced signal of §9 gets SIG ≈ 2.2 even when clean, because it is not real speech. This is one reason the benchmark uses real speech. The reference script slides a 9.01 s window by 1 s. The app uses non-overlapping windows (≈ 9× faster). The benchmark's short utterances fit in one window, so the two agree there.
