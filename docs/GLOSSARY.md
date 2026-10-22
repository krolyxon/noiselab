# Glossary: the scores and words in plain language

_Generated from `app/glossary.py` by `scripts/export_glossary.py`. Do not edit by hand._

The report figures, `results/summary.md` and the web app all use these words. Each term has a **plain name** first and the **standard technical name** second, so you can explain a graph simply and still match the literature.

## What the scores mean

| Plain name | Technical name | Better | Meaning | Think of it as |
|---|---|---|---|---|
| **Starting noise level** | input SNR | — | How loud the speech is compared with the noise before cleaning. Lower = noisier: 0 dB = equally loud, −5 dB = noise louder than speech, 20 dB = only faint noise. |  |
| **Speech clarity gain** | ΔSI-SDR | ↑ higher | How much closer the cleaned waveform is to the clean speech than the noisy input was. +3 dB means the remaining error energy was halved; +10 dB means it is 10× smaller. | Like counting spelling mistakes before and after proof-reading, on the waveform. |
| **Signal-to-noise gain** | ΔSNR | ↑ higher | How much louder the speech is than the leftover noise, compared with before cleaning. | How much easier it became to hear the speaker over the background. |
| **Frame-by-frame signal-to-noise** | segmental SNR | ↑ higher | Signal-to-noise ratio measured on short pieces and averaged, so quiet words count too. |  |
| **Sound-quality score** | PESQ | ↑ higher | An ITU-standard program that predicts how listeners would rate the quality, on a 1 (bad) to 4.5 (excellent) scale. Needs the clean speech for comparison. | An automatic judge trained to grade audio the way people do. |
| **Sound-quality improvement** | ΔPESQ | ↑ higher | How many points the sound-quality score (PESQ, 1–4.5) went up after cleaning. | Going from a 1.5-star to a 2.5-star recording is +1.0. |
| **Intelligibility score** | ESTOI | ↑ higher | Predicts what fraction of the words a listener would understand (0 to 1). Needs the clean speech for comparison. | 0.8 ≈ about 8 words in 10 understood. |
| **Intelligibility improvement** | ΔESTOI | ↑ higher | How much the predicted fraction of understood words went up (0.1 = 10 percentage points). |  |
| **Speech distortion** | LSD = log-spectral distance | ↓ lower | How much the shape of the speech spectrum differs from the clean speech. Noise left over and speech damaged both increase it. | How much the 'colour' of the voice was changed. |
| **Noise removed in pauses** | NR = noise reduction | ↑ higher | How much quieter the gaps between words became. Does not say whether the speech survived. | Turning down the background between sentences. |
| **Speech lost** | speech attenuation | ↓ lower | How much quieter the speech itself became: the damage side of noise removal. | Accidentally turning down the speaker too. |
| **AI listener score (overall)** | DNSMOS OVRL | ↑ higher | A neural network trained on thousands of human ratings guesses the overall rating (1–5). Needs no clean reference, so it works on your own recordings, but it is a prediction. | Asking a trained AI critic instead of a panel of listeners. |
| **AI listener score (speech)** | DNSMOS SIG | ↑ higher | Predicted rating of how natural and undamaged the speech sounds (1–5). |  |
| **AI listener score (background)** | DNSMOS BAK | ↑ higher | Predicted rating of how unobtrusive the background noise is (1–5). |  |
| **AI listener score improvement** | ΔDNSMOS OVRL | ↑ higher | How much the AI listener's overall rating went up after cleaning. |  |
| **CPU cost** | RTF = real-time factor | ↓ lower | Processing time ÷ audio length on one CPU core. 0.01 = 1 % of one core; above 1 = cannot run live. | 0.02 means 1 minute of audio is cleaned in about 1.2 seconds. |

## Other words on the graphs

| Plain name | Technical name | Meaning | Think of it as |
|---|---|---|---|
| **± uncertainty** | 95 % confidence interval | The true average very likely (95 %) lies within ± this value. Overlapping ranges = no clear winner. |  |
| **Is the difference real?** | Wilcoxon signed-rank p-value | Tests on the same clips whether the winner is really better. p < 0.05 = yes, very likely; p ≥ 0.05 = could be chance. | Like checking that a cricketer's higher average is not just luck from a few innings. |
| **Best value-for-cost line** | Pareto front | Methods on this line are not beaten by anything that is both cheaper and better. |  |
| **Gain the model actually applied** | effective gain mask | For ready-made models that do not report their gain, it is measured as output ÷ input in each time–frequency cell (0 = removed, 1 = kept). |  |
| **Keep/remove map** | gain mask G(k, m) | For every time and frequency: 1 = keep, 0 = remove. The algorithm's decision, as a picture. |  |
| **Steady vs changing noise** | stationary / non-stationary | Steady noise (fan, hum, hiss) sounds the same all the time; changing noise (babble, clicks, traffic) does not. Classical methods assume steady noise. |  |
| **Musical noise** | musical noise | Short random tones ('twinkling') left behind when noise is removed bin by bin. |  |

## Types of method

| Type | Technical name | Meaning |
|---|---|---|
| **Classical: our Simulink method** | spectral subtraction | Measure the average noise spectrum and subtract it. |
| **Statistical: probability-based (hand-written)** | MMSE / Wiener estimators | Smarter subtraction that weighs how likely each piece is to be speech. |
| **Ready-made DSP library** | DSP baselines | Well-known noise suppressors from existing software (no neural network). |
| **Neural network (pretrained)** | deep learning | A network trained on many hours of noisy speech learns what to remove. |

## The 11 methods in one line each

| Method id | What it does |
|---|---|
| `spectral_subtraction` | Measures the average noise in the first half-second and subtracts it from every frame. Our Simulink method. |
| `wiener_dd` | Like subtraction, but turns each frequency down by a smooth fraction based on how likely it is to be speech. Less 'twinkling' leftover noise. |
| `mmse_stsa` | Probability-based: picks the gain that makes the cleaned loudness of each frequency closest to the clean speech, on average. |
| `log_mmse` | Same idea as MMSE-STSA, but matches loudness on a log (decibel) scale, closer to how we hear. |
| `omlsa` | Log-MMSE that also keeps re-measuring the noise while you talk, so it follows noise that changes. |
| `noisereduce` | A noise gate per frequency: anything below a learned noise threshold is muted. |
| `webrtc_ns` | The noise suppressor built into Chrome / Google Meet calls (classic signal processing). |
| `rnnoise` | A very small neural network (used in Jitsi, OBS) that turns 22 frequency bands up or down. |
| `nsnet2` | Microsoft's neural network for noise suppression (from their DNS Challenge). |
| `deepfilternet3` | A modern neural network that also repairs the harmonics of the voice. Runs live on a laptop. |
| `demucs` | Meta's large neural network that rebuilds the clean waveform directly. Best quality, most CPU. |

## Knobs in the parameter sweep (figure 7)

| Parameter | Plain meaning |
|---|---|
| `alpha` | α: how much noise to subtract |
| `dd_alpha` | α_dd: how smooth the decision is over time |
| `g_min_db` | G_min: how much to turn down noise-only parts (dB) |
| `prop_decrease` | how much of the gated noise to remove |
| `level` | suppression strength (1 mild … 4 very high) |
| `atten_limit_db` | max. noise removal allowed (dB) |
