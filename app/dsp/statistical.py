"""Statistical (MMSE-family) speech enhancement, Eq. (22)–(33) in docs/ALGORITHM.md.

All four methods share the steps of spectral subtraction: the same STFT, a real gain
G_k(m) applied to the noisy spectrum, the noisy phase and the same overlap-add. Only
the gain rule differs. Instead of subtracting a noise estimate, these methods treat
speech and noise as random variables and choose G to minimise an expected error,
given two SNR estimates per bin:

    a-posteriori SNR  γ = |Y|² / λ_d                         Eq. (23)
    a-priori SNR      ξ = E|S|² / λ_d   (decision-directed)  Eq. (24)

    wiener_dd   G = ξ / (1 + ξ)                              Eq. (25)  Scalart & Vieira Filho 1996
    mmse_stsa   minimises E(|S| − |Ŝ|)²                      Eq. (26)  Ephraim & Malah 1984
    log_mmse    minimises E(log|S| − log|Ŝ|)²                Eq. (27)  Ephraim & Malah 1985
    omlsa       log-MMSE weighted by speech-presence prob.,
                with IMCRA adaptive noise tracking           Eq. (28)–(33)  Cohen 2002, 2003

Pure NumPy/SciPy, hand-written. No Flask, no third-party denoisers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import exp1, i0e, i1e

STATISTICAL_METHODS = ("wiener_dd", "mmse_stsa", "log_mmse", "omlsa")
_EPS = 1e-12
_V_MIN = 1e-10          # keeps E1(v) and √v finite when a bin is (numerically) empty


def db_to_power(db: float) -> float:
    return float(10.0 ** (db / 10.0))


def db_to_amplitude(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def noise_power_from_mag(noise_mag: np.ndarray) -> np.ndarray:
    """Eq. (22): λ_d = (4/π)·|N̂|².

    The noise estimators of Eq. (4)–(6) return the *mean magnitude* E|D|. The
    statistical gains need the *variance* E|D|². For a complex-Gaussian noise bin the
    magnitude is Rayleigh distributed, and E|D|² = (4/π)·(E|D|)².
    """
    return (4.0 / np.pi) * np.asarray(noise_mag, dtype=np.float64) ** 2


# --- gain functions (vectorised over bins) --------------------------------------------------

def wiener_gain(xi: np.ndarray) -> np.ndarray:
    """Eq. (25): G = ξ / (1 + ξ)."""
    return xi / (1.0 + xi)


def mmse_stsa_gain(xi: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Eq. (26): Ephraim–Malah short-time spectral amplitude estimator.

    G = (√π / 2) · (√v / γ) · exp(−v/2) · [ (1 + v)·I₀(v/2) + v·I₁(v/2) ],   v = ξγ/(1+ξ)

    exp(−v/2)·Iₙ(v/2) is evaluated as the exponentially scaled Bessel function
    iₙe(v/2), which never overflows. For large v, G → ξ/(1+ξ) (the Wiener gain).
    """
    v = np.maximum(xi * gamma / (1.0 + xi), _V_MIN)
    g = (np.sqrt(np.pi) / 2.0) * (np.sqrt(v) / np.maximum(gamma, _EPS)) \
        * ((1.0 + v) * i0e(v / 2.0) + v * i1e(v / 2.0))
    return g


def log_mmse_gain(xi: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Eq. (27): Ephraim–Malah log-spectral amplitude estimator.

    G = ξ/(1+ξ) · exp( ½ · E₁(v) ),   E₁(v) = ∫_v^∞ e^{−t}/t dt  (exponential integral)
    """
    v = np.maximum(xi * gamma / (1.0 + xi), _V_MIN)
    return xi / (1.0 + xi) * np.exp(0.5 * exp1(v))


# --- decision-directed estimators (fixed noise estimate) ------------------------------------

@dataclass(frozen=True)
class DDParams:
    dd_alpha: float = 0.98      # α_dd: weight of the previous frame's clean estimate
    xi_min_db: float = -25.0    # ξ_min: floor on the a-priori SNR (limits musical noise)
    gain_floor_db: float = -40.0  # G_min: lower limit of the gain (residual-noise level)


def decision_directed_gain(power: np.ndarray, noise_power: np.ndarray, method: str,
                           params: DDParams) -> np.ndarray:
    """Gain mask G(k, m) of wiener_dd / mmse_stsa / log_mmse, frame by frame.

    power        |Y_k(m)|², shape (n_frames, n_bins)
    noise_power  λ_d,k, shape (n_bins,) — fixed over time (from Eq. 4–6 via Eq. 22)

    Eq. (24), decision-directed a-priori SNR (Ephraim & Malah 1984):
        ξ(m) = max( α_dd · |Ŝ(m−1)|²/λ_d + (1 − α_dd) · max(γ(m) − 1, 0),  ξ_min )
    where |Ŝ(m−1)|²/λ_d = G(m−1)² · γ(m−1). The first term (the previous frame's
    clean estimate) dominates for α_dd ≈ 0.98, which smooths ξ over time. That
    smoothing is why these methods produce far less musical noise than Eq. (7).
    """
    if method not in ("wiener_dd", "mmse_stsa", "log_mmse"):
        raise ValueError(f"unknown decision-directed method {method!r}")
    gain_fn = {"wiener_dd": lambda xi, g: wiener_gain(xi),
               "mmse_stsa": mmse_stsa_gain, "log_mmse": log_mmse_gain}[method]
    lam = np.maximum(np.asarray(noise_power, dtype=np.float64), _EPS)
    gamma = power / lam[None, :]                                        # Eq. (23)
    a, xi_min = float(params.dd_alpha), db_to_power(params.xi_min_db)
    g_min = db_to_amplitude(params.gain_floor_db)

    gain = np.empty_like(gamma)
    prev_clean = np.zeros(gamma.shape[1])   # |Ŝ(m−1)|²/λ_d; 0 before the first frame
    for m in range(gamma.shape[0]):
        g_m = gamma[m]
        xi = np.maximum(a * prev_clean + (1.0 - a) * np.maximum(g_m - 1.0, 0.0), xi_min)  # Eq. (24)
        if m == 0:                         # no previous frame: maximum-likelihood start
            xi = np.maximum(np.maximum(g_m - 1.0, 0.0), xi_min)
        g = np.clip(gain_fn(xi, g_m), g_min, 1.0)
        gain[m] = g
        prev_clean = g * g * g_m
    return gain


# --- OM-LSA with IMCRA noise tracking (Cohen 2002, 2003) -------------------------------------

@dataclass(frozen=True)
class OmlsaParams:
    dd_alpha: float = 0.92          # α in Eq. (24) (Cohen's value for OM-LSA)
    g_min_db: float = -25.0         # G_min: gain applied when speech is absent, Eq. (28)
    tracking_window_s: float = 1.0  # length of IMCRA's minimum-search window (U·V frames)
    xi_min_db: float = -25.0


@dataclass(frozen=True)
class OmlsaResult:
    gain: np.ndarray                # G(k, m), shape (n_frames, n_bins)
    noise_power: np.ndarray         # λ_d(k, m) used in each frame, shape (n_frames, n_bins)
    speech_prob: np.ndarray         # p(k, m), speech-presence probability, same shape


# IMCRA constants (Cohen 2003, Table I): independent of the sample rate.
_ALPHA_S = 0.9      # recursive smoothing of the power spectrum, Eq. (30)
_ALPHA_D = 0.85     # noise-update smoothing, Eq. (33)
_B_MIN = 1.66       # bias of the minimum (for the chosen window length)
_GAMMA0 = 4.6       # rough-VAD threshold on γ_min, Eq. (31)
_GAMMA1 = 3.0       # upper threshold for the soft speech-absence probability, Eq. (32)
_ZETA0 = 1.67       # rough-VAD threshold on ζ, Eq. (31)
_BETA = 1.47        # bias compensation of the noise estimate, Eq. (33)
_U = 8              # number of sub-windows in the minimum search
_FREQ_WIN = np.array([0.25, 0.5, 0.25])   # b(i): 3-tap Hann smoothing across frequency


def _freq_smooth(x: np.ndarray) -> np.ndarray:
    """Eq. (30), first part: S_f(k) = Σ_i b(i)·x(k − i), edges reflected."""
    xp = np.concatenate(([x[1]], x, [x[-2]]))
    return _FREQ_WIN[0] * xp[:-2] + _FREQ_WIN[1] * xp[1:-1] + _FREQ_WIN[2] * xp[2:]


def omlsa_gain(power: np.ndarray, init_noise_power: np.ndarray, hop_s: float,
               params: OmlsaParams) -> OmlsaResult:
    """OM-LSA gain with IMCRA noise tracking, frame by frame (Eq. 28–33).

    power             |Y_k(m)|², shape (n_frames, n_bins)
    init_noise_power  λ_d at the first frame (from the selected noise mode, Eq. 22).
                      After that, IMCRA tracks the noise by itself, so the
                      estimate follows noise whose level changes over time.
    hop_s             frame hop in seconds, which sets V = window / (U · hop)
    """
    n_frames, n_bins = power.shape
    a_dd = float(params.dd_alpha)
    xi_min = db_to_power(params.xi_min_db)
    g_min = db_to_amplitude(params.g_min_db)
    v_len = max(2, int(round(params.tracking_window_s / (_U * hop_s))))  # frames per sub-window

    gain = np.empty_like(power)
    noise_out = np.empty_like(power)
    prob_out = np.empty_like(power)

    lam_tilde = np.maximum(np.asarray(init_noise_power, dtype=np.float64), _EPS) / _BETA
    lam = _BETA * lam_tilde
    g_prev = np.ones(n_bins)
    gamma_prev = np.ones(n_bins)

    s = s_min = s_tmp = None
    st = st_min = st_tmp = None
    buf: list[np.ndarray] = []
    buf_t: list[np.ndarray] = []
    j = 0

    for m in range(n_frames):
        y2 = power[m]

        # --- gain with the current noise estimate (Eq. 23, 24, 27) ------------------------
        gamma = y2 / np.maximum(lam, _EPS)
        if m == 0:
            xi = np.maximum(gamma - 1.0, xi_min)
        else:
            xi = np.maximum(a_dd * g_prev**2 * gamma_prev + (1.0 - a_dd) * np.maximum(gamma - 1.0, 0.0),
                            xi_min)
        v = np.maximum(xi * gamma / (1.0 + xi), _V_MIN)
        g_h1 = np.minimum(log_mmse_gain(xi, gamma), 1.0)

        # --- IMCRA, first iteration: smoothed spectrum + minimum tracking (Eq. 30, 31) ----
        s_f = _freq_smooth(y2)
        if m == 0:
            s = s_min = s_tmp = s_f.copy()
        else:
            s = _ALPHA_S * s + (1.0 - _ALPHA_S) * s_f
            s_min = np.minimum(s_min, s)
            s_tmp = np.minimum(s_tmp, s)
        # Rough VAD: a bin is "noise" if it is close to the tracked minimum.
        gamma_min = y2 / (_B_MIN * np.maximum(s_min, _EPS))
        zeta = s / (_B_MIN * np.maximum(s_min, _EPS))
        absent = (gamma_min < _GAMMA0) & (zeta < _ZETA0)

        # --- IMCRA, second iteration: smoothing over speech-absent bins only -------------
        num = _freq_smooth(absent * y2)
        den = _freq_smooth(absent.astype(np.float64))
        if m == 0:
            st_f = np.where(den > 0, num / np.maximum(den, _EPS), s)
            st = st_min = st_tmp = st_f.copy()
        else:
            st_f = np.where(den > 0, num / np.maximum(den, _EPS), st)
            st = _ALPHA_S * st + (1.0 - _ALPHA_S) * st_f
            st_min = np.minimum(st_min, st)
            st_tmp = np.minimum(st_tmp, st)

        # --- speech-absence probability q̂ (Eq. 32) -----------------------------------------
        gamma_min_t = y2 / (_B_MIN * np.maximum(st_min, _EPS))
        zeta_t = s / (_B_MIN * np.maximum(st_min, _EPS))
        q = np.zeros(n_bins)
        low = zeta_t < _ZETA0
        q[low & (gamma_min_t <= 1.0)] = 1.0
        mid = low & (gamma_min_t > 1.0) & (gamma_min_t < _GAMMA1)
        q[mid] = (_GAMMA1 - gamma_min_t[mid]) / (_GAMMA1 - 1.0)
        q = np.minimum(q, 0.99)            # keep p > 0 so the noise update never freezes

        # Speech-presence probability, Eq. (29): p = [1 + q/(1−q)·(1+ξ)·e^{−v}]⁻¹
        p = 1.0 / (1.0 + q / (1.0 - q) * (1.0 + xi) * np.exp(-v))

        # --- OM-LSA gain, Eq. (28): G = G_H1^p · G_min^(1−p) -------------------------------
        g = g_h1**p * g_min ** (1.0 - p)
        gain[m] = g
        noise_out[m] = lam
        prob_out[m] = p

        # --- noise update for the next frame, Eq. (33) --------------------------------------
        a_d = _ALPHA_D + (1.0 - _ALPHA_D) * p
        lam_tilde = a_d * lam_tilde + (1.0 - a_d) * y2
        lam = _BETA * lam_tilde

        # --- minimum-search window bookkeeping (U sub-windows of V frames) ----------------
        j += 1
        if j == v_len:
            j = 0
            buf.append(s_tmp.copy())
            buf_t.append(st_tmp.copy())
            if len(buf) > _U:
                buf.pop(0)
                buf_t.pop(0)
            s_min = np.minimum.reduce(buf + [s])
            st_min = np.minimum.reduce(buf_t + [st])
            s_tmp, st_tmp = s.copy(), st.copy()

        g_prev, gamma_prev = g_h1, gamma

    return OmlsaResult(gain=gain, noise_power=noise_out, speech_prob=prob_out)
