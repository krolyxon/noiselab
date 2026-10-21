#!/usr/bin/env python3
"""Figures and a statistics summary for the report, from scripts/benchmark.py output (CLAUDE.md §14).

    .venv-ml/bin/python scripts/report_figures.py                 # results/bench.csv (+ sweep.csv)
    .venv-ml/bin/python scripts/report_figures.py --bench results/bench_quick.csv --no-spectrograms

Writes results/figures/*.png and results/summary.md. The summary contains only numbers and
statements computed from them (means ± 95 % CI, winners, paired Wilcoxon tests). The
written conclusions belong in the report and must follow from these numbers.

Colour encodes the algorithm *family* (4 validated categorical slots). Line style and marker
separate the members of a family, and every line is also labelled directly, so identity never
rests on colour alone. Spectrograms use the perceptually uniform 'inferno' map.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"

# --- visual system (dataviz reference palette, light mode; validated: 4 slots pass) -----------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FAMILY_COLOR = {"classical": "#2a78d6", "statistical": "#eb6834", "library": "#1baf7a", "deep": "#eda100"}
FAMILY_NAME = {"classical": "Classical (Simulink SS)", "statistical": "Statistical (hand-written)",
               "library": "DSP library", "deep": "Deep learning"}
STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D")]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"

LABEL = {"ss_simulink": "SS (Simulink)", "wiener_dd": "Wiener-DD", "mmse_stsa": "MMSE-STSA",
         "log_mmse": "Log-MMSE", "omlsa": "OM-LSA", "noisereduce": "noisereduce", "webrtc_ns": "WebRTC NS",
         "rnnoise": "RNNoise", "nsnet2": "NSNet2", "deepfilternet3": "DeepFilterNet3", "demucs": "Demucs"}
ORDER = list(LABEL)
NOISE_LABEL = {"white": "white", "pink": "pink", "hum50": "50 Hz hum", "fan": "fan/AC", "babble": "babble",
               "clicks": "clicks", "DLIVING": "living room", "PCAFETER": "cafeteria", "STRAFFIC": "street",
               "TCAR": "car", "OOFFICE": "office"}
NOISE_ORDER = list(NOISE_LABEL)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
    })
    return plt


# --- data --------------------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("config", "algorithm", "family", "noise", "noise_source", "utt", "speaker", "sex",
                     "speech_source", "param"):
                continue
            try:
                r[k] = float(v) if v != "" else math.nan
            except ValueError:
                pass
        r["d_si_sdr"] = r["si_sdr_out"] - r["si_sdr_in"]
        r["d_snr"] = r["snr_out"] - r["snr_in"]
        r["d_pesq"] = r["pesq_out"] - r["pesq_in"]
        r["d_estoi"] = r["estoi_out"] - r["estoi_in"]
        r["d_stoi"] = r["stoi_out"] - r["stoi_in"]
        r["d_dnsmos"] = r.get("dnsmos_ovrl_out", math.nan) - r.get("dnsmos_ovrl_in", math.nan)
    return rows


def configs_in(rows) -> list[str]:
    present = {r["config"] for r in rows}
    return [c for c in ORDER if c in present]


def family_of(rows) -> dict[str, str]:
    return {r["config"]: r["family"] for r in rows}


def style_of(configs: list[str], fam: dict[str, str]) -> dict[str, tuple]:
    seen: dict[str, int] = defaultdict(int)
    out = {}
    for c in configs:
        f = fam[c]
        ls, mk = STYLES[seen[f] % len(STYLES)]
        seen[f] += 1
        out[c] = (FAMILY_COLOR[f], ls, mk)
    return out


def mean_ci(v) -> tuple[float, float, int]:
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return math.nan, math.nan, 0
    if len(v) == 1:
        return float(v[0]), math.nan, 1
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), len(v)


def group(rows, keys, metric):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r[metric])
    return {k: mean_ci(v)[0] for k, v in g.items()}


def noise_class(r) -> str:
    if r["noise_source"] == "DEMAND":
        return "Real noise (DEMAND)"
    return "Stationary synthetic" if r["stationary"] == 1 else "Non-stationary synthetic"


CLASSES = ("Stationary synthetic", "Non-stationary synthetic", "Real noise (DEMAND)")


def _family_legend(ax_or_fig, fams, **kw):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=FAMILY_COLOR[f], lw=3) for f in fams]
    ax_or_fig.legend(h, [FAMILY_NAME[f] for f in fams], **kw)


def _direct_labels(ax, ends: list[tuple[float, float, str, str]], x_pad: float):
    """Right-edge labels, nudged apart vertically so they do not collide."""
    if not ends:
        return
    ends = sorted(ends, key=lambda e: e[1])
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * 0.045
    ys = []
    for _, y, _, _ in ends:
        ys.append(max(y, ys[-1] + min_gap) if ys else y)
    for (x, _, text, color), y in zip(ends, ys):
        ax.annotate(text, (x, y), xytext=(x + x_pad, y), fontsize=7, color=INK2, va="center",
                    annotation_clip=False)
        ax.plot([x + x_pad * 0.15], [y], marker="o", ms=3, color=color, clip_on=False)


# --- figures -----------------------------------------------------------------------------------

def fig_metric_vs_snr(rows, metric, ylabel, title, path, plt):
    cfgs = configs_in(rows)
    fam = family_of(rows)
    sty = style_of(cfgs, fam)
    classes = [c for c in CLASSES if any(noise_class(r) == c for r in rows)]
    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes) + 1.2, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    snrs = sorted({r["snr_db"] for r in rows})
    for ax, cls in zip(axes, classes):
        sub = [r for r in rows if noise_class(r) == cls]
        g = group(sub, ("config", "snr_db"), metric)
        ends = []
        for c in cfgs:
            ys = [g.get((c, s), math.nan) for s in snrs]
            color, ls, mk = sty[c]
            ax.plot(snrs, ys, ls=ls, marker=mk, color=color, label=LABEL[c], ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1)
            if np.isfinite(ys[-1]):
                ends.append((snrs[-1], ys[-1], LABEL[c], color))
        ax.axhline(0, color=AXIS, lw=1, zorder=0)
        ax.set_title(cls)
        ax.set_xlabel("input SNR (dB)")
        ax.set_xticks(snrs)
        _direct_labels(ax, ends, x_pad=(snrs[-1] - snrs[0]) * 0.04 if len(snrs) > 1 else 0.5)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(cfgs), 6), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=11, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 0.93, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pesq_estoi(rows, path, plt):
    cfgs = configs_in(rows)
    sty = style_of(cfgs, family_of(rows))
    snrs = sorted({r["snr_db"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (out_m, in_m, ylab, ttl) in zip(axes, (
            ("pesq_out", "pesq_in", "PESQ (WB, MOS-LQO)", "Perceptual quality (PESQ)"),
            ("estoi_out", "estoi_in", "ESTOI", "Intelligibility (ESTOI)"))):
        g = group(rows, ("config", "snr_db"), out_m)
        gi = group(rows, ("snr_db",), in_m)
        ax.plot(snrs, [gi.get((s,), math.nan) for s in snrs], color=MUTED, ls="--", lw=1.5,
                label="unprocessed input")
        ends = [(snrs[-1], gi.get((snrs[-1],), math.nan), "unprocessed", MUTED)]
        for c in cfgs:
            ys = [g.get((c, s), math.nan) for s in snrs]
            color, ls, mk = sty[c]
            ax.plot(snrs, ys, ls=ls, marker=mk, color=color, label=LABEL[c],
                    markeredgecolor=SURFACE, markeredgewidth=1)
            ends.append((snrs[-1], ys[-1], LABEL[c], color))
        ax.set_title(ttl)
        ax.set_xlabel("input SNR (dB)")
        ax.set_ylabel(ylab)
        ax.set_xticks(snrs)
        _direct_labels(ax, [e for e in ends if np.isfinite(e[1])], (snrs[-1] - snrs[0]) * 0.04 or 0.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Output quality vs input SNR (all noise types pooled)", fontsize=11, fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 0.92, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_winner_grid(rows, path, plt):
    """Which configuration wins each (noise, SNR) cell — by ΔSI-SDR and by ΔPESQ."""
    from matplotlib.patches import Rectangle
    fam = family_of(rows)
    noises = [n for n in NOISE_ORDER if any(r["noise"] == n for r in rows)]
    snrs = sorted({r["snr_db"] for r in rows})
    metrics = [("d_si_sdr", "ΔSI-SDR")] + ([("d_pesq", "ΔPESQ")] if any(np.isfinite(r["d_pesq"]) for r in rows)
                                            else [])
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 0.42 * len(noises) + 1.8))
    axes = np.atleast_1d(axes)
    for ax, (m, mlabel) in zip(axes, metrics):
        g = group(rows, ("noise", "snr_db", "config"), m)
        for i, n in enumerate(noises):
            for j, s in enumerate(snrs):
                cand = [(v, c) for (nn, ss, c), v in g.items() if nn == n and ss == s and np.isfinite(v)]
                if not cand:
                    continue
                v, c = max(cand)
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=FAMILY_COLOR[fam[c]], edgecolor=SURFACE, lw=2,
                                       alpha=0.85))
                ax.text(j + 0.5, i + 0.5, f"{LABEL[c]}\n{v:+.1f}", ha="center", va="center", fontsize=6.2,
                        color=INK)
        ax.set_xlim(0, len(snrs))
        ax.set_ylim(len(noises), 0)
        ax.set_xticks(np.arange(len(snrs)) + 0.5, [f"{s:g}" for s in snrs])
        ax.set_yticks(np.arange(len(noises)) + 0.5, [NOISE_LABEL[n] for n in noises])
        ax.set_xlabel("input SNR (dB)")
        ax.grid(False)
        ax.set_title(f"Best by {mlabel} (cell: winner, mean improvement)")
    _family_legend(fig, sorted({fam[c] for c in fam}, key=list(FAMILY_COLOR).index), loc="lower center",
                   ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(rows, metric, title, path, plt, fmt="{:+.1f}"):
    """Configuration × noise table of the mean metric (sequential blue, values printed)."""
    from matplotlib.colors import LinearSegmentedColormap
    cfgs = configs_in(rows)
    noises = [n for n in NOISE_ORDER if any(r["noise"] == n for r in rows)]
    g = group(rows, ("config", "noise"), metric)
    z = np.array([[g.get((c, n), math.nan) for n in noises] for c in cfgs])
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(0.8 * len(noises) + 2.4, 0.4 * len(cfgs) + 1.4))
    im = ax.imshow(z, cmap=cmap, aspect="auto")
    lo, hi = np.nanmin(z), np.nanmax(z)
    for i in range(len(cfgs)):
        for j in range(len(noises)):
            if np.isfinite(z[i, j]):
                dark = (z[i, j] - lo) / (hi - lo + 1e-12) > 0.55
                ax.text(j, i, fmt.format(z[i, j]), ha="center", va="center", fontsize=7,
                        color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(noises)), [NOISE_LABEL[n] for n in noises], rotation=35, ha="right")
    ax.set_yticks(range(len(cfgs)), [LABEL[c] for c in cfgs])
    ax.grid(False)
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _scatter_labelled(ax, pts, fam):
    """Points coloured by family, each with a direct label placed where it overlaps nothing.

    Greedy: for each point try a few offsets (right-up, right-down, left-up, left-down,
    further out) and keep the first whose box does not overlap a box placed earlier.
    """
    for c, x, y in pts:
        ax.scatter([x], [y], s=60, color=FAMILY_COLOR[fam[c]], edgecolor=SURFACE, linewidth=2, zorder=3)
    fig = ax.figure
    ax.autoscale_view()
    lim = (ax.get_xlim(), ax.get_ylim())
    ax.set_xlim(*lim[0])                     # freeze the limits so the boxes stay valid
    ax.set_ylim(*lim[1])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Reserve each marker's own footprint so no label covers a point.
    from matplotlib.transforms import Bbox
    placed = [Bbox.from_bounds(*(ax.transData.transform((x, y)) - 5), 10, 10) for _, x, y in pts]
    offsets = [(6, 4, "left"), (6, -11, "left"), (-6, 4, "right"), (-6, -11, "right"),
               (6, 14, "left"), (6, -21, "left"), (-6, 14, "right"), (-6, -21, "right")]
    for c, x, y in sorted(pts, key=lambda p: -p[2]):
        for dx, dy, ha in offsets:
            t = ax.annotate(LABEL[c], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                            color=INK2, ha=ha)
            bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                break
            t.remove()
        else:
            t = ax.annotate(LABEL[c], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7.5,
                            color=INK2)
            placed.append(t.get_window_extent(renderer))


def fig_tradeoff(rows, path, plt):
    """Noise removed vs speech damaged: every method trades one for the other."""
    cfgs = configs_in(rows)
    fam = family_of(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    g_nr = group(rows, ("config",), "noise_reduction_db")
    g_sa = group(rows, ("config",), "speech_attenuation_db")
    g_lsd = group(rows, ("config",), "lsd_out")
    _scatter_labelled(axes[0], [(c, g_nr[(c,)], g_sa[(c,)]) for c in cfgs], fam)
    axes[0].set_xlabel("noise reduction in noise-only frames (dB)  →  more noise removed")
    axes[0].set_ylabel("attenuation of speech frames (dB)  →  more speech lost")
    axes[0].set_title("Reference-free trade-off (Eq. 19 vs Eq. 36)")
    lsd_in = mean_ci([r["lsd_in"] for r in rows])[0]
    axes[1].axhline(lsd_in, color=MUTED, ls="--", lw=1)          # before labelling: it changes the y-limits
    _scatter_labelled(axes[1], [(c, g_nr[(c,)], g_lsd[(c,)]) for c in cfgs], fam)
    axes[1].annotate("unprocessed input", (axes[1].get_xlim()[0], lsd_in), xytext=(4, 4),
                     textcoords="offset points", fontsize=7, color=MUTED)
    axes[1].set_xlabel("noise reduction in noise-only frames (dB)")
    axes[1].set_ylabel("log-spectral distance to clean speech (dB, Eq. 35)")
    axes[1].set_title("Spectral distortion of the output (lower is better)")
    _family_legend(fig, sorted({fam[c] for c in cfgs}, key=list(FAMILY_COLOR).index), loc="lower center",
                   ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cost(rows, path, plt):
    """Quality bought per unit of compute: ΔPESQ (or ΔSI-SDR) vs single-core real-time factor."""
    cfgs = configs_in(rows)
    fam = family_of(rows)
    metric, mlabel = ("d_pesq", "mean ΔPESQ") if any(np.isfinite(r["d_pesq"]) for r in rows) \
        else ("d_si_sdr", "mean ΔSI-SDR (dB)")
    g_q = group(rows, ("config",), metric)
    g_r = {c: float(np.median([r["rtf"] for r in rows if r["config"] == c])) for c in cfgs}
    pts = [(c, g_r[c], g_q[(c,)]) for c in cfgs]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # Pareto front: no other method is both cheaper and better.
    front = sorted([p for p in pts if not any(q[1] <= p[1] and q[2] > p[2] for q in pts if q is not p)],
                   key=lambda p: p[1])
    ax.plot([p[1] for p in front], [p[2] for p in front], color=AXIS, lw=1.2, ls="--", zorder=1,
            label="Pareto front")
    ax.axvline(1.0, color=MUTED, lw=1, ls=":")
    ax.set_xscale("log")
    _scatter_labelled(ax, pts, fam)
    ax.annotate("real time on one core", (1.0, ax.get_ylim()[0]), xytext=(-4, 4), textcoords="offset points",
                fontsize=7, color=MUTED, ha="right")
    ax.set_xlabel("real-time factor, single CPU core (processing time / audio duration, log scale)")
    ax.set_ylabel(mlabel)
    ax.set_title("Quality vs computational cost", loc="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _family_legend(fig, sorted({fam[c] for c in cfgs}, key=list(FAMILY_COLOR).index), loc="lower center",
                   ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_stationarity(rows, path, plt):
    cfgs = configs_in(rows)
    g = group(rows, ("config", "stationary"), "d_si_sdr")
    y = np.arange(len(cfgs))
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(cfgs) + 1.4))
    h = 0.38
    for k, (st, name, color) in enumerate(((1.0, "stationary noise", "#2a78d6"),
                                           (0.0, "non-stationary noise", "#eb6834"))):
        vals = [g.get((c, st), math.nan) for c in cfgs]
        ax.barh(y + (k - 0.5) * (h + 0.04), vals, height=h, color=color, label=name, edgecolor=SURFACE, lw=2)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:+.1f}", (v, yi + (k - 0.5) * (h + 0.04)), xytext=(3 if v >= 0 else -3, 0),
                            textcoords="offset points", va="center", ha="left" if v >= 0 else "right",
                            fontsize=7, color=INK2)
    ax.set_yticks(y, [LABEL[c] for c in cfgs])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("mean ΔSI-SDR (dB)")
    ax.set_title("Stationary vs non-stationary noise", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_dnsmos_vs_pesq(rows, path, plt):
    from scipy.stats import spearmanr
    pts = [(r["pesq_out"], r["dnsmos_ovrl_out"]) for r in rows
           if np.isfinite(r["pesq_out"]) and np.isfinite(r.get("dnsmos_ovrl_out", math.nan))]
    if len(pts) < 10:
        return None
    x, y = np.array(pts).T
    rho = spearmanr(x, y).statistic
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(x, y, s=8, color="#2a78d6", alpha=0.25, edgecolor="none")
    ax.set_xlabel("PESQ (intrusive, needs clean reference)")
    ax.set_ylabel("DNSMOS OVRL (non-intrusive prediction)")
    ax.set_title(f"Does the reference-free score agree?  Spearman ρ = {rho:.2f}", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rho


def fig_sweeps(rows, path, plt, metric="d_si_sdr", ylabel="mean ΔSI-SDR (dB)"):
    cfgs = [c for c in ORDER if any(r["config"] == c for r in rows)]
    noises = [n for n in NOISE_ORDER if any(r["noise"] == n for r in rows)]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    ncol = min(4, len(cfgs))
    nrow = math.ceil(len(cfgs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow), squeeze=False)
    for ax, c in zip(axes.flat, cfgs):
        sub = [r for r in rows if r["config"] == c]
        param = sub[0]["param"]
        g = group(sub, ("noise", "value"), metric)
        vals = sorted({r["value"] for r in sub})
        for k, n in enumerate(noises):
            ys = [g.get((n, v), math.nan) for v in vals]
            ax.plot(vals, ys, marker=STYLES[k % 4][1], ls=STYLES[k % 4][0], color=colors[k % 4],
                    label=NOISE_LABEL[n], markeredgecolor=SURFACE, markeredgewidth=1)
        if param == "atten_limit_db":
            ax.set_xticks(vals, ["off" if v >= 100 else f"{v:g}" for v in vals])
        ax.set_title(f"{LABEL[c]}: {param}")
        ax.set_xlabel(param)
    for ax in axes.flat[len(cfgs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(noises), fontsize=8, bbox_to_anchor=(0.5, -0.02),
               title=f"noise at {sub[0]['snr_db']:g} dB SNR", title_fontsize=8)
    fig.suptitle(f"Parameter sensitivity ({ylabel})", fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_spectrograms(rows, path, plt, noise="STRAFFIC", snr=5.0):
    """One mixture through every configuration (recomputed — needs the same environment as the bench)."""
    import scripts.benchmark as B
    from app.dsp.stft import StftConfig, stft
    from app.enhancers import registry
    from app.enhancers.runner import enhance_signal
    from app.params import validate
    if noise not in B.available_noises(None):
        noise = "babble"
    utt = B.load_utterances(1)[0]
    t = B.make_trial(utt, noise, snr)
    cfgs = [c for c in configs_in(rows) if registry.availability(B.CONFIGS[c]["algorithm"])[0]]
    panels = [("clean", t["clean"]), (f"noisy ({NOISE_LABEL[noise]}, {snr:g} dB)", t["noisy"])]
    panels += [(LABEL[c], enhance_signal(t["noisy"], B.SR, validate(B.CONFIGS[c]))) for c in cfgs]
    cfg = StftConfig(512, 0.75, "hann")
    ncol = 4
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow), sharex=True, sharey=True,
                             squeeze=False)
    ref = None
    for ax, (name, x) in zip(axes.flat, panels):
        db = 20 * np.log10(np.abs(stft(x, cfg)).T + 1e-9)
        if ref is None:
            ref = db.max()
        im = ax.imshow(db - ref, origin="lower", aspect="auto", cmap="inferno", vmin=-80, vmax=0,
                       extent=(0, len(x) / B.SR, 0, B.SR / 2000))
        ax.set_title(name, fontsize=8.5)
        ax.grid(False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("kHz")
    for ax in axes[-1, :]:
        ax.set_xlabel("s")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("dB re. clean peak")
    cb.outline.set_visible(False)
    fig.suptitle(f"Spectrograms of one utterance ({utt['id']})", fontsize=11, fontweight="bold", x=0.01,
                 ha="left")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# --- summary -----------------------------------------------------------------------------------

def _fmt_ci(m, ci, nd=2):
    if not np.isfinite(m):
        return "n/a"
    return f"{m:+.{nd}f} ± {ci:.{nd}f}" if np.isfinite(ci) else f"{m:+.{nd}f}"


def summary(rows, sweep_rows, path, rho):
    from scipy.stats import wilcoxon
    cfgs = configs_in(rows)
    n_utt = len({r["utt"] for r in rows})
    noises = [n for n in NOISE_ORDER if any(r["noise"] == n for r in rows)]
    snrs = sorted({r["snr_db"] for r in rows})
    src = sorted({r["speech_source"] for r in rows})
    L = ["# Benchmark summary (generated by scripts/report_figures.py, do not edit by hand)", "",
         f"- Runs: {len(rows)} ({len(cfgs)} configurations × {len(noises)} noise types × {len(snrs)} SNRs × "
         f"{n_utt} utterances; speech source: {', '.join(src)}).",
         f"- SNRs (dB): {', '.join(f'{s:g}' for s in snrs)}. Noises: {', '.join(NOISE_LABEL[n] for n in noises)}.",
         "- Values are mean ± 95 % confidence half-width (1.96·s/√n) over all runs of a configuration.",
         "- Δ = output − unprocessed input. RTF = single-core real-time factor (median).", ""]

    cols = [("d_si_sdr", "ΔSI-SDR (dB)", 2), ("d_snr", "ΔSNR (dB)", 2), ("d_pesq", "ΔPESQ", 3),
            ("d_estoi", "ΔESTOI", 3), ("lsd_out", "LSD out (dB)", 2), ("noise_reduction_db", "NR (dB)", 1),
            ("speech_attenuation_db", "speech atten. (dB)", 2), ("d_dnsmos", "ΔDNSMOS OVRL", 3)]
    cols = [c for c in cols if any(np.isfinite(r[c[0]]) for r in rows)]
    L += ["## Overall", "", "| Configuration | " + " | ".join(c[1] for c in cols) + " | RTF |",
          "|---|" + "---|" * (len(cols) + 1)]
    for c in cfgs:
        sub = [r for r in rows if r["config"] == c]
        cells = [_fmt_ci(*mean_ci([r[m] for r in sub])[:2], nd) for m, _, nd in cols]
        L.append(f"| {LABEL[c]} | " + " | ".join(cells) + f" | {np.median([r['rtf'] for r in sub]):.3f} |")
    L.append("")

    for m, mlabel in (("d_si_sdr", "ΔSI-SDR"), ("d_pesq", "ΔPESQ")):
        if not any(np.isfinite(r[m]) for r in rows):
            continue
        g = group(rows, ("noise", "config"), m)
        L += [f"## Best configuration per noise type ({mlabel}, all SNRs pooled)", "",
              "| Noise | best | 2nd | Wilcoxon p (best vs 2nd, paired) |", "|---|---|---|---|"]
        for n in noises:
            ranked = sorted(((g[(n, c)], c) for c in cfgs if (n, c) in g and np.isfinite(g[(n, c)])),
                            reverse=True)
            if len(ranked) < 2:
                continue
            (v1, c1), (v2, c2) = ranked[:2]
            a = {(r["snr_db"], r["utt"]): r[m] for r in rows if r["noise"] == n and r["config"] == c1}
            b = {(r["snr_db"], r["utt"]): r[m] for r in rows if r["noise"] == n and r["config"] == c2}
            keys = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
            try:
                p = wilcoxon([a[k] for k in keys], [b[k] for k in keys]).pvalue if len(keys) >= 6 else math.nan
            except ValueError:
                p = math.nan
            ptxt = "n/a (too few pairs)" if not np.isfinite(p) else (f"{p:.2g}" + (" (significant at 0.05)"
                                                                                    if p < 0.05 else ""))
            L.append(f"| {NOISE_LABEL[n]} | {LABEL[c1]} ({v1:+.2f}) | {LABEL[c2]} ({v2:+.2f}) | {ptxt} |")
        L.append("")

    g = group(rows, ("snr_db", "config"), "d_si_sdr")
    L += ["## Best configuration per input SNR (ΔSI-SDR, all noises pooled)", "", "| SNR (dB) | best | 2nd |",
          "|---|---|---|"]
    for s in snrs:
        ranked = sorted(((g[(s, c)], c) for c in cfgs if (s, c) in g), reverse=True)
        L.append(f"| {s:g} | {LABEL[ranked[0][1]]} ({ranked[0][0]:+.2f}) | "
                 f"{LABEL[ranked[1][1]]} ({ranked[1][0]:+.2f}) |" if len(ranked) > 1 else f"| {s:g} | n/a | n/a |")
    L.append("")

    worse = []
    for c in cfgs:
        for n in noises:
            v = group([r for r in rows if r["config"] == c and r["noise"] == n], ("config",), "d_si_sdr")
            if (c,) in v and v[(c,)] < 0:
                worse.append(f"{LABEL[c]} on {NOISE_LABEL[n]} ({v[(c,)]:+.2f} dB)")
    L += ["## Cases where a configuration made SI-SDR worse on average", "",
          ("- " + "\n- ".join(worse)) if worse else "- none", ""]
    if rho is not None:
        L += ["## Metric agreement", "", f"- Spearman ρ between PESQ and DNSMOS OVRL over all outputs: {rho:.2f}.", ""]
    if sweep_rows:
        L += ["## Parameter sweeps (ΔSI-SDR, mean over sweep noises at 5 dB)", ""]
        for c in [c for c in ORDER if any(r["config"] == c for r in sweep_rows)]:
            sub = [r for r in sweep_rows if r["config"] == c]
            g = group(sub, ("value",), "d_si_sdr")
            best = max(g.items(), key=lambda kv: kv[1])
            L.append(f"- {LABEL[c]}, {sub[0]['param']}: " +
                     ", ".join(f"{k[0]:g} → {v:+.2f}" for k, v in sorted(g.items())) +
                     f"  (best {best[0][0]:g})")
        L.append("")
    path.write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, default=RESULTS / "bench.csv")
    ap.add_argument("--sweep", type=Path, default=RESULTS / "sweep.csv")
    ap.add_argument("--out", type=Path, default=RESULTS / "figures")
    ap.add_argument("--summary", type=Path, default=RESULTS / "summary.md")
    ap.add_argument("--no-spectrograms", action="store_true", help="skip the figure that re-runs every model")
    args = ap.parse_args()
    if not args.bench.is_file():
        raise SystemExit(f"{args.bench} not found — run scripts/benchmark.py first")
    plt = _mpl()
    rows = load(args.bench)
    args.out.mkdir(parents=True, exist_ok=True)
    o = args.out
    fig_metric_vs_snr(rows, "d_si_sdr", "ΔSI-SDR (dB)", "SI-SDR improvement vs input SNR",
                      o / "fig1_si_sdr_vs_snr.png", plt)
    if any(np.isfinite(r["pesq_out"]) for r in rows):
        fig_pesq_estoi(rows, o / "fig2_pesq_estoi_vs_snr.png", plt)
    fig_winner_grid(rows, o / "fig3_best_per_condition.png", plt)
    fig_tradeoff(rows, o / "fig4_tradeoff.png", plt)
    fig_cost(rows, o / "fig5_quality_vs_cost.png", plt)
    fig_stationarity(rows, o / "fig6_stationary_vs_nonstationary.png", plt)
    sweep_rows = load(args.sweep) if args.sweep.is_file() else []
    if sweep_rows:
        fig_sweeps(sweep_rows, o / "fig7a_sensitivity_si_sdr.png", plt)
        if any(np.isfinite(r["d_pesq"]) for r in sweep_rows):
            fig_sweeps(sweep_rows, o / "fig7b_sensitivity_pesq.png", plt, "d_pesq", "mean ΔPESQ")
    if not args.no_spectrograms:
        fig_spectrograms(rows, o / "fig8_spectrograms.png", plt)
    fig_heatmap(rows, "d_si_sdr", "Mean ΔSI-SDR (dB) per configuration and noise type", o / "fig9_heatmap_si_sdr.png", plt)
    if any(np.isfinite(r["d_pesq"]) for r in rows):
        fig_heatmap(rows, "d_pesq", "Mean ΔPESQ per configuration and noise type", o / "fig10_heatmap_pesq.png",
                    plt, fmt="{:+.2f}")
    rho = fig_dnsmos_vs_pesq(rows, o / "fig11_dnsmos_vs_pesq.png", plt)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary(rows, sweep_rows, args.summary, rho)
    print(f"figures → {o}\nsummary → {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
