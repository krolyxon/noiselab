#!/usr/bin/env python3
"""Benchmark every available algorithm over noise types × SNRs × utterances (CLAUDE.md §14).

    .venv-ml/bin/python scripts/benchmark.py --quick            # smoke test (~1 min)
    .venv-ml/bin/python scripts/benchmark.py                    # full grid → results/bench.csv
    .venv-ml/bin/python scripts/benchmark.py --sweep            # parameter sensitivity → results/sweep.csv
    .venv-ml/bin/python scripts/report_figures.py               # figures + results/summary.md

Protocol (the same for every algorithm):
- Clean speech: data/clean (LibriSpeech, scripts/fetch_datasets.py), 16 kHz. Falls back to the
  synthetic voiced signal if data/ is missing, and the CSV records which was used.
- Each utterance gets LEAD = 1.0 s of silence before it and 0.25 s after. That gives the
  methods that need a noise-only lead-in (spectral subtraction, the statistical estimators in
  `initial` mode, noisereduce stationary) the same 1 s of noise that a real recording usually
  has. The deep models ignore it.
- Noise: synthetic (app/dsp/synth.py) and real (DEMAND channel 1, random 1 s-aligned offset).
  It is scaled so that the SNR over the *utterance* (not the padded silence) is exact.
- Seeds are derived from (noise, utterance), so every algorithm sees the same mixtures and
  every rerun is identical.
- Metrics are computed on ŝ before the output stage, over the whole padded signal.
- Each worker process uses 1 CPU thread (torch, onnxruntime, BLAS), so `rtf` is a
  single-core real-time factor, comparable across algorithms.
Resumable: rows already in the CSV are skipped. Delete the CSV to start over.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NOISELAB_ORT_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.dsp import metrics as M  # noqa: E402
from app.dsp import synth  # noqa: E402
from app.enhancers import quality as Q  # noqa: E402
from app.enhancers import registry  # noqa: E402
from app.enhancers.resample import resample  # noqa: E402
from app.enhancers.runner import enhance_signal  # noqa: E402
from app.params import PRESETS, validate  # noqa: E402

SR = 16000
LEAD_S, TAIL_S = 1.0, 0.25
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Algorithm configurations (label → parameters). Defaults unless noted, so the comparison
# shows each method as shipped. Parameter sensitivity is in --sweep.
CONFIGS: dict[str, dict] = {
    "ss_simulink": PRESETS["match_simulink_basic"]["params"],
    "wiener_dd": PRESETS["wiener_dd_default"]["params"],
    "mmse_stsa": PRESETS["mmse_stsa_default"]["params"],
    "log_mmse": PRESETS["log_mmse_default"]["params"],
    "omlsa": PRESETS["omlsa_default"]["params"],
    "noisereduce": {"algorithm": "noisereduce"},
    "webrtc_ns": {"algorithm": "webrtc_ns"},
    "rnnoise": {"algorithm": "rnnoise"},
    "nsnet2": {"algorithm": "nsnet2"},
    "deepfilternet3": {"algorithm": "deepfilternet3"},
    "demucs": {"algorithm": "demucs"},
}

SYNTH_NOISES = ("white", "pink", "hum50", "fan", "babble", "clicks")
DEMAND_NOISES = ("DLIVING", "PCAFETER", "STRAFFIC", "TCAR", "OOFFICE")
# Stationarity classes used by report_figures.py (judged from the material, not measured).
STATIONARY = {"white", "pink", "hum50", "fan", "TCAR"}
SNRS = (-5, 0, 5, 10, 15, 20)

# Parameter sweeps (--sweep): algorithm label → (parameter, values). "atten_limit_db" is the
# flat parameter of Eq. (34); the rest are algo_params / flat parameters as named.
SWEEPS: dict[str, tuple[str, tuple]] = {
    "ss_simulink": ("alpha", (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)),
    "log_mmse": ("dd_alpha", (0.90, 0.95, 0.98, 0.99, 0.995)),
    "omlsa": ("g_min_db", (-40.0, -30.0, -25.0, -20.0, -15.0, -10.0)),
    "noisereduce": ("prop_decrease", (0.25, 0.5, 0.75, 1.0)),
    "webrtc_ns": ("level", (1, 2, 3, 4)),
    "deepfilternet3": ("atten_limit_db", (6.0, 12.0, 18.0, 24.0, 36.0, 100.0)),
    "demucs": ("atten_limit_db", (6.0, 12.0, 18.0, 24.0, 36.0, 100.0)),
}
SWEEP_NOISES = ("white", "babble", "PCAFETER", "STRAFFIC")
SWEEP_SNR = 5

FIELDS = ["config", "algorithm", "family", "noise", "noise_source", "stationary", "snr_db", "utt",
          "speaker", "sex", "speech_source", "seed", "duration_s",
          "snr_in", "snr_out", "si_sdr_in", "si_sdr_out", "seg_snr_in", "seg_snr_out",
          "pesq_in", "pesq_out", "stoi_in", "stoi_out", "estoi_in", "estoi_out", "lsd_in", "lsd_out",
          "noise_reduction_db", "speech_attenuation_db",
          "dnsmos_sig_in", "dnsmos_bak_in", "dnsmos_ovrl_in",
          "dnsmos_sig_out", "dnsmos_bak_out", "dnsmos_ovrl_out", "proc_s", "rtf"]
SWEEP_FIELDS = ["config", "param", "value"] + FIELDS[1:]


# --- data --------------------------------------------------------------------------------------

def load_utterances(n: int) -> list[dict]:
    manifest = DATA / "clean" / "manifest.csv"
    if manifest.is_file():
        with open(manifest) as f:
            rows = list(csv.DictReader(f))[:n]
        out = []
        for r in rows:
            x, sr = sf.read(DATA / "clean" / r["file"], dtype="float64")
            out.append({"id": Path(r["file"]).stem, "speaker": r["speaker"], "sex": r["sex"],
                        "x": resample(x, sr, SR), "source": "librispeech"})
        return out
    print("! data/clean not found — using the synthetic voiced signal (run scripts/fetch_datasets.py)")
    return [{"id": f"synth{i}", "speaker": "synth", "sex": "?", "x": synth.clean_voiced(SR, 6.0, seed=i),
             "source": "synthetic"} for i in range(n)]


def available_noises(requested: list[str] | None) -> list[str]:
    real = [n for n in DEMAND_NOISES if (DATA / "noise" / f"{n}.wav").is_file()]
    allowed = list(SYNTH_NOISES) + real
    if requested:
        missing = [n for n in requested if n not in allowed]
        if missing:
            raise SystemExit(f"unknown or missing noise(s): {missing}; available: {allowed}")
        return requested
    return allowed


def make_noise(kind: str, n: int, seed: int) -> np.ndarray:
    if kind in SYNTH_NOISES:
        return synth.make_noise(kind, n, SR, seed)
    x, sr = sf.read(DATA / "noise" / f"{kind}.wav", dtype="float64")
    x = resample(x, sr, SR)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(len(x) - n, 1) // SR + 1)) * SR
    seg = x[start:start + n]
    return np.resize(seg, n) if len(seg) < n else seg


def make_trial(utt: dict, noise: str, snr: float) -> dict:
    seed = zlib.crc32(f"{noise}|{utt['id']}".encode()) % (2**31)
    speech = utt["x"]
    lead, tail = int(LEAD_S * SR), int(TAIL_S * SR)
    clean = np.concatenate([np.zeros(lead), speech, np.zeros(tail)])
    d = make_noise(noise, len(clean), seed)
    p_s = np.mean(speech**2)                              # SNR over the utterance (Eq. 21)
    d = d * np.sqrt(p_s / (np.mean(d**2) * 10.0 ** (snr / 10.0)))
    noisy = clean + d
    k = min(1.0, 0.99 / np.max(np.abs(noisy)))           # same factor keeps the SNR
    return {"clean": clean * k, "noisy": noisy * k, "seed": seed}


# --- metrics -----------------------------------------------------------------------------------

def _frame_energies(x: np.ndarray) -> np.ndarray:
    return M.frame_energy_db(x, 512, 256)


def input_metrics(clean: np.ndarray, noisy: np.ndarray, dnsmos: bool) -> dict:
    q = Q.intrusive(clean, noisy, SR)
    out = {"snr_in": M.snr_db(clean, noisy), "si_sdr_in": M.si_sdr_db(clean, noisy),
           "seg_snr_in": M.segmental_snr_db(clean, noisy), "pesq_in": q["pesq"], "stoi_in": q["stoi"],
           "estoi_in": q["estoi"], "lsd_in": M.log_spectral_distance_db(clean, noisy)}
    if dnsmos:
        d = Q.dnsmos(noisy, SR) or {}
        out |= {f"dnsmos_{k}_in": v for k, v in d.items()}
    return out


def output_metrics(clean: np.ndarray, noisy: np.ndarray, out: np.ndarray, speech: np.ndarray,
                   e_noisy: np.ndarray, dnsmos: bool) -> dict:
    q = Q.intrusive(clean, out, SR)
    e_out = _frame_energies(out)
    r = {"snr_out": M.snr_db(clean, out), "si_sdr_out": M.si_sdr_db(clean, out),
         "seg_snr_out": M.segmental_snr_db(clean, out), "pesq_out": q["pesq"], "stoi_out": q["stoi"],
         "estoi_out": q["estoi"], "lsd_out": M.log_spectral_distance_db(clean, out),
         "noise_reduction_db": M.noise_reduction_db(e_noisy, e_out, speech),
         "speech_attenuation_db": M.speech_attenuation_db(e_noisy, e_out, speech)}
    if dnsmos:
        d = Q.dnsmos(out, SR) or {}
        r |= {f"dnsmos_{k}_out": v for k, v in d.items()}
    return r


def _speech_frames(clean: np.ndarray) -> np.ndarray:
    """Oracle VAD from the clean signal: frames within 40 dB of the loudest clean frame.

    In the benchmark the clean signal is known, so speech/noise frames for Eq. (19)/(36)
    come from it rather than from the energy VAD (which would differ per noise type).
    """
    e = _frame_energies(clean)
    return e > e.max() - 40.0


# --- workers -----------------------------------------------------------------------------------

def _init_worker() -> None:
    warnings.filterwarnings("ignore")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def run_unit(utt: dict, noise: str, snr: float, todo: list[tuple[str, dict, dict]], dnsmos: bool) -> list[dict]:
    """One mixture, all pending configurations. todo = [(label, params, extra columns)]."""
    t = make_trial(utt, noise, snr)
    clean, noisy = t["clean"], t["noisy"]
    speech = _speech_frames(clean)
    e_noisy = _frame_energies(noisy)
    base = {"noise": noise, "noise_source": "synthetic" if noise in SYNTH_NOISES else "DEMAND",
            "stationary": int(noise in STATIONARY), "snr_db": snr, "utt": utt["id"], "speaker": utt["speaker"],
            "sex": utt["sex"], "speech_source": utt["source"], "seed": t["seed"],
            "duration_s": round(len(noisy) / SR, 3)}
    base |= input_metrics(clean, noisy, dnsmos)
    rows = []
    for label, params, extra in todo:
        p = validate(params)
        t0 = time.perf_counter()
        try:
            out = enhance_signal(noisy, SR, p)
        except Exception as exc:          # one failing model must not stop the benchmark
            print(f"! {label} failed on {utt['id']}/{noise}/{snr}: {type(exc).__name__}: {exc}", flush=True)
            continue
        dt = time.perf_counter() - t0
        row = dict(base) | extra | {"config": label, "algorithm": p.algorithm,
                                    "family": registry.spec(p.algorithm).family,
                                    "proc_s": dt, "rtf": dt / (len(noisy) / SR)}
        row |= output_metrics(clean, noisy, out, speech, e_noisy, dnsmos)
        rows.append(row)
    return rows


# --- driver ------------------------------------------------------------------------------------

def _done_keys(path: Path, key_cols: list[str]) -> set[tuple]:
    if not path.is_file():
        return set()
    with open(path) as f:
        return {tuple(r[c] for c in key_cols) for r in csv.DictReader(f)}


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.5g}"
    return str(v)


def run(units: list[tuple], path: Path, fields: list[str], workers: int, dnsmos: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    n_rows = sum(len(u[3]) for u in units)
    print(f"{len(units)} mixtures, {n_rows} algorithm runs, {workers} workers → {path}")
    t0 = time.time()
    done = 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futs = [ex.submit(run_unit, *u, dnsmos) for u in units]
            for fut in as_completed(futs):
                rows = fut.result()
                for r in rows:
                    w.writerow({k: _fmt(r.get(k)) for k in fields})
                f.flush()
                done += len(rows)
                el = time.time() - t0
                print(f"  {done}/{n_rows} runs, {el / 60:.1f} min elapsed, "
                      f"~{el / max(done, 1) * (n_rows - done) / 60:.1f} min left", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny grid: 2 utterances × 3 noises × 2 SNRs")
    ap.add_argument("--sweep", action="store_true", help="parameter sensitivity instead of the main grid")
    ap.add_argument("--configs", nargs="*", help=f"subset of {list(CONFIGS)}")
    ap.add_argument("--noises", nargs="*", help="subset of noise types")
    ap.add_argument("--snrs", nargs="*", type=float, help=f"input SNRs (default {list(SNRS)})")
    ap.add_argument("--n-utts", type=int, default=20, help="utterances (default 20)")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--no-dnsmos", action="store_true", help="skip DNSMOS (≈ 2× faster)")
    ap.add_argument("--out", type=Path, help="output CSV (default results/bench.csv or results/sweep.csv)")
    args = ap.parse_args()

    configs = args.configs or list(CONFIGS)
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        raise SystemExit(f"unknown config(s): {sorted(unknown)}")
    skipped = [c for c in configs if not registry.availability(CONFIGS[c]["algorithm"])[0]]
    for c in skipped:
        print(f"! skipping {c}: {registry.availability(CONFIGS[c]['algorithm'])[1]}")
    configs = [c for c in configs if c not in skipped]
    dnsmos = not args.no_dnsmos and Q.available()["dnsmos"]
    if not Q.available()["pesq"]:
        print("! pesq / pystoi not installed: those columns will be empty (install requirements-ml.txt)")

    if args.quick:
        n_utts, noises, snrs = 2, ["white", "babble"] + (["STRAFFIC"] if "STRAFFIC" in available_noises(None)
                                                         else ["fan"]), [0.0, 10.0]
    else:
        n_utts = args.n_utts
        noises = available_noises(args.noises)
        snrs = [float(s) for s in (args.snrs or SNRS)]
    utts = load_utterances(n_utts)

    if args.sweep:
        path = args.out or RESULTS / ("sweep_quick.csv" if args.quick else "sweep.csv")
        noises = [n for n in SWEEP_NOISES if n in available_noises(None)]
        snr = float(SWEEP_SNR)
        done = _done_keys(path, ["config", "param", "value", "noise", "utt"])
        units = []
        for utt in utts[: (2 if args.quick else min(n_utts, 10))]:
            for noise in noises:
                todo = []
                for c in configs:
                    if c not in SWEEPS:
                        continue
                    param, values = SWEEPS[c]
                    for v in values:
                        if (c, param, _fmt(float(v) if isinstance(v, float) else v), noise, utt["id"]) in done:
                            continue
                        base = dict(CONFIGS[c])
                        if param in registry.spec(base["algorithm"]).schema:
                            prm = base | {"algo_params": dict(base.get("algo_params", {})) | {param: v}}
                        else:
                            prm = base | {param: v}
                        todo.append((c, prm, {"param": param, "value": v}))
                if todo:
                    units.append((utt, noise, snr, todo))
        run(units, path, SWEEP_FIELDS, args.workers, dnsmos)
        return 0

    path = args.out or RESULTS / ("bench_quick.csv" if args.quick else "bench.csv")
    done = _done_keys(path, ["config", "noise", "snr_db", "utt"])
    units = []
    for utt in utts:
        for noise in noises:
            for snr in snrs:
                todo = [(c, CONFIGS[c], {}) for c in configs
                        if (c, noise, _fmt(snr), utt["id"]) not in done]
                if todo:
                    units.append((utt, noise, snr, todo))
    run(units, path, FIELDS, args.workers, dnsmos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
