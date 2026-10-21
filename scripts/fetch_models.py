#!/usr/bin/env python3
"""Download the model files of the comparison bench (CLAUDE.md §14) into models/.

    .venv-ml/bin/python scripts/fetch_models.py            # ONNX files + pre-warm torch weights
    .venv-ml/bin/python scripts/fetch_models.py --no-torch # ONNX files only

ONNX files (NSNet2, DNSMOS P.835) are verified against SHA-256 sums. DeepFilterNet3
and Demucs download their own weights on first use; this script just triggers that
now so the first request in the web app is not slow. Nothing here is needed by the
basic spectral-subtraction app.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.enhancers.model_files import FILES, MODEL_DIR, sha256_of  # noqa: E402


def fetch_onnx() -> bool:
    ok = True
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for key, mf in FILES.items():
        if mf.path.is_file() and sha256_of(mf.path) == mf.sha256:
            print(f"  ok      {mf.name}")
            continue
        print(f"  get     {mf.name}  ({mf.size / 1e6:.1f} MB, {mf.licence})")
        tmp = mf.path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(mf.url, tmp)
        except OSError as exc:
            print(f"  FAILED  {key}: {exc}")
            ok = False
            continue
        digest = sha256_of(tmp)
        if digest != mf.sha256:
            tmp.unlink()
            print(f"  FAILED  {key}: SHA-256 mismatch ({digest})")
            ok = False
            continue
        tmp.replace(mf.path)
    return ok


def warm_torch() -> bool:
    ok = True
    for name, fn in (("DeepFilterNet3", _warm_dfn), ("Demucs dns64", _warm_demucs)):
        try:
            fn()
            print(f"  ok      {name} weights cached")
        except Exception as exc:      # report and carry on: every model is optional
            print(f"  FAILED  {name}: {type(exc).__name__}: {exc}")
            ok = False
    return ok


def _warm_dfn() -> None:
    from app.enhancers.thirdparty.deepfilter_ import load_model
    load_model()


def _warm_demucs() -> None:
    from app.enhancers.thirdparty.demucs_ import load_model
    load_model("dns64")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-torch", action="store_true", help="skip pre-warming DeepFilterNet3 / Demucs weights")
    args = ap.parse_args()
    print(f"model dir: {MODEL_DIR}")
    ok = fetch_onnx()
    if not args.no_torch:
        ok = warm_torch() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
