"""Downloadable model files for the comparison bench (CLAUDE.md §14).

Shared by scripts/fetch_models.py (downloads + verifies) and the wrappers in
thirdparty/ (locate the file). Torch models (DeepFilterNet3, Demucs) manage their
own weight cache; fetch_models.py only pre-warms it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("NOISELAB_MODEL_DIR", ROOT / "models"))

_DNS = "https://raw.githubusercontent.com/microsoft/DNS-Challenge"


@dataclass(frozen=True)
class ModelFile:
    name: str            # file name inside MODEL_DIR
    url: str
    sha256: str
    size: int
    licence: str
    used_by: str

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.name


FILES: dict[str, ModelFile] = {
    "nsnet2": ModelFile(
        "nsnet2-20ms-baseline.onnx",
        f"{_DNS}/icassp2021-final/NSNet2-baseline/nsnet2-20ms-baseline.onnx",
        "88429b6253600be840ab816f46f466811d20078142fb12bff8cafe2b27bd4ca9", 10_752_263,
        "CC BY 4.0 (Microsoft DNS-Challenge)", "nsnet2 enhancer"),
    "dnsmos_p835": ModelFile(
        "dnsmos_sig_bak_ovr.onnx",
        f"{_DNS}/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd", 1_157_965,
        "CC BY 4.0 (Microsoft DNS-Challenge)", "DNSMOS P.835 quality metric"),
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate(key: str) -> Path | None:
    """Path of a downloaded model file, or None if it is missing (size check only — cheap)."""
    mf = FILES[key]
    p = mf.path
    return p if p.is_file() and p.stat().st_size == mf.size else None
