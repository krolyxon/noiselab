#!/usr/bin/env python3
"""Download the benchmark's real speech and real noise into data/ (CLAUDE.md §14).

    python scripts/fetch_datasets.py                 # 40 utterances + 5 DEMAND environments
    python scripts/fetch_datasets.py --n-utts 20 --demand DLIVING PCAFETER
    python scripts/fetch_datasets.py --no-demand     # speech only

Clean speech: LibriSpeech test-clean (V. Panayotov et al., ICASSP 2015), CC BY 4.0.
  OpenSLR only offers the whole archive (347 MB), so it is *streamed*: one 4–10 s
  utterance per speaker (40 speakers, 20 F / 20 M) is kept as FLAC (~20 MB), and the
  archive's MD5 is checked at the end of the stream. Nothing else is written to disk.

Real noise: DEMAND (J. Thiemann, N. Ito, E. Vincent, 2013), CC BY 4.0 (Zenodo 1227121).
  Each environment is a ~100 MB zip of 16 microphone channels, 5 min at 16 kHz. Only
  channel 1 is kept (~9.6 MB each). The zip's MD5 is checked, then the zip is deleted.

Output:
  data/clean/<speaker>-<chapter>-<utt>.flac, data/clean/manifest.csv
  data/noise/<ENV>.wav, data/LICENCES.md
Everything under data/ is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LIBRI_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRI_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"            # openslr.org/resources/12/md5sum.txt

DEMAND_URL = "https://zenodo.org/api/records/1227121/files/{env}_16k.zip/content"
DEMAND_MD5 = {                                            # Zenodo record metadata
    "DLIVING": "46741384d9e434a0bd8b3ec1830b6052",
    "DKITCHEN": "7ffbf52d7f4699f96927846103dc8788",
    "PCAFETER": "99927d148128254141a9417d051510bb",
    "STRAFFIC": "2efa87262f272bbf9ba578088e81939c",
    "TCAR": "4d930012796bd298932245a26189f973",
    "OOFFICE": "7b61cc2d182d5a654cb9c3101ddd4041",
    "NFIELD": "a740046c6f4e174e16f5d568aaec5024",
    "TBUS": "706b11b0d8504f9f3b3f3211e91b3863",
}
DEMAND_DEFAULT = ("DLIVING", "PCAFETER", "STRAFFIC", "TCAR", "OOFFICE")
DEMAND_DESC = {
    "DLIVING": "domestic living room", "DKITCHEN": "domestic kitchen", "PCAFETER": "busy cafeteria (babble)",
    "STRAFFIC": "busy street traffic", "TCAR": "inside a moving car", "OOFFICE": "small office",
    "NFIELD": "sports field (outdoor)", "TBUS": "inside a bus",
}

MIN_S, MAX_S = 4.0, 10.0


class _HashingReader(io.RawIOBase):
    """File-like wrapper that MD5-hashes everything read through it."""

    def __init__(self, raw):
        self.raw, self.md5, self.n = raw, hashlib.md5(), 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self.raw.read(len(b))
        self.md5.update(data)
        self.n += len(data)
        b[: len(data)] = data
        return len(data)


def _speaker_sex(tar_text: str) -> dict[str, str]:
    out = {}
    for line in tar_text.splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[2] == "test-clean":
            out[parts[0]] = parts[1]
    return out


def fetch_librispeech(n_utts: int) -> bool:
    out_dir = DATA / "clean"
    manifest = out_dir / "manifest.csv"
    if manifest.is_file():
        with open(manifest) as f:
            rows = list(csv.DictReader(f))
        if len(rows) >= n_utts and all((out_dir / r["file"]).is_file() for r in rows):
            print(f"  ok      LibriSpeech: {len(rows)} utterances already in {out_dir}")
            return True
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  stream  LibriSpeech test-clean (347 MB, CC BY 4.0) → keep 1 utterance per speaker")
    sex: dict[str, str] = {}
    chosen: dict[str, dict] = {}
    with urllib.request.urlopen(LIBRI_URL) as resp:
        hr = _HashingReader(resp)
        br = io.BufferedReader(hr, 1 << 20)
        with tarfile.open(fileobj=br, mode="r|gz") as tar:
            for m in tar:
                if m.name.endswith("SPEAKERS.TXT"):
                    sex = _speaker_sex(tar.extractfile(m).read().decode("utf-8", "replace"))
                    continue
                if not m.name.endswith(".flac"):
                    continue
                spk = Path(m.name).name.split("-")[0]
                if spk in chosen:
                    continue
                data = tar.extractfile(m).read()
                info = sf.info(io.BytesIO(data))
                if not (MIN_S <= info.duration <= MAX_S):
                    continue
                (out_dir / Path(m.name).name).write_bytes(data)
                chosen[spk] = {"file": Path(m.name).name, "speaker": spk,
                               "duration_s": round(info.duration, 3), "sr": info.samplerate}
                if len(chosen) % 10 == 0:
                    print(f"          {len(chosen)} speakers… ({hr.n / 1e6:.0f} MB read)")
        while br.read(1 << 20):                   # drain to the end so the MD5 covers the whole file
            pass
    digest = hr.md5.hexdigest()
    if digest != LIBRI_MD5:
        for r in chosen.values():
            (out_dir / r["file"]).unlink(missing_ok=True)
        print(f"  FAILED  LibriSpeech MD5 mismatch ({digest}); files removed")
        return False
    # Balance the sexes, then keep the first n_utts in speaker order (deterministic).
    rows = sorted(chosen.values(), key=lambda r: int(r["speaker"]))
    for r in rows:
        r["sex"] = sex.get(r["speaker"], "?")
    f_rows = [r for r in rows if r["sex"] == "F"]
    m_rows = [r for r in rows if r["sex"] == "M"]
    keep = []
    for i in range(max(len(f_rows), len(m_rows))):
        keep += [x[i] for x in (f_rows, m_rows) if i < len(x)]
    keep += [r for r in rows if r["sex"] not in ("F", "M")]
    keep = keep[:n_utts]
    for r in rows:
        if r not in keep:
            (out_dir / r["file"]).unlink(missing_ok=True)
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "sex", "duration_s", "sr"])
        w.writeheader()
        w.writerows(keep)
    print(f"  ok      LibriSpeech: {len(keep)} utterances "
          f"({sum(r['sex'] == 'F' for r in keep)} F / {sum(r['sex'] == 'M' for r in keep)} M), MD5 verified")
    return True


def fetch_demand(envs: list[str]) -> bool:
    out_dir = DATA / "noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for env in envs:
        dest = out_dir / f"{env}.wav"
        if dest.is_file():
            print(f"  ok      DEMAND {env}")
            continue
        print(f"  get     DEMAND {env} ({DEMAND_DESC.get(env, '')}), keep channel 1")
        with tempfile.TemporaryDirectory() as td:
            zpath = Path(td) / f"{env}.zip"
            try:
                urllib.request.urlretrieve(DEMAND_URL.format(env=env), zpath)
            except OSError as exc:
                print(f"  FAILED  {env}: {exc}")
                ok = False
                continue
            md5 = hashlib.md5(zpath.read_bytes()).hexdigest()
            if md5 != DEMAND_MD5[env]:
                print(f"  FAILED  {env}: MD5 mismatch ({md5})")
                ok = False
                continue
            with zipfile.ZipFile(zpath) as z:
                name = next(n for n in z.namelist() if n.endswith("ch01.wav"))
                dest.write_bytes(z.read(name))
    return ok


LICENCES = """# Licences of downloaded data (scripts/fetch_datasets.py)

- **LibriSpeech test-clean**: V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus
  based on public domain audio books," ICASSP 2015. CC BY 4.0. https://www.openslr.org/12/
  One 4–10 s utterance per speaker is kept, unmodified (FLAC).
- **DEMAND**: J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise
  Database (DEMAND)," Proc. Meetings on Acoustics 19, 2013. CC BY 4.0. https://zenodo.org/records/1227121
  Only channel 1 of each environment is kept, unmodified (16 kHz WAV).

These files are not committed to the repository (data/ is gitignored).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-utts", type=int, default=40, help="utterances to keep (≤ 40, one per speaker)")
    ap.add_argument("--demand", nargs="*", default=list(DEMAND_DEFAULT), choices=sorted(DEMAND_MD5),
                    help="DEMAND environments to download")
    ap.add_argument("--no-demand", action="store_true", help="skip the DEMAND noises")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    (DATA / "LICENCES.md").write_text(LICENCES)
    print(f"data dir: {DATA}")
    ok = fetch_librispeech(min(args.n_utts, 40))
    if not args.no_demand:
        ok = fetch_demand(args.demand) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
