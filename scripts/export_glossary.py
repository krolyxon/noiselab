#!/usr/bin/env python3
"""Write docs/GLOSSARY.md from app/glossary.py (the single source of plain-language wording).

    python scripts/export_glossary.py          # tests/test_glossary.py checks the file is up to date
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import glossary as G  # noqa: E402

OUT = ROOT / "docs" / "GLOSSARY.md"
BETTER = {"higher": "↑ higher", "lower": "↓ lower", "": "—"}


def render() -> str:
    L = ["# Glossary: the scores and words in plain language", "",
         "_Generated from `app/glossary.py` by `scripts/export_glossary.py`. Do not edit by hand._", "",
         "The report figures, `results/summary.md` and the web app all use these words. Each term has a "
         "**plain name** first and the **standard technical name** second, so you can explain a graph simply "
         "and still match the literature.", "",
         "## What the scores mean", "", "| Plain name | Technical name | Better | Meaning | Think of it as |",
         "|---|---|---|---|---|"]
    for k in G.GLOSSARY_ORDER:
        t = G.METRICS[k]
        L.append(f"| **{t.short}** | {t.tech} | {BETTER[t.better]} | {t.explain} | {t.analogy} |")
    L += ["", "## Other words on the graphs", "", "| Plain name | Technical name | Meaning | Think of it as |",
          "|---|---|---|---|"]
    for t in G.TERMS.values():
        L.append(f"| **{t.short}** | {t.tech} | {t.explain} | {t.analogy} |")
    L += ["", "## Types of method", "", "| Type | Technical name | Meaning |", "|---|---|---|"]
    for t in G.FAMILIES.values():
        L.append(f"| **{t.short}** | {t.tech} | {t.explain} |")
    L += ["", "## The 11 methods in one line each", "", "| Method id | What it does |", "|---|---|"]
    for k, v in G.ALGORITHMS.items():
        L.append(f"| `{k}` | {v} |")
    L += ["", "## Knobs in the parameter sweep (figure 7)", "", "| Parameter | Plain meaning |", "|---|---|"]
    for k, v in G.PARAMS.items():
        L.append(f"| `{k}` | {v} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"→ {OUT}")
