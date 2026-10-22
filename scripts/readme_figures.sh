#!/usr/bin/env bash
# Copy the benchmark figures into docs/figures/ for the README, shrunk for GitHub.
#
#   scripts/readme_figures.sh            # after scripts/report_figures.py
#
# results/ is gitignored, so the README cannot link there. Each figure is resized to at most
# 1600 px wide, stripped of metadata and reduced to a 256-colour PNG (about 4x smaller).
# fig0 (the glossary table) is left out on purpose: the README links docs/GLOSSARY.md instead.
# Needs ImageMagick 7 (`magick`).
set -euo pipefail

cd "$(dirname "$0")/.."
src=results/figures
dst=docs/figures

command -v magick >/dev/null || { echo "ImageMagick 7 ('magick') not found" >&2; exit 1; }
[ -d "$src" ] || { echo "$src not found: run scripts/report_figures.py first" >&2; exit 1; }
mkdir -p "$dst"

shopt -s nullglob
for f in "$src"/fig*.png; do
    name=$(basename "$f")
    [[ $name == fig0_* ]] && continue
    magick "$f" -resize '1600x>' -strip -dither None -colors 256 "PNG8:$dst/$name"
    echo "$dst/$name"
done
du -sh "$dst"
