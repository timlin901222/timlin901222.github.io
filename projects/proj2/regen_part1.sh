#!/usr/bin/env bash
# Regenerate every Part 1 figure with the parameters quoted on the webpage,
# then copy them into photos/. One command, so the page can never drift out of
# sync with the data after a source image is replaced.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-./.venv/bin/python}
SELFIE=${SELFIE:-data/selfie.jpeg}

echo "== 1.1 =="
$PY convolve.py --image "$SELFIE" --bench
$PY convolve.py --image "$SELFIE"
$PY convolve.py --image "$SELFIE" --accum
$PY convolve.py --image "$SELFIE" --scale 0.4 --scaling

echo "== 1.2  (t = 0.25) =="
$PY edges.py --threshold 0.25
$PY edges.py --sweep 0.10 0.15 0.20 0.25 0.30 0.35
$PY edges.py --detail 0.10 0.15 0.20 0.25 0.30
$PY edges.py --curve --threshold 0.25

echo "== 1.3  (sigma = 2, t = 0.042) =="
$PY edges.py --kernels --sigma 2
$PY edges.py --sigma-sweep
$PY edges.py --dog --sigma 2 --threshold 0.042
$PY edges.py --equivalence --sigma 2 --threshold 0.042
$PY edges.py --compare --sigma 2 --threshold 0.042
$PY edges.py --orientation --dog --sigma 2

cp output/p1*.jpg photos/
rm -f photos/p13_orientation_full.jpg photos/p13_orientation_dog_full.jpg
echo
echo "copied to photos/ — check the numbers quoted in index.html still match"
$PY -c "
import glob, utils
for f in sorted(glob.glob('photos/*.jpg')):
    mb = utils.check_size(f)
    print(f'  {f:<40} {mb:.2f} MB')"
