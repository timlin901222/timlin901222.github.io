# CS180 Project 1 — Prokudin-Gorskii Colorization

## Setup

```bash
cd projects/proj1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Put the images in `data/` (gitignored — they're large and shouldn't go in the
website repo). Expected layout:

```
data/
  cathedral.jpg  monastery.jpg  tobolsk.jpg      # small, single-scale
  church.tif  emir.tif  harvesters.tif  ...       # large, pyramid
```

## Run

```bash
python colorize.py data/cathedral.jpg                    # one image -> output/cathedral.jpg
python colorize.py data/emir.tif --features gradient      # for brightness-mismatched plates
python colorize.py --all data --outdir output             # batch everything, prints offsets
```
