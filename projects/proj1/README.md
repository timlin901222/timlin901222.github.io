# CS180 Project 1 — Prokudin-Gorskii Colorization

Reconstruct color photos from the digitized Prokudin-Gorskii glass plates by
splitting each plate into B/G/R thirds, aligning them, and (optionally)
cleaning the result up with several automatic post-processing steps.

Assignment: https://cal-cs180.github.io/fa26/hw/proj1/index.html

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

Every run also writes the **pre**-bells-and-whistles image (aligned stack
only, no crop/contrast/wb/color-map) to `<outdir>/before/<name>.jpg`, so
you have a matched before/after pair for each plate for the write-up
(disable with `--no-compare`).

## Base pipeline (the deliverable)

- Split into equal B/G/R thirds, top to bottom.
- Align G and R to B using exhaustive search over `[-window, window]`,
  scored with NCC or L2/SSD on raw RGB pixel values, printed as `(x, y)`.
- Small images (`.jpg`) get a single-scale search; large images (`.tif`)
  get a hand-written coarse-to-fine image pyramid (recursion + `np.roll`
  are all custom; only downsampling itself uses `skimage.transform.rescale`).
- Runs in a few seconds per `.jpg` and ~15–20s per `.tif` — all 14 sample
  plates finish in well under 3 minutes total.

## Bells & whistles

All of these are on by default except `--color-map decorrelation`, which is
slower/more experimental. Each is implemented from scratch (plain numpy —
no canned "auto-crop" or "white balance" library call).

| Flag | What it does | Why |
|---|---|---|
| `--no-crop` | disables `auto_crop` | see below |
| `--wb {gray_world,white_patch,none}` (default `gray_world`) | automatic white balance | see below |
| `--no-contrast` | disables `auto_contrast` | see below |
| `--color-map decorrelation` (default `none`) | PCA decorrelation stretch | see below |
| `--features {pixel,gradient}` (default `pixel`) | align on gradient magnitude instead of raw pixels | see below |

**Order applied:** crop → color-map → white balance → contrast.

### 1. Automatic cropping (`auto_crop`)
Detects the plate border instead of chopping a fixed margin. Three signals,
any one of which flags a row/column as border, scanning inward from each
edge:
- **Clipping** — the scanned plate frame is very often saturated (e.g. a
  clipped magenta strip reads `R=1.0, B=1.0` exactly). This turned out to
  be the most reliable signal on this dataset and, unlike the other two,
  doesn't get confused by strongly-colored real content.
- **Channel disagreement** — R/G/B broadly agree inside a well-aligned
  photo; at the edge they don't. Kept as a *stricter* secondary signal,
  since real saturated color (sky, foliage) also disagrees across channels
  and would trip a loose threshold (this bit us during development — see
  below).
- **Flat + extreme** — a uniform strip that's also near pure black/white.
  Requiring *both* matters: a flat sky is also low-texture but isn't near
  0 or 1, so it stays unflagged.

Before/after: `output/before/church.jpg` (big magenta bar across the top)
vs. `output/church.jpg` (clean).

**Known false step during development, worth mentioning in the write-up:**
an earlier version used only "channel disagreement vs. typical interior"
and it misfired on `church.tif` — a blue sky has legitimately different
R/G/B values (that's just its color), so the whole sky got flagged as
"border" and cropped away up to the safety cap. Adding the clipping signal
and requiring flatness to also be near an extreme value fixed it.

### 2. Automatic white balance (`white_balance_gray_world` / `_white_patch`)
Two illuminant estimators, each doing the "shift the estimated illuminant
to neutral" step from Wikipedia's Color Balance article:
- **Gray-world** (default): assumes the scene averages to gray; scales
  each channel so the three channel means match.
- **White-patch**: assumes the brightest surface is the illuminant; scales
  channels so the 99th-percentile pixel becomes white.

### 3. Automatic contrast (`auto_contrast`)
Per-channel percentile stretch (0.5–99.5 by default) so the dark end maps
to 0 and the bright end to 1. Percentiles rather than the literal min/max
pixel so a few scanner dust specks don't skew the whole channel's range.

### 4. Better color mapping (`decorrelation_stretch`, opt-in)
Instead of assuming the plate's three filters map directly to R/G/B, this
computes the principal components of the pixel color distribution (via
`np.linalg.eigh` on the 3×3 color covariance — plain linear algebra, not a
packaged routine) and stretches each component toward a shared target
variance, then rotates back to RGB. The channels of these plates are highly
correlated (mostly all just measuring brightness), so this pulls out the
actual chromatic differences. A `strength` parameter (0-1, default 0.45)
blends between "leave it alone" (0) and "fully equalize" (1) instead of
always fully equalizing: full equalization reliably blows out any plate
that already has real saturated color, turning it neon rather than better
(tried on `melons.tif`, `cathedral.jpg`, `monastery.jpg`, and others at
several strengths — even a well-exposed plate starts looking artificial
past a fairly low ceiling, and no amount of damping rescues `melons.tif`
specifically, since the component it amplifies there is mostly shadow
noise, not color). It's a saturation/vividness effect, not a color
*correction* -- it doesn't know or care what's "right," so it can just as
easily deepen an existing color cast as reveal hidden chromatic signal
(the write-up's before/after uses `siren.tif`, the same plate as the
white-balance example, deliberately: the two operations are independent
and can pull in different directions, which is exactly why color-map runs
before white-balance in the pipeline order above, not after).

### 5. Better features for alignment (`--features gradient`)
Aligns on gradient magnitude instead of raw pixel intensity. This directly
targets the case the spec calls out: `emir.tif`, where the three channels
have genuinely different brightness levels, so raw-pixel NCC/SSD gets
confused, but edges (doors, robe outline, turban) still line up. Compare
`output/emir_pixel.jpg`-style output (regenerate with default `--features
pixel`) against `--features gradient`: the gradient version has visibly
less red/cyan ghosting around edges. Off by default because the
deliverable's base pyramid results are specified to use raw RGB pixel
values.

## Deliverables checklist (from the spec)

- [x] Single-scale results on the small `.jpg` images
- [x] Pyramid results + `(x,y)` offset vectors on every provided `.tif`
- [ ] 3+ extra plates chosen from the LoC collection (https://www.loc.gov/collections/prokudin-gorskii/?st=grid)
- [x] Bells & whistles implemented (all 5 above) — before/after images in `output/before/` vs `output/`
- [ ] Notes on any alignment failures across the 14 provided + your own images
- [ ] Write-up page + submit URL to Google Form / Gradescope
