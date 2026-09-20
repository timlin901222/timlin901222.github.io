# CS180 Project 2 — Fun with Filters and Frequencies

See [OUTLINE.md](OUTLINE.md) for the deliverable checklist.

## Setup

```bash
cd projects/proj2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./fetch_data.sh
```

## Data

Everything lives in `data/` (gitignored — large, and it shouldn't go in the
website repo). `fetch_data.sh` grabs `taj.jpg`, `cameraman.png`, and the
spline/blending set. You supply:

- `selfie.jpeg` — self-portrait for Part 1.1
- `DerekPicture.jpg`, `nutmeg.jpg` — from the Drive link on the spec page
- your own hybrid pairs and blending pairs

Results go to `output/` (also gitignored); copy the finished figures into
`photos/` when you build the webpage.

## Run

```bash
# Part 1
python convolve.py --image data/selfie.jpeg --bench        # runtime table
python convolve.py --image data/selfie.jpeg --bench --scale 0.5   # if the 4-loop is too slow
python convolve.py --image data/selfie.jpeg                # box / Dx / Dy
python convolve.py --image data/selfie.jpeg --scaling      # runtime vs kernel area
python convolve.py --image data/selfie.jpeg --accum        # 2-loop, tap by tap
python edges.py --sweep 0.05 0.1 0.15 0.2 0.25 0.3        # whole-image sweep
python edges.py --detail 0.1 0.15 0.2 0.25 0.3            # competing regions, zoomed
python edges.py --curve --threshold 0.25                  # edge density vs threshold
python edges.py --threshold 0.25                          # 1.2 final figure
python edges.py --kernels --sigma 2                       # G and the DoG kernels
python edges.py --sigma-sweep                             # pick sigma, equal retention
python edges.py --dog --sigma 2 --threshold 0.042         # 1.3 final figure
python edges.py --compare --sigma 2 --threshold 0.042     # 1.2 vs 1.3
python edges.py --orientation --dog --sigma 2             # 1.3 B&W

# Part 2
python sharpen.py --image data/taj.jpg --sweep 0 0.5 1 2 4
python sharpen.py --image data/sharp.jpg --resharpen --sigma 3 --tag mine

python align.py data/nutmeg.jpg data/DerekPicture.jpg --save derek
python hybrid.py --low output/aligned/derek_2.jpg \
                 --high output/aligned/derek_1.jpg \
                 --sigma-low 8 --sigma-high 5 --tag derek --fft

python stacks.py --image data/apple.jpeg --levels 5 --tag apple
python blend.py --a data/apple.jpeg --b data/orange.jpeg \
                --mask vertical --levels 6 --tag oraple --process
```

## What's stubbed

The plumbing, CLIs, and figure generation are done. The functions the spec
says to write from scratch raise `NotImplementedError` and carry a TODO with
the approach:

| file | function |
|---|---|
| `edges.py` | `orientation_hsv` (B&W) |
| `stacks.py` | `gaussian_stack`, `laplacian_stack` |
| `blend.py` | `blend` |
