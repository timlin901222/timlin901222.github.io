# Project 2 — Fun with Filters and Frequencies

**Due:** 11:59pm Tue Sep 30, 2026 · **Spec:** https://cal-cs180.github.io/fa26/hw/proj2/index.html
**Scoring (CS180):** Part 1 = 30, Part 2 = 65, clarity/presentation = 5.
Bells & whistles are extra credit for CS180, mandatory for CS280A — the stubs
are in place either way.

**Hard constraint:** every image on the webpage must be under 0.8 MB.
`utils.check_size()` warns you; run it before you publish.

---

## Part 1 — Fun with Filters (30 pts)

### 1.1 Convolutions from scratch (10) → `convolve.py`
- [x] `conv2d_4loop` — four nested loops, zero-padded, kernel flipped
- [x] `conv2d_2loop` — two loops over the kernel, vectorized inner update
- [x] Runtime table: 4-loop vs 2-loop vs `scipy.signal.convolve2d` (`--bench`)
      — verified against scipy for `same`/`full`, odd/even/asymmetric kernels
- [x] Self-portrait (grayscale) × {9×9 box, D_x, D_y} — 1280×1280, real
      `--bench` numbers in the table (box9: 28.96s / 0.187s / 0.159s)
- [x] Write: boundaries, kernel flip, where the time goes
- [x] Figures: loop-order diagram, tap-by-tap accumulation, runtime-vs-kernel-area plot

### 1.2 Finite difference operator (10) → `edges.py`
- [x] cameraman: ∂/∂x, ∂/∂y, gradient magnitude
- [x] Binarized edge image at **t = 0.25**, justified by direct measurement:
      the tower edge peaks at |∇I| = 0.162 = the grass's 90th percentile, so
      no global threshold separates them; 0.25 is the smallest value that gets
      the texture under control (`--sweep`, `--detail`, `--curve`)
- [x] Write: why no global threshold separates grass texture from faint
      structure — the bridge into 1.3

### 1.3 Derivative of Gaussian (10) → `edges.py --dog`
- [x] 2D Gaussian via `cv2.getGaussianKernel()` outer product
- [x] Blur-then-difference: magnitude + binarized at **σ = 2, t = 0.042**
- [x] DoG kernels visualized, incl. centre-row profile (taps sum to 1.4e-17)
- [x] Equivalence verified: max|diff| = 9.69e-16 with `boundary="symm"`
      (breaks to 0.257 at the border under zero padding — noted on the page)
- [x] σ chosen by equal-tower-retention sweep: grass 63.49% → 4.32%, edge
      FWHM 2 px → 5 px (`--sigma-sweep`, `--compare`)
- [x] Write: what changed vs 1.2, incl. that thresholds don't transfer across σ

### 1.3 B&W — gradient orientation → `edges.py --orientation`
- [x] `gradient_angle` — quadrants by hand, no `arctan2`; verified against 8
      known directions + arctan2 oracle on 5000 vectors (max|diff| 8.9e-16)
- [x] Hue = orientation, value = magnitude, colour-wheel key included
- [x] Explain why cyclic hue is the right channel and why value carries |∇I|

---

## Part 2 — Fun with Frequencies (65 pts)

### 2.1 Image sharpening (15) → `sharpen.py`
- [ ] Unsharp mask as a **single** kernel: (1+α)δ − αG; visualize it
- [ ] Taj: original / blurred / high-freq / sharpened
- [ ] ≥1 image of your own, with an α sweep
- [ ] Blur a sharp image then re-sharpen (`--resharpen`); compare to the
      original and explain what does not come back (lost frequencies are lost —
      this is not deconvolution) and what artifacts appear (halos, noise gain)

### 2.2 Hybrid images (20) → `align.py`, `hybrid.py`
- [ ] Derek + Nutmeg
- [ ] ≥2 more of your own (expression change, object morph, time passing, …)
- [ ] One failure case is worth including — say why it failed
- [ ] Full frequency analysis on **one** result: log-magnitude FFT of both
      inputs, both filtered images, and the hybrid (`--fft`)
- [ ] Aligned originals shown; cutoff choices justified

### 2.2 B&W — color
- [ ] Try color on low only / high only / both / neither, report which reads best

### 2.3 Gaussian & Laplacian stacks (10) → `stacks.py`
- [ ] `gaussian_stack` and `laplacian_stack`, no `pyrDown`/`pyramid_gaussian`
- [ ] Verify the Laplacian stack sums back to the original
- [ ] Apple, orange, and the Oraple — recreate Szeliski Fig. 3.42 (a)–(l)

### 2.4 Multiresolution blending (20) → `blend.py`
- [ ] Oraple with a vertical seam, plus one horizontal-seam blend
- [ ] ≥2 blends with irregular masks (Burt & Adelson Fig. 8 style)
- [ ] At least one creative pair of your own, with the concept explained
- [ ] Process figure (`--process`): masked inputs + all three Laplacian stacks,
      laid out like Burt & Adelson Fig. 10
- [ ] Write: why the stack algorithm differs from the pyramid one (no
      resampling, so no expand step, and all levels are pixel-aligned)

### 2.4 B&W — color
- [ ] Color vs grayscale blending; which bands actually need color

---

## Webpage (`index.html`, 5 pts)
- [ ] All results, in spec order, with captions and parameters visible
- [ ] Code snippets for the from-scratch parts
- [ ] **"The most important thing I learned"** paragraph — explicitly graded
- [ ] Link the card on the root `index.html` (flip `Coming soon` → `Live`)
- [ ] Submit the URL via the Google Form on the spec page

---

## Suggested order
1. `./fetch_data.sh`, shoot the self-portrait
2. 1.1 → 1.2 → 1.3 (each builds on the last; `conv2d_2loop` is reused nowhere
   else — Parts 2.x use scipy for speed, which the spec allows)
3. 2.1 (quick win), then 2.2 (alignment/clicking is the slow part — shoot your
   hybrid source photos early)
4. 2.3 before 2.4; 2.4 is just stacks plus a mask
5. Webpage last, but drop each figure into it as you go
