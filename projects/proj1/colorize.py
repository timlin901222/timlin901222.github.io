import argparse
import os
import time

import numpy as np
import skimage as sk
import skimage.io as skio


def split_channels(plate):
    h = plate.shape[0] // 3
    b = plate[0:h]
    g = plate[h:2 * h]
    r = plate[2 * h:3 * h]
    # make all three exactly the same shape
    w = min(b.shape[1], g.shape[1], r.shape[1])
    return b[:, :w], g[:, :w], r[:, :w]


def inner_crop(img, frac=0.08):
    #Return the central region of img, dropping `frac` off every edge.
    
    h, w = img.shape[:2]
    dh, dw = int(h * frac), int(w * frac)
    return img[dh:h - dh, dw:w - dw]


# ==========================================================================
# Similarity metrics
# ==========================================================================
def ssd(a, b):
    return -np.sum((a - b) ** 2)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.sum((a / np.linalg.norm(a)) * (b / np.linalg.norm(b))))


METRICS = {"ncc": ncc, "ssd": ssd}


def _features(img, mode):
    if mode == "gradient":
        gy, gx = np.gradient(img)
        return np.hypot(gx, gy)
    return img


# ==========================================================================
# Single-scale exhaustive alignment
# ==========================================================================
def single_scale_align(mov, ref, window=15, metric="ncc", base=(0, 0), features="pixel"):
    """Search integer shifts (dy, dx) in [-window, window] around `base`.

    Returns the (dy, dx) that best aligns `mov` onto `ref`.
    """
    score_fn = METRICS[metric]
    mov_f = _features(mov, features)
    ref_f = _features(ref, features)
    ref_c = inner_crop(ref_f)
    best_score, best_shift = -np.inf, base
    for dy in range(base[0] - window, base[0] + window + 1):
        for dx in range(base[1] - window, base[1] + window + 1):
            shifted = np.roll(mov_f, shift=(dy, dx), axis=(0, 1))
            s = score_fn(inner_crop(shifted), ref_c)
            if s > best_score:
                best_score, best_shift = s, (dy, dx)
    return best_shift


def pyramid_align(mov, ref, metric="ncc", min_size=256, coarse_window=15,
                  fine_window=4, features="pixel"):
    """Recursively align on a Gaussian pyramid.

    Coarsest level does a full +/-coarse_window search; each finer level only
    refines within +/-fine_window of the (doubled) estimate from above.
    Downsampling itself uses skimage's rescale (an allowed "resizing"
    utility) -- the pyramid recursion and the search are hand-written.
    """
    if min(mov.shape) <= min_size:
        return single_scale_align(mov, ref, window=coarse_window, metric=metric,
                                  features=features)

    mov_small = sk.transform.rescale(mov, 0.5, anti_aliasing=True)
    ref_small = sk.transform.rescale(ref, 0.5, anti_aliasing=True)

    dy, dx = pyramid_align(mov_small, ref_small, metric=metric,
                           min_size=min_size, coarse_window=coarse_window,
                           fine_window=fine_window, features=features)
    dy, dx = dy * 2, dx * 2                    # scale estimate up to this level

    return single_scale_align(mov, ref, window=fine_window, metric=metric,
                              base=(dy, dx), features=features)


def align(mov, ref, metric="ncc", features="pixel"):
    """Pick exhaustive vs. pyramid automatically based on image size."""
    if max(mov.shape) > 400:
        return pyramid_align(mov, ref, metric=metric, features=features)
    return single_scale_align(mov, ref, window=15, metric=metric, features=features)


# ==========================================================================
# Bells & whistle: automatic cropping (detect the border, don't guess a margin)
# ==========================================================================
def _dilate(mask, size):
    """OR each position with its `size`-wide neighborhood.

    The plate border is rarely one clean contiguous band: a thin near-black
    strip often separates the true edge from the colored misalignment fringe
    just inside it. Dilating bridges that gap so the scan below doesn't stop
    early, in the middle of the border, at that one clean-looking pixel.
    """
    if size <= 1:
        return mask
    n = len(mask)
    k = size // 2
    out = np.zeros_like(mask)
    for i in range(n):
        out[i] = mask[max(0, i - k):min(n, i + k + 1)].any()
    return out


def _scan_border(is_border_1d, max_margin):
    n = len(is_border_1d)
    limit = min(max_margin, n // 2 - 1) if n > 2 else 0
    i = 0
    while i < limit and is_border_1d[i]:
        i += 1
    return i


def auto_crop(rgb, max_frac=0.12, disagree_tol=2.2, flat_tol=0.35,
             sat_frac=0.5, dilate=35, dark_thresh=0.08):

    h, w = rgb.shape[:2]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    disagreement = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
    gray = rgb.mean(axis=2)
    clipped = (rgb.max(axis=2) > 0.995) | (rgb.min(axis=2) < 0.005)

    max_h, max_w = max(1, int(h * max_frac)), max(1, int(w * max_frac))
    if h - 2 * max_h < 4 or w - 2 * max_w < 4:
        return rgb, (0, 0, 0, 0)
    interior_rows = slice(max_h, h - max_h)
    interior_cols = slice(max_w, w - max_w)

    row_disagree = np.median(disagreement, axis=1)
    col_disagree = np.median(disagreement, axis=0)
    row_std = gray.std(axis=1)
    col_std = gray.std(axis=0)
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    row_clip = clipped.mean(axis=1)
    col_clip = clipped.mean(axis=0)

    typ_row_dis = np.median(row_disagree[interior_rows]) + 1e-8
    typ_col_dis = np.median(col_disagree[interior_cols]) + 1e-8
    typ_row_std = np.median(row_std[interior_rows]) + 1e-8
    typ_col_std = np.median(col_std[interior_cols]) + 1e-8

    extreme = lambda m: (m < 0.15) | (m > 0.92)
    row_is_border = (row_clip > sat_frac) | (row_mean < dark_thresh) | \
        (row_disagree > disagree_tol * typ_row_dis) | \
        ((row_std < flat_tol * typ_row_std) & extreme(row_mean))
    col_is_border = (col_clip > sat_frac) | (col_mean < dark_thresh) | \
        (col_disagree > disagree_tol * typ_col_dis) | \
        ((col_std < flat_tol * typ_col_std) & extreme(col_mean))

    row_is_border = _dilate(row_is_border, dilate)
    col_is_border = _dilate(col_is_border, dilate)

    top = _scan_border(row_is_border, max_h)
    bottom = _scan_border(row_is_border[::-1], max_h)
    left = _scan_border(col_is_border, max_w)
    right = _scan_border(col_is_border[::-1], max_w)

    cropped = rgb[top:h - bottom, left:w - right]
    if min(cropped.shape[:2]) < 10:            # safety net on pathological images
        return rgb, (0, 0, 0, 0)
    return cropped, (top, bottom, left, right)


# ==========================================================================
# Bells & whistle: automatic contrast
# ==========================================================================
def auto_contrast(rgb, low_pct=0.5, high_pct=99.5):
    """Per-channel percentile stretch: low_pct -> 0, high_pct -> 1.

    Percentile clipping (rather than the literal single darkest/brightest
    pixel) keeps a handful of scanner dust specks or sensor noise from
    skewing the whole channel's range.
    """
    out = np.empty_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1e-6
        out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0.0, 1.0)
    return out


# ==========================================================================
# Bells & whistle: automatic white balance
# ==========================================================================
def white_balance_gray_world(rgb):
    """Gray-world: assume the scene averages out to gray, scale channels to match."""
    means = rgb.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    scale = gray / np.clip(means, 1e-6, None)
    return np.clip(rgb * scale, 0.0, 1.0)


def white_balance_white_patch(rgb, pct=99):
    """White-patch/max-white: assume the brightest surface is the illuminant color."""
    ref = np.percentile(rgb.reshape(-1, 3), pct, axis=0)
    scale = 1.0 / np.clip(ref, 1e-6, None)
    return np.clip(rgb * scale, 0.0, 1.0)


WHITE_BALANCE = {
    "gray_world": white_balance_gray_world,
    "white_patch": white_balance_white_patch,
}


# ==========================================================================
# Bells & whistle: better color mapping (decorrelation stretch)
# ==========================================================================
def decorrelation_stretch(rgb, strength=0.45):
    """PCA-based color mapping instead of assuming the filters map to R/G/B.

    """
    h, w, _ = rgb.shape
    x = rgb.reshape(-1, 3).astype(np.float64)
    mean = x.mean(axis=0)
    xc = x - mean

    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, 1e-8, None)

    orig_std = np.sqrt(eigval)
    target_std = orig_std.mean()                # fully-equalized scale
    scale = orig_std * (1 - strength) + target_std * strength

    whitened = (xc @ eigvec) / orig_std
    stretched = whitened * scale

    out = stretched @ eigvec.T + mean
    return np.clip(out.reshape(h, w, 3), 0.0, 1.0)


# ==========================================================================
# Top-level driver
# ==========================================================================
def colorize(path, metric="ncc", features="pixel",
            crop=True, contrast=True, wb="gray_world", color_map="none"):
    """Run the full pipeline on one plate file.

    Returns (final_rgb, baseline_rgb, info). `baseline_rgb` is the
    pre-bells-and-whistles result (aligned stack only) for before/after
    comparisons; `info` carries the offsets to print / log.
    """
    plate = sk.img_as_float(skio.imread(path))
    b, g, r = split_channels(plate)

    g_off = align(g, b, metric=metric, features=features)
    r_off = align(r, b, metric=metric, features=features)
    g_a = np.roll(g, shift=g_off, axis=(0, 1))
    r_a = np.roll(r, shift=r_off, axis=(0, 1))

    baseline = np.clip(np.dstack([r_a, g_a, b]), 0.0, 1.0)
    rgb = baseline

    crop_box = (0, 0, 0, 0)
    if crop:
        rgb, crop_box = auto_crop(rgb)
    if color_map == "decorrelation":
        rgb = decorrelation_stretch(rgb)
    if wb != "none":
        rgb = WHITE_BALANCE[wb](rgb)
    if contrast:
        rgb = auto_contrast(rgb)

    info = dict(g_off=g_off, r_off=r_off, crop_box=crop_box)
    return rgb, baseline, info


def _save(rgb, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    skio.imsave(out_path, sk.img_as_ubyte(rgb))


def _to_xy(dy_dx):
    """(dy, dx) -> (x, y), since the spec asks for the (x,y) displacement."""
    dy, dx = dy_dx
    return (dx, dy)


def _process_one(in_path, out_path, opts, compare=True):
    t0 = time.time()
    rgb, baseline, info = colorize(in_path, **opts)
    _save(rgb, out_path)
    if compare:
        before_path = os.path.join(os.path.dirname(out_path), "before",
                                   os.path.basename(out_path))
        _save(baseline, before_path)

    name = os.path.basename(in_path)
    g_str = str(_to_xy(info["g_off"]))
    r_str = str(_to_xy(info["r_off"]))
    top, bottom, left, right = info["crop_box"]
    print(f"{name:<24}  G (x,y) {g_str:<22}  R (x,y) {r_str:<22}"
          f"  crop(T{top},B{bottom},L{left},R{right})"
          f"  ({time.time() - t0:5.1f}s)  -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="path to one plate image")
    ap.add_argument("--out", help="output path for the single-image case")
    ap.add_argument("--all", metavar="DIR",
                    help="process every .jpg/.tif in DIR")
    ap.add_argument("--outdir", default="output",
                    help="output directory for --all (default: output/)")
    ap.add_argument("--metric", choices=list(METRICS), default="ncc")
    ap.add_argument("--features", choices=["pixel", "gradient"], default="pixel",
                    help="score alignment on raw pixels (deliverable default) "
                        "or gradient magnitude (bells & whistles, for plates "
                        "like emir.tif with mismatched channel brightness)")
    ap.add_argument("--no-crop", dest="crop", action="store_false",
                    help="disable automatic border cropping")
    ap.add_argument("--no-contrast", dest="contrast", action="store_false",
                    help="disable automatic contrast stretch")
    ap.add_argument("--wb", choices=["gray_world", "white_patch", "none"],
                    default="gray_world", help="automatic white balance method")
    ap.add_argument("--color-map", choices=["none", "decorrelation"], default="none",
                    help="bells & whistles: PCA decorrelation stretch instead of "
                        "assuming the plate's filters are literally R/G/B")
    ap.add_argument("--no-compare", dest="compare", action="store_false",
                    help="don't also save the pre-bells-and-whistles image to "
                        "<outdir>/before/ (saved by default, for the write-up)")
    args = ap.parse_args()

    opts = dict(metric=args.metric, features=args.features,
               crop=args.crop, contrast=args.contrast, wb=args.wb,
               color_map=args.color_map)

    if args.all:
        exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        files = sorted(f for f in os.listdir(args.all)
                       if f.lower().endswith(exts))
        if not files:
            ap.error(f"no images found in {args.all}")
        for f in files:
            stem = os.path.splitext(f)[0]
            _process_one(os.path.join(args.all, f),
                         os.path.join(args.outdir, f"{stem}.jpg"),
                         opts, compare=args.compare)
        return

    if not args.image:
        ap.error("give an image path, or use --all DIR")
    out = args.out or os.path.join(
        "output", os.path.splitext(os.path.basename(args.image))[0] + ".jpg")
    _process_one(args.image, out, opts, compare=args.compare)


if __name__ == "__main__":
    main()
