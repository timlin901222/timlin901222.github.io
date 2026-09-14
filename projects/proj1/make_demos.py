"""
One-off script to generate isolated before/after demo images for the
webpage -- each bells & whistle shown on its own, not combined with the
others. Not part of the graded pipeline; safe to delete after the webpage
assets are built. Keep this in sync with what index.html actually
references (see make_web_assets.py) -- it drifted out of sync once before.
"""
import os
import numpy as np
import skimage as sk
import skimage.io as skio

import colorize as c

OUT = "demo_output"
os.makedirs(OUT, exist_ok=True)


def save(rgb, name):
    path = os.path.join(OUT, name)
    skio.imsave(path, sk.img_as_ubyte(np.clip(rgb, 0, 1)))
    print("wrote", path)


def plain_baseline(path, metric="ncc", features="pixel"):
    """Aligned stack only -- no bells & whistles at all."""
    rgb, baseline, info = c.colorize(path, metric=metric, features=features,
                                     crop=False, contrast=False, wb="none",
                                     color_map="none")
    return baseline, info


# ---------------------------------------------------------------------
# 1. Auto-crop: church.tif
# ---------------------------------------------------------------------
base, _ = plain_baseline("data/church.tif")
save(base, "crop_before_church.jpg")
after, box = c.auto_crop(base)
save(after, "crop_after_church.jpg")
print("church crop box", box)

# ---------------------------------------------------------------------
# 2. Auto-contrast: own_hazy_city.jpg (hazy, low dynamic range)
# ---------------------------------------------------------------------
base, _ = plain_baseline("data/own_hazy_city.jpg")
cropped, _ = c.auto_crop(base)          # crop first so the border doesn't skew the stretch
save(cropped, "contrast_before_city.jpg")
after = c.auto_contrast(cropped)
save(after, "contrast_after_city.jpg")

# ---------------------------------------------------------------------
# 3. Auto white balance: siren.tif (strong blue cast; also the plate used
#    in the color-mapping demo below, deliberately -- see index.html §4.4)
# ---------------------------------------------------------------------
base, _ = plain_baseline("data/siren.tif")
cropped, _ = c.auto_crop(base)
save(cropped, "wb_before_siren.jpg")
after = c.white_balance_gray_world(cropped)
save(after, "wb_after_siren.jpg")

# ---------------------------------------------------------------------
# 4. Better color mapping (decorrelation stretch): siren.tif -- the same
#    lilac-bush plate used in the white-balance demo above, deliberately
#    (see index.html §4.4): shows these two corrections are independent
#    and can pull in different directions.
# ---------------------------------------------------------------------
base, _ = plain_baseline("data/siren.tif")
cropped, _ = c.auto_crop(base)
save(cropped, "colormap_before_siren.jpg")
after = c.decorrelation_stretch(cropped)   # uses the default strength
save(after, "colormap_after_siren.jpg")

# ---------------------------------------------------------------------
# 5. Better features (gradient alignment): emir.tif
# ---------------------------------------------------------------------
base_pixel, info_pixel = plain_baseline("data/emir.tif", features="pixel")
save(base_pixel, "features_before_emir.jpg")
base_grad, info_grad = plain_baseline("data/emir.tif", features="gradient")
save(base_grad, "features_after_emir.jpg")
print("emir pixel R offset", info_pixel["r_off"], " gradient R offset", info_grad["r_off"])

print("done")
