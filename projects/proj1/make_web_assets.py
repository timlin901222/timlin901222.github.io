"""Resize/copy the chosen images into photos/ for the webpage. One-off script."""
import os
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr

SRC_DST = [
    # Section 1: single-scale (small JPEGs), baseline aligned only
    ("output/before/cathedral.jpg", "photos/single_cathedral.jpg", 700),
    ("output/before/monastery.jpg", "photos/single_monastery.jpg", 700),
    ("output/before/tobolsk.jpg", "photos/single_tobolsk.jpg", 700),

    # Section 2: pyramid, all 14 provided plates, baseline aligned only
    ("output/before/cathedral.jpg", "photos/grid_cathedral.jpg", 420),
    ("output/before/monastery.jpg", "photos/grid_monastery.jpg", 420),
    ("output/before/tobolsk.jpg", "photos/grid_tobolsk.jpg", 420),
    ("output/before/church.jpg", "photos/grid_church.jpg", 420),
    ("output/before/emir.jpg", "photos/grid_emir.jpg", 420),
    ("output/before/harvesters.jpg", "photos/grid_harvesters.jpg", 420),
    ("output/before/icon.jpg", "photos/grid_icon.jpg", 420),
    ("output/before/ilemselga.jpg", "photos/grid_ilemselga.jpg", 420),
    ("output/before/melons.jpg", "photos/grid_melons.jpg", 420),
    ("output/before/religous_painting.jpg", "photos/grid_religous_painting.jpg", 420),
    ("output/before/self_portrait.jpg", "photos/grid_self_portrait.jpg", 420),
    ("output/before/siren.jpg", "photos/grid_siren.jpg", 420),
    ("output/before/three_generations.jpg", "photos/grid_three_generations.jpg", 420),
    ("output/before/wharf.jpg", "photos/grid_wharf.jpg", 420),

    # Section 3: own images, full pipeline
    ("output/own_village_church.jpg", "photos/own_village_church.jpg", 700),
    ("output/own_brick_church.jpg", "photos/own_brick_church.jpg", 700),
    ("output/own_hazy_city.jpg", "photos/own_hazy_city.jpg", 700),
    ("output/own_palace_interior.jpg", "photos/own_palace_interior.jpg", 700),

    # Section 4: bells & whistles, isolated before/after
    ("demo_output/crop_before_church.jpg", "photos/bw_crop_before.jpg", 550),
    ("demo_output/crop_after_church.jpg", "photos/bw_crop_after.jpg", 550),
    ("demo_output/contrast_before_city.jpg", "photos/bw_contrast_before.jpg", 550),
    ("demo_output/contrast_after_city.jpg", "photos/bw_contrast_after.jpg", 550),
    ("demo_output/wb_before_siren.jpg", "photos/bw_wb_before.jpg", 550),
    ("demo_output/wb_after_siren.jpg", "photos/bw_wb_after.jpg", 550),
    ("demo_output/colormap_before_siren.jpg", "photos/bw_colormap_before.jpg", 550),
    ("demo_output/colormap_after_siren.jpg", "photos/bw_colormap_after.jpg", 550),
    ("demo_output/features_before_emir.jpg", "photos/bw_features_before.jpg", 550),
    ("demo_output/features_after_emir.jpg", "photos/bw_features_after.jpg", 550),
]

os.makedirs("photos", exist_ok=True)

for src, dst, max_w in SRC_DST:
    im = skio.imread(src)
    h, w = im.shape[:2]
    if w > max_w:
        scale = max_w / w
        im = sktr.resize(im, (int(h * scale), max_w), anti_aliasing=True)
        im = sk.img_as_ubyte(im)
    skio.imsave(dst, im, quality=85)
    print(f"{src:50s} -> {dst:35s} {im.shape}")

print("done,", len(SRC_DST), "files")
