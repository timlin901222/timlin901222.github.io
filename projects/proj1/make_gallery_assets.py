"""Resize the full-pipeline outputs (all bells & whistles applied) into
photos/gallery_*.jpg for gallery.html. One-off script, not part of the
graded pipeline. Re-run after `python colorize.py --all data --outdir output`
if the pipeline changes."""
import os
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr

STEMS = [
    "cathedral", "monastery", "tobolsk",
    "church", "emir", "harvesters", "icon", "ilemselga", "melons",
    "religous_painting", "self_portrait", "siren", "three_generations", "wharf",
    "own_village_church", "own_brick_church", "own_hazy_city", "own_palace_interior",
]

MAX_W = 620

os.makedirs("photos", exist_ok=True)
for stem in STEMS:
    src = f"output/{stem}.jpg"
    dst = f"photos/gallery_{stem}.jpg"
    im = skio.imread(src)
    h, w = im.shape[:2]
    if w > MAX_W:
        scale = MAX_W / w
        im = sk.img_as_ubyte(sktr.resize(im, (int(h * scale), MAX_W), anti_aliasing=True))
    skio.imsave(dst, im, quality=85)
    print(f"{src:30s} -> {dst:35s} {im.shape}")

print("done,", len(STEMS), "files")
