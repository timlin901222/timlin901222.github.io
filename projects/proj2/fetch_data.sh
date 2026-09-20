#!/usr/bin/env bash
# Pull the images the spec hands out. Everything lands in data/ (gitignored).
#
# Never overwrites: anything already in data/ is left alone, so running this
# again is safe even after you have put your own files there.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data && cd data

have() { [ -s "$1" ]; }

# Fetch $2 to $1, but only if $1 is missing, and only keep it if the server
# actually returned an image. Several links on the spec page are dead and
# answer with an HTML soft-404 instead of a 404 status, which curl -f cannot
# catch -- so check the bytes, not the exit code.
get() {
  local out=$1 url=$2
  if have "$out"; then echo "  keep    $out (already present)"; return 0; fi
  if ! curl -fsSL -o "$out.tmp" "$url"; then
    echo "  FAILED  $out -- download error"; rm -f "$out.tmp"; return 0
  fi
  case "$(file -b --mime-type "$out.tmp")" in
    image/*|application/zip) mv "$out.tmp" "$out"; echo "  got     $out" ;;
    *) echo "  DEAD    $out -- server returned $(file -b --mime-type "$out.tmp"), not an image"
       rm -f "$out.tmp" ;;
  esac
}

get taj.jpg       https://cal-cs180.github.io/fa26/hw/proj2/files/taj.jpg
get cameraman.png https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj2/cameraman.png
get spline.zip    https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj2/spline.zip
[ -f spline.zip ] && unzip -n -q spline.zip && rm -f spline.zip

echo
echo "Still needed by hand (links on the spec page are dead or gated):"
have cameraman.png || echo "  data/cameraman.png  -- 1.2/1.3. Spec link 404s; download it yourself,"
have cameraman.png || echo "                         or: python3 -c \"import skimage.data as d, skimage.io as io; io.imsave('data/cameraman.png', d.camera())\""
have selfie.jpeg  || have selfie.jpg || echo "  data/selfie.jpg     -- your self-portrait, Part 1.1"
have apple.jpeg   || echo "  data/apple.jpeg, data/orange.jpeg -- Parts 2.3/2.4 (spline.zip link is dead)"
have nutmeg.jpg   || echo "  data/DerekPicture.jpg, data/nutmeg.jpg -- Part 2.2, from the Google Drive"
have nutmeg.jpg   || echo "                         link on the spec page"
