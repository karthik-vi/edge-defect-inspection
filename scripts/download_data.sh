#!/usr/bin/env bash
# Download only 3 MVTec AD categories (CC BY-NC-SA 4.0) into data/mvtec_ad/. Never the full dataset.
set -euo pipefail
DEST="${1:-data/mvtec_ad}"; mkdir -p "$DEST"
declare -A URL=(
 [bottle]="https://www.mydrive.ch/shares/163640/9a565b9a02f45cd135e46b1c36e01ec5/download/420937370-1629958698/bottle.tar.xz"
 [metal_nut]="https://www.mydrive.ch/shares/163647/6d5a833cdee9035a45d5ce0921ecd24d/download/420937637-1629959294/metal_nut.tar.xz"
 [screw]="https://www.mydrive.ch/shares/163649/cfd394b4e3a49013745fab9e2084988a/download/420938130-1629960389/screw.tar.xz")
for c in bottle metal_nut screw; do
  [ -d "$DEST/$c/test" ] && { echo "$c present"; continue; }
  curl -fSL --retry 3 -o "$DEST/$c.tar.xz" "${URL[$c]}"
  tar -xJf "$DEST/$c.tar.xz" -C "$DEST" && rm "$DEST/$c.tar.xz"
  echo "$c done"
done
