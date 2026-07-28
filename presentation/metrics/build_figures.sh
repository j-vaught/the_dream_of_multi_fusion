#!/bin/sh
set -eu

metrics_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
font_dir="/Applications/Microsoft Word.app/Contents/Resources/DFonts"
build_dir="$metrics_dir/figures/build"

mkdir -p "$build_dir"

for source in "$metrics_dir"/figures/0[1-6]_*.typ
do
    stem=$(basename "$source" .typ)
    typst compile \
        --root "$metrics_dir" \
        --font-path "$font_dir" \
        --creation-timestamp 0 \
        "$source" \
        "$build_dir/$stem.pdf"
    typst compile \
        --root "$metrics_dir" \
        --font-path "$font_dir" \
        --creation-timestamp 0 \
        "$source" \
        "$build_dir/$stem.png"
done
