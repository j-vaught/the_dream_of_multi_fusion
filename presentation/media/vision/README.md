# Vision-only presentation media

This directory contains deterministic, text-free presentation assets for the
vision-only detection experiment. The correct-comparison asset holds camera
frame 119 for five seconds at 60 frames per second. It shows solid
class-colored predictions over dashed ground-truth boxes for the matched boat
and buoy.

The false-positive montage uses verified context around camera frames 341 and
416. Both error frames receive a short hold so the solid red duplicate boat
boxes remain visible during playback. The montage preserves the source zoom
and contains no labels, confidence values, captions, or other burned-in text.

Run the build with the project environment.

```sh
uv run python presentation/media/vision/build_vision_media.py
```

Run verification without rewriting the outputs.

```sh
uv run python presentation/media/vision/build_vision_media.py --verify-only
```

The manifest records source hashes, output hashes, source-frame indices,
camera-frame indices, decoded frame hashes, matching coordinates, timing, and
the exact output timeline.
