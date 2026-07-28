# Radar-bounded presentation media

This directory builds seven deterministic, text-free presentation videos from the shared 300-frame evaluation sequence. Every output is a single H.264 `yuv420p` MP4 with no audio. Complete sequences contain 300 frames at 60 frames per second and therefore run for five seconds.

The fixed method order is vision-only, radar-confidence-gated, and radar-bounded from left to right. Atlantic, Honeycomb, and Garnet square borders encode that order. Dashed white boxes show ground truth. Garnet and Grass solid boxes show correct boat and buoy predictions. Rose solid boxes show false positives. The videos contain no labels, confidence values, captions, or other burned-in text.

Run the build from the repository root.

```sh
uv run python presentation/media/radar_bounded/build_media.py
```

The default build uses `01_input_raw.mp4`, which is the clean 300-frame sequence derived from the full-resolution PNG inputs. To render directly from original frames, provide their directory.

```sh
uv run python presentation/media/radar_bounded/build_media.py \
  --rgb-dir /path/to/rgb_out
```

The generated `assets/manifest.json` indexes the seven videos. Each adjacent asset manifest records source-file hashes, exact frame selections, selection hashes, uncompressed rendered-frame hashes, final MP4 hashes, dimensions, frame rate, codec, pixel format, and audio status.

The outputs are `01_radar_inspection_zones.mp4`, `02_full_frame_extracted_crop_composite.mp4`, `03_radar_bounded_false_positive_episodes.mp4`, `04_radar_bounded_correct_predictions_over_ground_truth.mp4`, `05_three_method_full_frame_composite.mp4`, `06_three_method_zoom_composite.mp4`, and `07_method_disagreement_montage.mp4`.
