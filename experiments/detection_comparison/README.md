# Detection comparison

This directory records the three object-detection experiments on camera frames
119 through 418. All methods were evaluated against the same YOLO-style ground
truth at an intersection-over-union threshold of 0.5.

The original full-frame vision run failed because each 5320 by 3032 image was
reduced to approximately 1333 pixels on its long edge. Most buoy annotations
then occupied only one to four model-input pixels. It also used a shorter prompt,
a higher confidence threshold, incomplete class normalization, and no
class-aware non-maximum suppression.

The corrected vision-only method divides every full-resolution frame into nine
overlapping 2048 by 1600 tiles. It uses the same descriptive Grounding DINO
prompt and image scale as the radar-crop pipeline, then merges duplicate
detections with class-aware non-maximum suppression. It does not use radar.

The radar-confidence-gated method starts with the same whole-image tiled
detections. Detections above 0.18 are accepted everywhere. Lower-confidence
detections are retained only when their center or overlap is supported by a
padded radar return. Candidate thresholds from 0.10 through 0.18 were compared
on the validation split. The selected threshold was 0.16. The test split was
not used to select it.

The radar-bounded method runs the detector only on padded radar-return crops.
This magnifies tiny objects but can also magnify clutter and create additional
false positives.

## Test results

| Method | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision only, tiled | 73 | 1 | 287 | 0.986 | 0.203 | 0.336 |
| Radar confidence gated | 83 | 5 | 277 | 0.943 | 0.231 | 0.371 |
| Radar-bounded crops | 90 | 61 | 270 | 0.596 | 0.250 | 0.352 |

All three methods detect every test-frame boat. The confidence-gated method
improves buoy true positives from 28 to 38 while adding four net false
positives. Radar-bounded crops recover seven additional buoy instances, but
produce 56 more false positives than confidence gating.

## Reproduction

Install the experiment dependencies.

```sh
uv sync --group experiments
```

Run the two inference shards on separate GPUs.

```sh
uv run python run_detection_experiments.py \
  --output out/experiments/detection_comparison/shard_000_149.jsonl \
  --start-index 0 --end-index 150 --gpu-id 0

uv run python run_detection_experiments.py \
  --output out/experiments/detection_comparison/shard_150_299.jsonl \
  --start-index 150 --end-index 300 --gpu-id 1
```

Merge the shards, select the radar-gate threshold on validation data, and score
all methods.

```sh
uv run python evaluate_detection_experiments.py \
  --shard out/experiments/detection_comparison/shard_000_149.jsonl \
  --shard out/experiments/detection_comparison/shard_150_299.jsonl
```

`metrics.json` contains all, train, validation, and test metrics plus the full
validation threshold sweep. The JSONL files preserve raw and final detections
for frame-level review.

## Evaluation videos

Render aligned full-resolution videos for all three methods.

```sh
uv run python render_detection_experiment_videos.py
```

The videos use solid magenta for true-positive boats, solid yellow for
true-positive buoys, and solid red for false positives. False-negative
ground-truth boxes use a dashed magenta outline with an X. The renderer includes
no labels or confidence text so object edges remain visible.

A crisp three-times zoom around the central boat and nearby buoys can be
rendered directly from the raw frames.

```sh
uv run python render_detection_experiment_videos.py \
  --crop 2550 1300 600 500 \
  --scale-factor 3 \
  --line-width 2 \
  --output-dir out/videos/detection_experiments_zoom
```
