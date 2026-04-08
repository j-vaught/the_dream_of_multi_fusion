# The Dream of Multi-Fusion

Radar-camera multi-fusion pipeline that combines radar bounding box detections with Grounding DINO zero-shot object detection for self-labeling of buoys and boats.

## Pipeline

1. **Radar bounding boxes** are read from a pre-computed `frames.jsonl` (produced by the radar-camera projection clustering pipeline).
2. Each radar bounding box is **cropped at full camera resolution** with a configurable pixel buffer.
3. **Grounding DINO** runs on each full-resolution crop to detect and label boats and buoys.
4. Three output videos are produced.

| Video | Description |
|-------|-------------|
| `01_input_raw.mp4` | Raw camera input frames |
| `02_radar_bounding.mp4` | Camera frames with radar bounding boxes, centroids, and projected radar points |
| `03_dino_labeled_crops.mp4` | Per-frame composite of DINO-labeled crops (radar bbox + buffer) at full resolution |

## Usage

```bash
python dream_multi_fusion.py \
    --frames-jsonl /path/to/frames.jsonl \
    --points-dir /path/to/points/ \
    --rgb-dir /path/to/rgb_out/ \
    --out-dir /path/to/output/ \
    --buffer-px 150 \
    --fps 20
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--frames-jsonl` | `all_projections_fast/compute/frames.jsonl` | Radar detection metadata |
| `--points-dir` | `all_projections_fast/compute/points` | Projected radar point files (.npz) |
| `--rgb-dir` | `lucid_20251010_125220/cam1/rgb_out` | Full-resolution RGB images |
| `--out-dir` | `dream_multi_fusion_output` | Output directory |
| `--buffer-px` | 150 | Pixel padding around radar bbox for DINO crops |
| `--model-id` | `IDEA-Research/grounding-dino-base` | Grounding DINO model |
| `--box-thresh` | 0.25 | DINO box confidence threshold |
| `--text-thresh` | 0.25 | DINO text confidence threshold |
| `--dino-prompt` | `boat . buoy . vessel . navigation buoy .` | Zero-shot detection prompt |
| `--max-frames` | 0 (all) | Limit number of frames to process |
| `--fps` | 20 | Output video frame rate |
| `--crop-height` | 720 | Fixed height for crop tiles in composite video |
| `--skip-video` | false | Produce frames only, skip ffmpeg encoding |

## Requirements

- Python 3.10+
- PyTorch with MPS (Apple Silicon) or CUDA support
- `transformers` (Hugging Face)
- `Pillow`
- `numpy`
- `ffmpeg` (system install)

```bash
pip install torch transformers pillow numpy
```

## Data

This pipeline expects pre-computed radar-camera projection outputs. The projection pipeline clusters radar returns in East-North space, projects them onto the camera image plane using IMU-corrected transforms, and assigns persistent track IDs across frames.

## Author

J.C. Vaught
