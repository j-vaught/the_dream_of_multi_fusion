# The Dream of Multi-Fusion

Radar-camera multi-fusion pipeline that combines marine radar bounding box detections with zero-shot vision models (Grounding DINO, YOLO-World) for self-labeling of buoys and boats in maritime environments.

## Overview

Marine radar provides reliable object detection at range but lacks semantic understanding. Camera imagery provides rich visual context but struggles in adverse conditions. This pipeline fuses both modalities: radar detections define regions of interest, and vision models classify what is in each region at full camera resolution.

## Pipeline Architecture

```
Radar returns (polar)
    |
    v
DBSCAN clustering in East-North space
    |
    v
IMU-corrected projection onto camera image plane
    |
    v
Persistent track assignment (centroid + IoU)
    |
    v
Bounding box crops + buffer at full resolution (5320x3032)
    |
    v
Aspect ratio filter (removes shoreline / land returns)
    |
    v
Zero-shot vision model (Grounding DINO or YOLO-World)
    |
    v
Labeled detections: boat, buoy, vessel
```

## Scripts

### `dream_multi_fusion.py` (Grounding DINO)

Primary pipeline using Grounding DINO for zero-shot detection. Higher accuracy, slower inference. Best suited for GPU servers.

### `dream_multi_fusion_yolo.py` (YOLO-World)

Alternative pipeline using YOLO-World for open-vocabulary detection. 10-20x faster inference, lower memory usage. Suitable for edge devices and Apple Silicon.

## Output Structure

```
output/
  vid1_raw/            Raw camera frames (1920px wide)
  vid2_radar/          Camera frames with radar overlay (bboxes, points, centroids)
  vid3_dino/           Vertically stacked DINO-labeled crops per frame
  crops/               Individual full-resolution crop images
  videos/
    01_input_raw.mp4
    02_radar_bounding.mp4
    03_dino_labeled_crops.mp4
  detections.jsonl     Per-frame metadata: radar bbox, padded bbox, DINO detections
```

### `detections.jsonl` Format

Each line contains a JSON record per frame with the following fields.

```json
{
  "frame_idx": 0,
  "camera_frame": 119,
  "radar_frame": 1,
  "num_radar_dets": 3,
  "num_crops": 2,
  "crops": [
    {
      "track_id": 2,
      "radar_bbox_xyxy": [2539.9, 1639.9, 2917.3, 1742.3],
      "padded_bbox_xyxy": [2389, 1489, 3067, 1892],
      "crop_size": [678, 403],
      "crop_file": "000000_T2.jpg",
      "dino_detections": [
        {"label": "buoy", "score": 0.37, "bbox_xyxy": [180.2, 95.1, 210.8, 280.4]}
      ]
    }
  ]
}
```

## Usage

### Grounding DINO (GPU server recommended)

```bash
CUDA_VISIBLE_DEVICES=3 python dream_multi_fusion.py \
    --frames-jsonl /path/to/frames.jsonl \
    --points-dir /path/to/points/ \
    --rgb-dir /path/to/rgb_out/ \
    --out-dir /path/to/output/ \
    --buffer-px 150 \
    --max-aspect-ratio 2.1 \
    --max-frames 300 \
    --fps 20
```

### YOLO-World (local / edge)

```bash
python dream_multi_fusion_yolo.py \
    --frames-jsonl /path/to/frames.jsonl \
    --points-dir /path/to/points/ \
    --rgb-dir /path/to/rgb_out/ \
    --out-dir /path/to/output/ \
    --buffer-px 150 \
    --classes boat buoy \
    --max-frames 300
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--frames-jsonl` | see script | Radar detection metadata (from projection pipeline) |
| `--points-dir` | see script | Projected radar point files (.npz) |
| `--rgb-dir` | see script | Full-resolution RGB camera images |
| `--out-dir` | see script | Output directory |
| `--buffer-px` | 150 | Pixel padding around radar bbox before cropping |
| `--max-aspect-ratio` | 2.1 | Filters wide crops (shoreline returns) after padding |
| `--box-thresh` / `--conf-thresh` | 0.25 | Detection confidence threshold |
| `--max-frames` | 0 (all) | Limit number of frames to process |
| `--fps` | 20 | Output video frame rate |
| `--crop-width` | 1920 | Width of crop tiles in composite video |
| `--skip-video` | false | Produce frames only, skip ffmpeg encoding |

## Aspect Ratio Filter

Radar returns from shorelines produce very wide, thin bounding boxes that are not useful for object detection. After adding the pixel buffer, any crop with `width / height > max_aspect_ratio` is skipped. With 150px buffer and a ratio of 2.1, typical shoreline returns (ratio ~6.8 after padding) are filtered while real objects (ratio ~1.7-2.0 after padding) pass through.

## Sensor Configuration

| Sensor | Specs |
|--------|-------|
| Camera | Lucid Vision, 5320x3032, ~20 fps |
| Radar | Simrad marine radar, ~1 rev/s |
| IMU | 6-DOF (roll, pitch, yaw + accelerometer) |
| GPS | Lat/lon + heading |

Data was collected on Lake Murray, SC on 2025-10-10.

## Requirements

- Python 3.10+
- PyTorch with CUDA or MPS support
- `transformers` (Hugging Face) for Grounding DINO
- `ultralytics` for YOLO-World
- `Pillow`, `numpy`, `scipy`
- `ffmpeg` (system install)

```bash
pip install torch transformers ultralytics pillow numpy scipy
```

## Upstream Pipeline

This pipeline consumes outputs from the radar-camera projection clustering pipeline (`project_radar_camera_clust_preproj_fast.py`), which performs the following.

1. Loads marine radar returns from MATLAB .mat files.
2. Clusters radar points in East-North space using DBSCAN.
3. Projects clusters onto the camera image plane using IMU-corrected rigid body transforms and camera intrinsics.
4. Assigns persistent track IDs across frames using centroid distance and bounding box IoU.
5. Emits `frames.jsonl` with per-frame detection metadata and `.npz` files with projected point coordinates.

## Author

J.C. Vaught
