#!/usr/bin/env python3
"""
Re-project DINO crop detections back into the full camera frame.

For each frame:
  - Draw radar bounding boxes (cyan)
  - Draw re-projected DINO detections (garnet for boat, atlantic for buoy)
  - Calculate IoU between each radar bbox and the best-matching DINO detection
  - Produce a 4th video: full frame with both radar + DINO overlaid
  - Write iou_results.jsonl with per-detection IoU metrics
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-project DINO detections and compute IoU")
    p.add_argument("--detections-jsonl",
                    default="/Volumes/WorkDrive/10_10Data/dream_multi_fusion_output/detections.jsonl")
    p.add_argument("--rgb-dir",
                    default="/Volumes/WorkDrive/10_10Data/lucid_20251010_125220/cam1/rgb_out")
    p.add_argument("--points-dir",
                    default="/Volumes/WorkDrive/10_10Data/all_projections_fast/compute/points")
    p.add_argument("--frames-jsonl",
                    default="/Volumes/WorkDrive/10_10Data/all_projections_fast/compute/frames.jsonl")
    p.add_argument("--out-dir",
                    default="/Volumes/WorkDrive/10_10Data/dream_multi_fusion_output")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--skip-video", action="store_true")
    return p.parse_args()


def get_font(size: int = 28):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def iou(a: List[float], b: List[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def reproject_dino_box(dino_bbox: List[float], crop_origin: List[int]) -> List[float]:
    """Offset a DINO bbox from crop coords to full-image coords."""
    ox, oy = crop_origin[0], crop_origin[1]
    return [
        dino_bbox[0] + ox,
        dino_bbox[1] + oy,
        dino_bbox[2] + ox,
        dino_bbox[3] + oy,
    ]


RADAR_COLOR = (0, 255, 255)          # cyan
DINO_COLORS = {
    "boat": (115, 0, 10),            # garnet
    "buoy": (70, 106, 159),          # atlantic
}
DINO_DEFAULT = (206, 211, 24)        # grass
IOU_COLOR_GOOD = (101, 120, 11)      # horseshoe (high IoU)
IOU_COLOR_LOW = (204, 46, 64)        # rose (low IoU)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    vid4_dir = out_dir / "vid4_fused"
    video_dir = out_dir / "videos"

    for d in [vid4_dir, video_dir]:
        if d.exists():
            for f in d.glob("[0-9]*.*"):
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    # Load detection records
    det_records = []
    with open(args.detections_jsonl, "r") as f:
        for line in f:
            det_records.append(json.loads(line))
    print(f"Loaded {len(det_records)} frame records")

    # Load original frames.jsonl for radar points
    frames_lookup = {}
    with open(args.frames_jsonl, "r") as f:
        for line in f:
            rec = json.loads(line)
            frames_lookup[rec["camera_frame"]] = rec

    rgb_dir = Path(args.rgb_dir)
    points_dir = Path(args.points_dir)
    font = get_font(28)
    font_sm = get_font(20)

    iou_results_path = out_dir / "iou_results.jsonl"
    iou_file = iou_results_path.open("w", encoding="utf-8")

    all_ious = []

    for det_rec in det_records:
        idx = det_rec["frame_idx"]
        cf = det_rec["camera_frame"]
        rf = det_rec["radar_frame"]
        tag = f"[{idx + 1}/{len(det_records)}] RF{rf:04d} CF{cf}"

        img_path = rgb_dir / f"{cf}_rgb.png"
        if not img_path.exists():
            print(f"{tag}: image missing, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)

        # Draw radar points
        orig_rec = frames_lookup.get(cf)
        if orig_rec:
            npz_path = points_dir / orig_rec["points_file"]
            if npz_path.exists():
                data = np.load(npz_path)
                u, v, inten = data["u"], data["v"], data["intensity"]
                for det in orig_rec["detections"]:
                    s, e = int(det["point_start"]), int(det["point_end"])
                    for px, py, pi in zip(u[s:e], v[s:e], inten[s:e]):
                        t = float(np.clip(float(pi) / 17.0, 0.0, 1.0))
                        c = int(np.clip(round(255.0 * t), 0, 255))
                        r = 2.0
                        draw.ellipse((float(px) - r, float(py) - r,
                                      float(px) + r, float(py) + r),
                                     fill=(255, c, 0))

        frame_iou_records = []

        for crop_rec in det_rec["crops"]:
            track_id = crop_rec["track_id"]
            radar_bbox = crop_rec["radar_bbox_xyxy"]
            padded_bbox = crop_rec["padded_bbox_xyxy"]
            crop_origin = [padded_bbox[0], padded_bbox[1]]

            # Draw radar bbox
            draw.rectangle(radar_bbox, outline=RADAR_COLOR, width=4)
            cx_r = (radar_bbox[0] + radar_bbox[2]) / 2
            cy_r = (radar_bbox[1] + radar_bbox[3]) / 2
            cr = 9
            draw.line((cx_r - cr, cy_r - cr, cx_r + cr, cy_r + cr),
                      fill=(0, 255, 0), width=3)
            draw.line((cx_r - cr, cy_r + cr, cx_r + cr, cy_r - cr),
                      fill=(0, 255, 0), width=3)

            # Re-project and draw DINO detections
            best_iou = 0.0
            best_dino = None

            for dino_det in crop_rec["dino_detections"]:
                full_bbox = reproject_dino_box(dino_det["bbox_xyxy"], crop_origin)
                label = dino_det["label"]
                score = dino_det["score"]
                color = DINO_COLORS.get(label, DINO_DEFAULT)

                draw.rectangle(full_bbox, outline=color, width=3)
                draw.text((int(full_bbox[0]) + 2, max(0, int(full_bbox[1]) - 24)),
                          f"{label} {score:.2f}", fill=color, font=font_sm)

                # IoU with radar bbox
                det_iou = iou(radar_bbox, full_bbox)
                if det_iou > best_iou:
                    best_iou = det_iou
                    best_dino = {
                        "label": label,
                        "score": score,
                        "bbox_xyxy": full_bbox,
                        "iou": det_iou,
                    }

                all_ious.append(det_iou)

            # Draw IoU label on radar bbox
            iou_color = IOU_COLOR_GOOD if best_iou > 0.1 else IOU_COLOR_LOW
            draw.text((int(radar_bbox[0]), max(0, int(radar_bbox[1]) - 56)),
                      f"T{track_id} IoU:{best_iou:.2f}",
                      fill=iou_color, font=font)

            frame_iou_records.append({
                "track_id": track_id,
                "radar_bbox_xyxy": radar_bbox,
                "best_iou": best_iou,
                "best_dino_match": best_dino,
                "num_dino_dets": len(crop_rec["dino_detections"]),
            })

        # Save frame
        resized = img.resize((1920, int(1920 * img_h / img_w)), Image.LANCZOS)
        resized.save(vid4_dir / f"{idx:06d}.jpg", quality=95)

        # Write IoU record
        iou_file.write(json.dumps({
            "frame_idx": idx,
            "camera_frame": cf,
            "radar_frame": rf,
            "detections": frame_iou_records,
        }) + "\n")
        iou_file.flush()

        avg_iou = np.mean([d["best_iou"] for d in frame_iou_records]) if frame_iou_records else 0
        print(f"{tag}: {len(frame_iou_records)} tracks, avg IoU={avg_iou:.3f}")

        del img, resized

    iou_file.close()

    # Print summary
    if all_ious:
        arr = np.array(all_ious)
        print(f"\n--- IoU Summary ({len(arr)} total pairs) ---")
        print(f"  Mean:   {arr.mean():.4f}")
        print(f"  Median: {np.median(arr):.4f}")
        print(f"  >0.0:   {(arr > 0.0).sum()} ({100*(arr > 0.0).mean():.1f}%)")
        print(f"  >0.1:   {(arr > 0.1).sum()} ({100*(arr > 0.1).mean():.1f}%)")
        print(f"  >0.3:   {(arr > 0.3).sum()} ({100*(arr > 0.3).mean():.1f}%)")
        print(f"  >0.5:   {(arr > 0.5).sum()} ({100*(arr > 0.5).mean():.1f}%)")

    # Encode video
    if args.skip_video:
        print("Skipping video encoding.")
        return

    pattern = str(vid4_dir / "%06d.jpg")
    dst = str(video_dir / "04_fused_iou.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        dst,
    ]
    print(f"Encoding 04_fused_iou.mp4...")
    subprocess.run(cmd, check=True)
    print(f"  -> {dst}")

    print(f"\nDone. Fused overlay + IoU in {out_dir}")


if __name__ == "__main__":
    main()
