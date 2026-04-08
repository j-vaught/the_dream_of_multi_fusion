#!/usr/bin/env python3
"""
Clean overlay: full frames with radar bounding boxes (cyan)
and re-projected DINO detections (garnet/atlantic). No radar points.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean overlay: radar bbox + DINO bbox on full frame")
    p.add_argument("--detections-jsonl",
                    default="/Volumes/WorkDrive/10_10Data/dream_multi_fusion_output/detections.jsonl")
    p.add_argument("--rgb-dir",
                    default="/Volumes/WorkDrive/10_10Data/lucid_20251010_125220/cam1/rgb_out")
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


RADAR_COLOR = (0, 255, 255)
DINO_COLORS = {
    "boat": (115, 0, 10),
    "buoy": (70, 106, 159),
}
DINO_DEFAULT = (206, 211, 24)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    vid6_dir = out_dir / "vid6_clean"
    video_dir = out_dir / "videos"

    if vid6_dir.exists():
        for f in vid6_dir.glob("[0-9]*.*"):
            f.unlink()
    vid6_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    det_records = []
    with open(args.detections_jsonl, "r") as f:
        for line in f:
            det_records.append(json.loads(line))
    print(f"Loaded {len(det_records)} frames")

    rgb_dir = Path(args.rgb_dir)
    font = get_font(28)
    font_sm = get_font(20)

    for det_rec in det_records:
        idx = det_rec["frame_idx"]
        cf = det_rec["camera_frame"]
        rf = det_rec["radar_frame"]

        img_path = rgb_dir / f"{cf}_rgb.png"
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)

        for crop_rec in det_rec["crops"]:
            track_id = crop_rec["track_id"]
            radar_bbox = crop_rec["radar_bbox_xyxy"]
            padded_bbox = crop_rec["padded_bbox_xyxy"]
            ox, oy = padded_bbox[0], padded_bbox[1]

            # Radar bbox in cyan
            draw.rectangle(radar_bbox, outline=RADAR_COLOR, width=4)
            draw.text((int(radar_bbox[0]), max(0, int(radar_bbox[1]) - 34)),
                      f"T{track_id}", fill=RADAR_COLOR, font=font)

            # DINO detections re-projected
            for dino_det in crop_rec["dino_detections"]:
                bx = dino_det["bbox_xyxy"]
                full_bbox = [bx[0] + ox, bx[1] + oy, bx[2] + ox, bx[3] + oy]
                label = dino_det["label"]
                score = dino_det["score"]
                color = DINO_COLORS.get(label, DINO_DEFAULT)

                draw.rectangle(full_bbox, outline=color, width=3)
                draw.text((int(full_bbox[0]) + 2, max(0, int(full_bbox[1]) - 24)),
                          f"{label} {score:.2f}", fill=color, font=font_sm)

        resized = img.resize((1920, int(1920 * img_h / img_w)), Image.LANCZOS)
        resized.save(vid6_dir / f"{idx:06d}.jpg", quality=95)

        print(f"[{idx + 1}/{len(det_records)}] CF{cf}")
        del img, resized

    if args.skip_video:
        return

    pattern = str(vid6_dir / "%06d.jpg")
    dst = str(video_dir / "06_clean_overlay.mp4")
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
    print("Encoding 06_clean_overlay.mp4...")
    subprocess.run(cmd, check=True)
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()
