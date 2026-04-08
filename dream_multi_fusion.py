#!/usr/bin/env python3
"""
Multi-fusion pipeline: radar bounding boxes + Grounding DINO self-labeling.

Produces three frame sequences (then encode to video with ffmpeg):
  1. vid1_raw/       – raw camera input frames
  2. vid2_radar/     – camera frames with radar bounding boxes overlaid
  3. vid3_dino/      – per-frame composite of DINO-labeled crops (radar bbox + buffer)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-fusion: radar bbox + DINO labeling pipeline")
    p.add_argument("--frames-jsonl",
                    default="/Volumes/WorkDrive/10_10Data/all_projections_fast/compute/frames.jsonl")
    p.add_argument("--points-dir",
                    default="/Volumes/WorkDrive/10_10Data/all_projections_fast/compute/points")
    p.add_argument("--rgb-dir",
                    default="/Volumes/WorkDrive/10_10Data/lucid_20251010_125220/cam1/rgb_out")
    p.add_argument("--out-dir",
                    default="/Volumes/WorkDrive/10_10Data/dream_multi_fusion_output")
    p.add_argument("--buffer-px", type=int, default=150,
                    help="Padding in pixels around each radar bbox for the DINO crop")
    p.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--box-thresh", type=float, default=0.25)
    p.add_argument("--text-thresh", type=float, default=0.25)
    p.add_argument("--dino-prompt", default="boat . buoy . vessel . navigation buoy .")
    p.add_argument("--max-aspect-ratio", type=float, default=2.1,
                    help="Skip padded crops with width/height ratio above this (filters shoreline)")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--skip-video", action="store_true", help="Only produce frames, skip ffmpeg encode")
    p.add_argument("--crop-width", type=int, default=1920,
                    help="Fixed width for each crop tile in vid3 composite")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DINO inference
# ---------------------------------------------------------------------------

def load_dino(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def run_dino(processor, model, image: Image.Image, prompt: str,
             box_thresh: float, text_thresh: float, device: str) -> List[Dict]:
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)
    try:
        res = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            box_threshold=float(box_thresh),
            text_threshold=float(text_thresh),
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        res = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            threshold=float(box_thresh),
            text_threshold=float(text_thresh),
            target_sizes=target_sizes,
        )[0]

    labels = res.get("text_labels") or res.get("labels", [])
    dets = []
    for lab, box, score in zip(labels, res.get("boxes", []), res.get("scores", [])):
        clean = str(lab).strip().lower()
        if "boat" in clean or "vessel" in clean:
            label = "boat"
        elif "buoy" in clean or "marker" in clean:
            label = "buoy"
        else:
            label = clean
        dets.append({
            "label": label,
            "score": float(score),
            "bbox_xyxy": [float(v) for v in box.tolist()],
        })
    return dets


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

RADAR_COLORS = {
    "bbox": (0, 255, 255),       # cyan
    "centroid": (0, 255, 0),     # green
    "point_hot": (255, 160, 0),  # orange
}

DINO_COLORS = {
    "boat": (115, 0, 10),       # garnet
    "buoy": (70, 106, 159),     # atlantic
}
DINO_DEFAULT_COLOR = (206, 211, 24)  # grass


def get_font(size: int = 28):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def clamp_box(x1, y1, x2, y2, w, h):
    return (max(0, int(x1)), max(0, int(y1)),
            min(w, int(x2)), min(h, int(y2)))


def draw_radar_overlay(img: Image.Image, detections: List[Dict],
                       points_data, font) -> Image.Image:
    """Draw radar bounding boxes, centroids, and points on the image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)

    u = points_data["u"]
    v = points_data["v"]
    inten = points_data["intensity"]

    for det in detections:
        s, e = int(det["point_start"]), int(det["point_end"])
        uu, vv, ii = u[s:e], v[s:e], inten[s:e]

        for px, py, pi in zip(uu, vv, ii):
            t = float(np.clip((float(pi)) / 17.0, 0.0, 1.0))
            c = int(np.clip(round(255.0 * t), 0, 255))
            r = 2.0
            draw.ellipse((float(px) - r, float(py) - r,
                          float(px) + r, float(py) + r),
                         fill=(255, c, 0))

        x1, y1, x2, y2 = det["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=RADAR_COLORS["bbox"], width=4)

        cx, cy = det["centroid_uv"]
        cr = 9
        draw.line((cx - cr, cy - cr, cx + cr, cy + cr),
                  fill=RADAR_COLORS["centroid"], width=3)
        draw.line((cx - cr, cy + cr, cx + cr, cy - cr),
                  fill=RADAR_COLORS["centroid"], width=3)

        draw.text((int(x1), max(0, int(y1) - 34)),
                  f"T{det['track_id']}",
                  fill=RADAR_COLORS["bbox"], font=font)

    return out


def draw_dino_on_crop(crop: Image.Image, dets: List[Dict], font) -> Image.Image:
    """Draw DINO detections on a crop image."""
    out = crop.copy()
    draw = ImageDraw.Draw(out)
    for d in dets:
        label = d["label"]
        color = DINO_COLORS.get(label, DINO_DEFAULT_COLOR)
        x1, y1, x2, y2 = d["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f"{label} {d['score']:.2f}"
        draw.text((int(x1) + 2, max(0, int(y1) - 24)),
                  text, fill=color, font=font)
    return out


def make_crop_composite(crops: List[Image.Image], labels: List[str],
                        target_w: int, font) -> Image.Image:
    """Stack labeled crops vertically into a single composite image."""
    if not crops:
        placeholder = Image.new("RGB", (target_w, 360), (0, 0, 0))
        d = ImageDraw.Draw(placeholder)
        d.text((20, 170), "No radar detections", fill=(180, 180, 180), font=font)
        return placeholder

    resized = []
    for crop, lab in zip(crops, labels):
        cw, ch = crop.size
        if cw == 0:
            continue
        scale = target_w / cw
        new_h = max(1, int(ch * scale))
        r = crop.resize((target_w, new_h), Image.LANCZOS)

        # Add label banner at top
        d = ImageDraw.Draw(r)
        d.rectangle((0, 0, target_w, 28), fill=(0, 0, 0))
        d.text((4, 2), lab, fill=(0, 255, 255), font=font)
        resized.append(r)

    if not resized:
        placeholder = Image.new("RGB", (target_w, 360), (0, 0, 0))
        return placeholder

    total_h = sum(r.height for r in resized) + 4 * (len(resized) - 1)
    comp = Image.new("RGB", (target_w, total_h), (0, 0, 0))
    y_off = 0
    for r in resized:
        comp.paste(r, (0, y_off))
        y_off += r.height + 4

    return comp


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    vid1_dir = out_dir / "vid1_raw"
    vid2_dir = out_dir / "vid2_radar"
    vid3_dir = out_dir / "vid3_dino"
    video_dir = out_dir / "videos"

    for d in [vid1_dir, vid2_dir, vid3_dir, video_dir]:
        if d.exists():
            for f in d.glob("[0-9]*.png"):
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    # Load frames.jsonl
    frames = []
    with open(args.frames_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            frames.append(json.loads(line))
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    print(f"Loaded {len(frames)} frames from {args.frames_jsonl}")

    # Load DINO model
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Grounding DINO on {device}...")
    processor, model = load_dino(args.model_id, device)
    print("Model loaded.")

    font_lg = get_font(28)
    font_sm = get_font(20)

    rgb_dir = Path(args.rgb_dir)
    points_dir = Path(args.points_dir)

    for idx, rec in enumerate(frames):
        cf = int(rec["camera_frame"])
        rf = int(rec["radar_frame"])
        tag = f"[{idx + 1}/{len(frames)}] RF{rf:04d} CF{cf}"

        img_path = rgb_dir / f"{cf}_rgb.png"
        if not img_path.exists():
            print(f"{tag}: image missing, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # Load radar points
        npz_path = points_dir / rec["points_file"]
        if not npz_path.exists():
            print(f"{tag}: points missing, skipping")
            continue
        points_data = np.load(npz_path)

        detections = rec["detections"]

        # --- Video 1: raw input (downscale to 1920 wide for reasonable video size) ---
        raw_resized = img.resize((1920, int(1920 * img_h / img_w)), Image.LANCZOS)
        raw_resized.save(vid1_dir / f"{idx:06d}.png")

        # --- Video 2: radar bounding boxes ---
        radar_img = draw_radar_overlay(img, detections, points_data, font_lg)
        radar_resized = radar_img.resize((1920, int(1920 * img_h / img_w)), Image.LANCZOS)
        radar_resized.save(vid2_dir / f"{idx:06d}.png")

        # --- Video 3: DINO-labeled crops at full resolution ---
        crop_images = []
        crop_labels = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            buf = args.buffer_px
            cx1, cy1, cx2, cy2 = clamp_box(
                x1 - buf, y1 - buf, x2 + buf, y2 + buf, img_w, img_h)

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop_w = cx2 - cx1
            crop_h = cy2 - cy1
            if crop_h > 0 and (crop_w / crop_h) > args.max_aspect_ratio:
                continue

            crop = img.crop((cx1, cy1, cx2, cy2))

            # Run DINO on the full-resolution crop
            dino_dets = run_dino(processor, model, crop, args.dino_prompt,
                                 args.box_thresh, args.text_thresh, device)

            crop_labeled = draw_dino_on_crop(crop, dino_dets, font_sm)
            crop_images.append(crop_labeled)

            n_boats = sum(1 for d in dino_dets if d["label"] == "boat")
            n_buoys = sum(1 for d in dino_dets if d["label"] == "buoy")
            crop_labels.append(f"T{det['track_id']}  boats:{n_boats} buoys:{n_buoys}")

        composite = make_crop_composite(crop_images, crop_labels,
                                        args.crop_width, font_sm)
        composite.save(vid3_dir / f"{idx:06d}.png")

        print(f"{tag}: {len(detections)} radar dets, "
              f"{len(crop_images)} crops processed")

    # --- Encode videos ---
    if args.skip_video:
        print("Skipping video encoding (--skip-video).")
        return

    videos = [
        ("vid1_raw", "01_input_raw.mp4"),
        ("vid2_radar", "02_radar_bounding.mp4"),
        ("vid3_dino", "03_dino_labeled_crops.mp4"),
    ]

    for subdir, filename in videos:
        src = out_dir / subdir
        dst = video_dir / filename
        pattern = str(src / "%06d.png")
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
        print(f"Encoding {filename}...")
        # vid3 frames may have variable width; pad to even dimensions
        if subdir == "vid3_dino":
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(args.fps),
                "-i", pattern,
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-preset", "medium",
                str(dst),
            ]
        subprocess.run(cmd, check=True)
        print(f"  -> {dst}")

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
