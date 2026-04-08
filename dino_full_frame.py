#!/usr/bin/env python3
"""
Run Grounding DINO on full camera frames (no radar crop guidance).
Comparison baseline to show radar-guided cropping detects more objects.

Produces:
  - vid5_dino_full/   Full frames with DINO detections overlaid
  - 05_dino_full_frame.mp4
  - dino_full_frame_detections.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINO on full frames (baseline comparison)")
    p.add_argument("--frames-jsonl",
                    default="/Volumes/WorkDrive/10_10Data/all_projections_fast/compute/frames.jsonl")
    p.add_argument("--rgb-dir",
                    default="/Volumes/WorkDrive/10_10Data/lucid_20251010_125220/cam1/rgb_out")
    p.add_argument("--out-dir",
                    default="/Volumes/WorkDrive/10_10Data/dream_multi_fusion_output")
    p.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--box-thresh", type=float, default=0.25)
    p.add_argument("--text-thresh", type=float, default=0.25)
    p.add_argument("--dino-prompt", default="boat . buoy . vessel . navigation buoy .")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
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


DINO_COLORS = {
    "boat": (115, 0, 10),
    "buoy": (70, 106, 159),
}
DINO_DEFAULT = (206, 211, 24)


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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    vid5_dir = out_dir / "vid5_dino_full"
    video_dir = out_dir / "videos"

    for d in [vid5_dir, video_dir]:
        if d.exists():
            for f in d.glob("[0-9]*.*"):
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    # Get camera frames from frames.jsonl
    frames = []
    with open(args.frames_jsonl, "r") as f:
        for line in f:
            frames.append(json.loads(line))
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    print(f"Loaded {len(frames)} frames")

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading Grounding DINO on {device}...")
    processor, model = load_dino(args.model_id, device)
    print("Model loaded.")

    font = get_font(28)
    rgb_dir = Path(args.rgb_dir)

    det_path = out_dir / "dino_full_frame_detections.jsonl"
    det_file = det_path.open("w", encoding="utf-8")

    total_dets = 0

    for idx, rec in enumerate(frames):
        cf = int(rec["camera_frame"])
        rf = int(rec["radar_frame"])
        tag = f"[{idx + 1}/{len(frames)}] RF{rf:04d} CF{cf}"

        img_path = rgb_dir / f"{cf}_rgb.png"
        if not img_path.exists():
            print(f"{tag}: missing, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # Run DINO on full frame
        dets = run_dino(processor, model, img, args.dino_prompt,
                        args.box_thresh, args.text_thresh, device)

        # Draw detections
        draw = ImageDraw.Draw(img)
        for d in dets:
            color = DINO_COLORS.get(d["label"], DINO_DEFAULT)
            x1, y1, x2, y2 = d["bbox_xyxy"]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            draw.text((int(x1) + 2, max(0, int(y1) - 32)),
                      f"{d['label']} {d['score']:.2f}", fill=color, font=font)

        # Count label
        n_boats = sum(1 for d in dets if d["label"] == "boat")
        n_buoys = sum(1 for d in dets if d["label"] == "buoy")

        # Frame info overlay
        draw.text((24, 24),
                  f"DINO full-frame | boats:{n_boats} buoys:{n_buoys}",
                  fill=(255, 255, 255), font=font)

        resized = img.resize((1920, int(1920 * img_h / img_w)), Image.LANCZOS)
        resized.save(vid5_dir / f"{idx:06d}.jpg", quality=95)

        det_file.write(json.dumps({
            "frame_idx": idx,
            "camera_frame": cf,
            "radar_frame": rf,
            "num_detections": len(dets),
            "num_boats": n_boats,
            "num_buoys": n_buoys,
            "detections": dets,
        }) + "\n")
        det_file.flush()

        total_dets += len(dets)
        print(f"{tag}: {len(dets)} dets (boats:{n_boats} buoys:{n_buoys})")

        del img, resized
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    det_file.close()
    print(f"\nTotal detections across {len(frames)} frames: {total_dets}")
    print(f"Avg per frame: {total_dets / max(1, len(frames)):.1f}")

    if args.skip_video:
        print("Skipping video encoding.")
        return

    pattern = str(vid5_dir / "%06d.jpg")
    dst = str(video_dir / "05_dino_full_frame.mp4")
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
    print("Encoding 05_dino_full_frame.mp4...")
    subprocess.run(cmd, check=True)
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()
