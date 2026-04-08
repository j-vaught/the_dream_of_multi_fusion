#!/usr/bin/env python3
"""
Benchmark: DINO on full frames vs radar-guided crops.
Runs 100 frames each way and reports total + per-frame timing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def load_dino(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def run_dino(processor, model, image, prompt, box_thresh, text_thresh, device):
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)
    try:
        res = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            box_threshold=box_thresh, text_threshold=text_thresh,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        res = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            threshold=box_thresh, text_threshold=text_thresh,
            target_sizes=target_sizes,
        )[0]
    return len(res.get("boxes", []))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--frames-jsonl", default="data/frames.jsonl")
    p.add_argument("--rgb-dir", default="data/rgb_out")
    p.add_argument("--num-frames", type=int, default=100)
    p.add_argument("--buffer-px", type=int, default=150)
    p.add_argument("--max-aspect-ratio", type=float, default=2.1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = "boat . buoy . vessel . navigation buoy ."
    thresh = 0.25

    print(f"Device: {device}")
    print(f"Loading model...")
    processor, model = load_dino("IDEA-Research/grounding-dino-base", device)
    print("Model loaded.\n")

    # Load frames
    frames = []
    with open(args.frames_jsonl) as f:
        for line in f:
            frames.append(json.loads(line))
    frames = frames[:args.num_frames]

    rgb_dir = Path(args.rgb_dir)

    # Warmup
    img0 = Image.open(rgb_dir / f"{frames[0]['camera_frame']}_rgb.png").convert("RGB")
    crop0 = img0.crop((2389, 1489, 3067, 1892))
    run_dino(processor, model, crop0, prompt, thresh, thresh, device)
    run_dino(processor, model, img0, prompt, thresh, thresh, device)
    if device == "cuda":
        torch.cuda.synchronize()
    del img0, crop0

    # --- Benchmark: full frame ---
    print(f"=== FULL FRAME ({args.num_frames} frames) ===")
    full_total_dets = 0
    t0 = time.perf_counter()
    for rec in frames:
        cf = rec["camera_frame"]
        img = Image.open(rgb_dir / f"{cf}_rgb.png").convert("RGB")
        n = run_dino(processor, model, img, prompt, thresh, thresh, device)
        full_total_dets += n
        del img
    if device == "cuda":
        torch.cuda.synchronize()
    full_time = time.perf_counter() - t0
    print(f"  Total time:    {full_time:.2f}s")
    print(f"  Per frame:     {full_time / args.num_frames * 1000:.1f}ms")
    print(f"  Total dets:    {full_total_dets}")
    print(f"  Avg dets/frame:{full_total_dets / args.num_frames:.1f}")
    print()

    # --- Benchmark: radar-guided crops ---
    print(f"=== RADAR-GUIDED CROPS ({args.num_frames} frames) ===")
    crop_total_dets = 0
    crop_total_inferences = 0
    t0 = time.perf_counter()
    for rec in frames:
        cf = rec["camera_frame"]
        img = Image.open(rgb_dir / f"{cf}_rgb.png").convert("RGB")
        img_w, img_h = img.size
        for det in rec["detections"]:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            buf = args.buffer_px
            cx1 = max(0, int(x1 - buf))
            cy1 = max(0, int(y1 - buf))
            cx2 = min(img_w, int(x2 + buf))
            cy2 = min(img_h, int(y2 + buf))
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            cw, ch = cx2 - cx1, cy2 - cy1
            if ch > 0 and (cw / ch) > args.max_aspect_ratio:
                continue
            crop = img.crop((cx1, cy1, cx2, cy2))
            n = run_dino(processor, model, crop, prompt, thresh, thresh, device)
            crop_total_dets += n
            crop_total_inferences += 1
            del crop
        del img
    if device == "cuda":
        torch.cuda.synchronize()
    crop_time = time.perf_counter() - t0
    print(f"  Total time:    {crop_time:.2f}s")
    print(f"  Per frame:     {crop_time / args.num_frames * 1000:.1f}ms")
    print(f"  Total crops:   {crop_total_inferences}")
    print(f"  Per crop:      {crop_time / max(1, crop_total_inferences) * 1000:.1f}ms")
    print(f"  Total dets:    {crop_total_dets}")
    print(f"  Avg dets/frame:{crop_total_dets / args.num_frames:.1f}")
    print()

    # --- Summary ---
    print("=== COMPARISON ===")
    print(f"  Full frame:  {full_time:.2f}s  |  {full_total_dets} detections")
    print(f"  Radar crops: {crop_time:.2f}s  |  {crop_total_dets} detections")
    speedup = full_time / crop_time if crop_time > 0 else 0
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"  Detection gain: {crop_total_dets - full_total_dets:+d} ({crop_total_dets} vs {full_total_dets})")


if __name__ == "__main__":
    main()
