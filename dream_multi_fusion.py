#!/usr/bin/env python3
"""
Multi-fusion pipeline: radar bounding boxes + SegFormer land/water filter +
Grounding DINO self-labeling with cross-crop NMS.

Produces four frame sequences (then encode to video with ffmpeg):
  1. vid1_raw/         – raw camera input frames
  2. vid2_radar/       – camera frames with water-mask tint + KEEP/SKIP radar bboxes
  3. vid3_dino/        – per-frame composite of DINO-labeled crops (KEEP only)
  4. vid4_dino_remap/  – full frame with DINO detections remapped + class-aware NMS
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms as _nms_op
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from land_segmentation import compute_water_mask, load_segformer, water_fraction


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-fusion: radar bbox + land/water + DINO pipeline")
    p.add_argument("--frames-jsonl", default="data/frames.jsonl")
    p.add_argument("--points-dir",   default="data/points")
    p.add_argument("--rgb-dir",      default="data/rgb_out")
    p.add_argument("--out-dir",      default="out")

    p.add_argument("--buffer-px", type=int, default=150,
                   help="Padding in pixels around each radar bbox for the DINO crop")
    p.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--box-thresh", type=float, default=0.18,
                   help="Lower = higher recall on small buoys (0.25 misses "
                        "most sub-40px targets in our scenes)")
    p.add_argument("--text-thresh", type=float, default=0.18)
    p.add_argument("--dino-prompt",
                   default=("boat . vessel . ship . buoy . navigation buoy . "
                            "channel marker . red buoy . green buoy ."))
    p.add_argument("--dino-short-edge", type=int, default=1024,
                   help="Shortest-edge resize target for DINO (default 800 "
                        "collapses sub-40px targets; 1024 keeps them visible)")
    p.add_argument("--dino-long-edge", type=int, default=1600,
                   help="Longest-edge cap for DINO resize")
    p.add_argument("--max-aspect-ratio", type=float, default=2.1,
                   help="Skip padded crops with width/height above this")

    # Land / water filter
    p.add_argument("--seg-model",
                   default="nvidia/segformer-b2-finetuned-ade-512-512")
    p.add_argument("--water-thresh", type=float, default=0.90,
                   help="Radar bbox is KEEP if water fraction >= this")

    # Crop merging (before DINO) + cross-crop dedup (after DINO)
    p.add_argument("--merge-iou", type=float, default=0.3,
                   help="Merge KEEP padded crops when intersection/smaller-area "
                        "exceeds this (0 disables merging; 1 requires full containment)")
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="IoU threshold for class-aware NMS on remapped detections")

    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--start-idx", type=int, default=0,
                   help="Inclusive frame index to start at (for sharded runs)")
    p.add_argument("--end-idx", type=int, default=-1,
                   help="Exclusive frame index to stop at; -1 = end")
    p.add_argument("--gpu-id", type=int, default=-1,
                   help="CUDA device index (-1 = autodetect). Sets "
                        "CUDA_VISIBLE_DEVICES so each shard only sees its GPU.")
    p.add_argument("--no-clean", action="store_true",
                   help="Don't wipe existing frames in output dirs (for shards)")
    p.add_argument("--det-jsonl", default="",
                   help="Override detections.jsonl filename (for sharded runs)")
    p.add_argument("--encode-only", action="store_true",
                   help="Skip inference; just run ffmpeg on whatever frames "
                        "are already on disk in vid{1..4} dirs.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--skip-video", action="store_true")
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


BOAT_SUBSTRINGS = ("boat", "vess", "ship", "watercraft", "barge", "yacht", "dinghy")
BUOY_SUBSTRINGS = ("buoy", "navig", "marker", "channel", "beacon", "daymark", "daybeacon")
# Color words only appear in our prompt as part of buoy descriptions (red buoy,
# green buoy); if DINO returns just the color token, it still grounded the buoy.
BUOY_COLOR_TOKENS = frozenset({"red", "green", "orange", "yellow", "black"})


def normalize_label(lab: str) -> str:
    """Collapse DINO phrases (including truncated ones) to {boat, buoy, other}."""
    clean = str(lab).strip().lower()
    if not clean:
        return "other"
    if any(s in clean for s in BOAT_SUBSTRINGS):
        return "boat"
    if any(s in clean for s in BUOY_SUBSTRINGS):
        return "buoy"
    tokens = set(clean.replace(".", " ").split())
    if tokens & BUOY_COLOR_TOKENS:
        return "buoy"
    # Tail / standalone truncations like "navigation bu" or lone "bu"
    if clean == "bu" or clean.endswith(" bu"):
        return "buoy"
    return clean


def run_dino(processor, model, image: Image.Image, prompt: str,
             box_thresh: float, text_thresh: float, device: str,
             short_edge: int = 1024, long_edge: int = 1600) -> List[Dict]:
    size_arg = {"shortest_edge": int(short_edge), "longest_edge": int(long_edge)}
    try:
        inputs = processor(images=image, text=prompt, return_tensors="pt",
                           size=size_arg)
    except TypeError:
        # Older processor: the `size` kwarg path differs; fall back to defaults.
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
        dets.append({
            "label": normalize_label(lab),
            "raw_label": str(lab).strip(),
            "score": float(score),
            "bbox_xyxy": [float(v) for v in box.tolist()],
        })
    return dets


# ---------------------------------------------------------------------------
# Colors & fonts
# ---------------------------------------------------------------------------

# Brand-aligned palette
ATLANTIC     = (70, 106, 159)
GARNET       = (115, 0, 10)
ROSE         = (204, 46, 64)
GRASS        = (206, 211, 24)
SKIP_GREY    = (162, 162, 162)   # 50% black

STATUS_COLORS = {
    "keep":       ATLANTIC,
    "skip_land":  SKIP_GREY,
    "skip_shape": SKIP_GREY,
    "skip_empty": SKIP_GREY,
}

DINO_COLORS = {
    "boat": GARNET,
    "buoy": ATLANTIC,
}
DINO_DEFAULT_COLOR = GRASS

WATER_TINT = ATLANTIC


def get_font(size: int = 28):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def clamp_box(x1, y1, x2, y2, w, h):
    return (max(0, int(x1)), max(0, int(y1)),
            min(w, int(x2)), min(h, int(y2)))


def ioa_smaller(a: Tuple[float, float, float, float],
                b: Tuple[float, float, float, float]) -> float:
    """Intersection area / smaller bbox area. Robust to nesting."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    m = min(area_a, area_b)
    return inter / m if m > 0 else 0.0


def union_bbox(bboxes: List[List[float]]) -> List[int]:
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [int(x1), int(y1), int(x2), int(y2)]


def merge_keep_crops(keep_dets: List[Dict], merge_iou: float) -> List[List[Dict]]:
    """Union-find grouping of KEEP dets whose padded crops overlap.

    Two dets are merged when ioa_smaller(padded_a, padded_b) > merge_iou.
    Returns a list of groups (each a list of dets). With merge_iou = 0 the
    original list is returned untouched (one group per det)."""
    n = len(keep_dets)
    if n <= 1 or merge_iou <= 0:
        return [[d] for d in keep_dets]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    boxes = [d["padded_bbox"] for d in keep_dets]
    for i in range(n):
        for j in range(i + 1, n):
            if ioa_smaller(boxes[i], boxes[j]) > merge_iou:
                union(i, j)

    groups: Dict[int, List[Dict]] = {}
    for i, d in enumerate(keep_dets):
        groups.setdefault(find(i), []).append(d)
    # Sort groups by top-left y then x for stable composite ordering
    ordered = list(groups.values())
    ordered.sort(key=lambda g: (min(d["padded_bbox"][1] for d in g),
                                min(d["padded_bbox"][0] for d in g)))
    return ordered


# ---------------------------------------------------------------------------
# Classification: decide KEEP vs SKIP for each radar detection
# ---------------------------------------------------------------------------

def classify_detection(det: Dict, water_mask: np.ndarray,
                       img_w: int, img_h: int,
                       buffer_px: int, water_thresh: float,
                       max_aspect_ratio: float) -> Dict:
    """Annotate det with water_frac + status, and (for KEEP) padded_bbox."""
    x1, y1, x2, y2 = det["bbox_xyxy"]
    bx1, by1, bx2, by2 = clamp_box(x1, y1, x2, y2, img_w, img_h)
    det["bbox_clamped"] = [bx1, by1, bx2, by2]

    if bx2 <= bx1 or by2 <= by1:
        det["water_frac"] = 0.0
        det["status"] = "skip_empty"
        return det

    det["water_frac"] = water_fraction(water_mask, (bx1, by1, bx2, by2))

    if det["water_frac"] < water_thresh:
        det["status"] = "skip_land"
        return det

    cx1, cy1, cx2, cy2 = clamp_box(
        x1 - buffer_px, y1 - buffer_px,
        x2 + buffer_px, y2 + buffer_px,
        img_w, img_h)
    if cx2 <= cx1 or cy2 <= cy1:
        det["status"] = "skip_empty"
        return det
    crop_w, crop_h = cx2 - cx1, cy2 - cy1
    if crop_h > 0 and (crop_w / crop_h) > max_aspect_ratio:
        det["status"] = "skip_shape"
        return det

    det["status"] = "keep"
    det["padded_bbox"] = [cx1, cy1, cx2, cy2]
    return det


# ---------------------------------------------------------------------------
# Cross-crop NMS (class-aware)
# ---------------------------------------------------------------------------

def class_aware_nms(dets: List[Dict], iou_thresh: float) -> List[Dict]:
    if not dets:
        return []
    kept: List[Dict] = []
    by_label: Dict[str, List[Dict]] = {}
    for d in dets:
        by_label.setdefault(d["label"], []).append(d)
    for _, lst in by_label.items():
        boxes  = torch.tensor([d["bbox_xyxy"] for d in lst], dtype=torch.float32)
        scores = torch.tensor([d["score"] for d in lst],      dtype=torch.float32)
        keep_idx = _nms_op(boxes, scores, iou_thresh)
        for i in keep_idx.tolist():
            kept.append(lst[i])
    kept.sort(key=lambda d: d["score"], reverse=True)
    return kept


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def apply_water_tint(img: Image.Image, water_mask: np.ndarray,
                     color: Tuple[int, int, int], alpha: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    c = np.array(color, dtype=np.float32)
    arr[water_mask] = arr[water_mask] * (1.0 - alpha) + c * alpha
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def draw_radar_status_overlay(img: Image.Image, classified_dets: List[Dict],
                              points_data, water_mask: np.ndarray,
                              font, water_thresh: float) -> Image.Image:
    """Water tint + radar returns + KEEP/SKIP color-coded bboxes."""
    out = apply_water_tint(img, water_mask, WATER_TINT, alpha=0.22)
    draw = ImageDraw.Draw(out)

    u = points_data["u"]
    v = points_data["v"]
    inten = points_data["intensity"]

    # Radar return points (color = intensity)
    for det in classified_dets:
        s, e = int(det["point_start"]), int(det["point_end"])
        uu, vv, ii = u[s:e], v[s:e], inten[s:e]
        for px, py, pi in zip(uu, vv, ii):
            t = float(np.clip(float(pi) / 17.0, 0.0, 1.0))
            c = int(np.clip(round(255.0 * t), 0, 255))
            r = 2.0
            draw.ellipse((float(px) - r, float(py) - r,
                          float(px) + r, float(py) + r),
                         fill=(255, c, 0))

    # Color-coded bboxes + X-out on skips
    for det in classified_dets:
        status = det.get("status", "keep")
        color = STATUS_COLORS.get(status, SKIP_GREY)
        frac = det.get("water_frac", 0.0)
        x1, y1, x2, y2 = det.get("bbox_clamped", det["bbox_xyxy"])
        if status == "keep":
            lw = 5
            tag = f"T{det['track_id']} KEEP  water {frac:.0%}"
            draw.rectangle((x1, y1, x2, y2), outline=color, width=lw)
        else:
            lw = 3
            reason = status.replace("skip_", "").upper()
            tag = f"T{det['track_id']} SKIP:{reason}  water {frac:.0%}"
            draw.rectangle((x1, y1, x2, y2), outline=color, width=lw)
            draw.line((x1, y1, x2, y2), fill=color, width=lw)
            draw.line((x1, y2, x2, y1), fill=color, width=lw)
        draw.text((int(x1), max(0, int(y1) - 38)),
                  tag, fill=color, font=font)

    # Frame legend (top-left)
    kept = sum(1 for d in classified_dets if d.get("status") == "keep")
    total = len(classified_dets)
    draw.text((24, 18),
              f"radar tracks: {kept} KEEP / {total - kept} SKIP   "
              f"(water thr {water_thresh:.0%})",
              fill=(255, 255, 255), font=font)
    return out


def draw_dino_on_crop(crop: Image.Image, dets: List[Dict], font) -> Image.Image:
    out = crop.copy()
    draw = ImageDraw.Draw(out)
    for d in dets:
        color = DINO_COLORS.get(d["label"], DINO_DEFAULT_COLOR)
        x1, y1, x2, y2 = d["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f"{d['label']} {d['score']:.2f}"
        draw.text((int(x1) + 2, max(0, int(y1) - 24)),
                  text, fill=color, font=font)
    return out


def draw_dino_on_full_frame(img: Image.Image, deduped: List[Dict],
                            keep_crop_bboxes: List[List[int]],
                            font) -> Image.Image:
    """Full frame: thin atlantic outlines for KEEP crops + DINO dets on top."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for bb in keep_crop_bboxes:
        x1, y1, x2, y2 = bb
        draw.rectangle((x1, y1, x2, y2), outline=ATLANTIC, width=2)
    for d in deduped:
        color = DINO_COLORS.get(d["label"], DINO_DEFAULT_COLOR)
        x1, y1, x2, y2 = d["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        text = f"{d['label']} {d['score']:.2f}"
        draw.text((int(x1) + 2, max(0, int(y1) - 34)),
                  text, fill=color, font=font)
    n_boats = sum(1 for d in deduped if d["label"] == "boat")
    n_buoys = sum(1 for d in deduped if d["label"] == "buoy")
    draw.text((24, 18),
              f"DINO remapped + NMS: boats {n_boats}, buoys {n_buoys}, "
              f"total {len(deduped)}",
              fill=(255, 255, 255), font=font)
    return out


def make_crop_composite(crops: List[Image.Image], labels: List[str],
                        target_w: int, font) -> Image.Image:
    if not crops:
        placeholder = Image.new("RGB", (target_w, 360), (0, 0, 0))
        d = ImageDraw.Draw(placeholder)
        d.text((20, 170), "No KEEP radar detections", fill=(180, 180, 180),
               font=font)
        return placeholder

    resized = []
    for crop, lab in zip(crops, labels):
        cw, ch = crop.size
        if cw == 0:
            continue
        scale = target_w / cw
        new_h = max(1, int(ch * scale))
        r = crop.resize((target_w, new_h), Image.LANCZOS)
        d = ImageDraw.Draw(r)
        d.rectangle((0, 0, target_w, 28), fill=(0, 0, 0))
        d.text((4, 2), lab, fill=ATLANTIC, font=font)
        resized.append(r)

    if not resized:
        return Image.new("RGB", (target_w, 360), (0, 0, 0))
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

    # Pin this process to a specific GPU *before* any torch CUDA init happens.
    if args.gpu_id >= 0:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    out_dir = Path(args.out_dir)
    vid1_dir = out_dir / "vid1_raw"
    vid2_dir = out_dir / "vid2_radar"
    vid3_dir = out_dir / "vid3_dino"
    vid4_dir = out_dir / "vid4_dino_remap"
    video_dir = out_dir / "videos"
    crops_dir = out_dir / "crops"
    det_name = args.det_jsonl or "detections.jsonl"
    detections_jsonl = out_dir / det_name

    for d in [vid1_dir, vid2_dir, vid3_dir, vid4_dir, video_dir, crops_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.encode_only:
        # encode-only must never touch frame files on disk
        run_encoding(out_dir, video_dir, args.fps)
        return

    if not args.no_clean:
        for d in [vid1_dir, vid2_dir, vid3_dir, vid4_dir, crops_dir]:
            for f in d.glob("[0-9]*.*"):
                f.unlink()

    frames_all = []
    with open(args.frames_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            frames_all.append(json.loads(line))

    # Apply slicing: [start:end] (global indices preserved for filenames)
    end = args.end_idx if args.end_idx >= 0 else len(frames_all)
    frame_slice = list(range(args.start_idx, min(end, len(frames_all))))
    if args.max_frames > 0:
        frame_slice = frame_slice[:args.max_frames]
    frames = [(gi, frames_all[gi]) for gi in frame_slice]
    print(f"Loaded {len(frames_all)} total frames; this shard handles "
          f"{len(frames)} (global idx {frame_slice[0] if frames else '-'} .. "
          f"{frame_slice[-1] if frames else '-'})")

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading Grounding DINO on {device}...")
    dino_proc, dino_model = load_dino(args.model_id, device)
    print("Loading SegFormer for water segmentation...")
    seg_proc, seg_model, water_ids = load_segformer(args.seg_model, device)
    print(f"Models loaded on {device}.")

    font_lg = get_font(28)
    font_sm = get_font(20)

    rgb_dir = Path(args.rgb_dir)
    points_dir = Path(args.points_dir)
    det_file = detections_jsonl.open("w", encoding="utf-8")

    for i, (idx, rec) in enumerate(frames):
        cf = int(rec["camera_frame"])
        rf = int(rec["radar_frame"])
        tag = f"[{i + 1}/{len(frames)} global {idx:04d}] RF{rf:04d} CF{cf}"

        img_path = rgb_dir / f"{cf}_rgb.png"
        if not img_path.exists():
            print(f"{tag}: image missing, skipping")
            continue
        npz_path = points_dir / rec["points_file"]
        if not npz_path.exists():
            print(f"{tag}: points missing, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        points_data = np.load(npz_path)
        radar_dets = rec["detections"]

        # ---- 1. Segmentation + classification ----
        water_mask = compute_water_mask(seg_proc, seg_model, img,
                                        water_ids, device)
        classified = [
            classify_detection(dict(det), water_mask, img_w, img_h,
                               args.buffer_px, args.water_thresh,
                               args.max_aspect_ratio)
            for det in radar_dets
        ]

        # ---- vid1: raw ----
        raw_resized = img.resize(
            (1920, int(1920 * img_h / img_w)), Image.LANCZOS)
        raw_resized.save(vid1_dir / f"{idx:06d}.jpg", quality=92)

        # ---- vid2: radar status overlay ----
        radar_img = draw_radar_status_overlay(img, classified, points_data,
                                              water_mask, font_lg,
                                              args.water_thresh)
        radar_resized = radar_img.resize(
            (1920, int(1920 * img_h / img_w)), Image.LANCZOS)
        radar_resized.save(vid2_dir / f"{idx:06d}.jpg", quality=92)

        # ---- Merge overlapping KEEP crops so each water region is scanned once ----
        keep_dets = [c for c in classified if c.get("status") == "keep"]
        groups = merge_keep_crops(keep_dets, args.merge_iou)

        crop_images_composite: List[Image.Image] = []
        crop_labels: List[str] = []
        remapped_all: List[Dict] = []
        keep_crop_bboxes: List[List[int]] = []
        per_crop_records: List[Dict] = []

        for group in groups:
            gx1, gy1, gx2, gy2 = union_bbox([d["padded_bbox"] for d in group])
            keep_crop_bboxes.append([gx1, gy1, gx2, gy2])

            crop = img.crop((gx1, gy1, gx2, gy2))
            sorted_group = sorted(group, key=lambda d: int(d["track_id"]))
            track_tag = "+".join(f"T{int(d['track_id'])}" for d in sorted_group)
            source_tracks = [int(d["track_id"]) for d in sorted_group]
            crop_name = f"{idx:06d}_{track_tag}.jpg"
            crop.save(crops_dir / crop_name, quality=92)

            dino_dets = run_dino(dino_proc, dino_model, crop,
                                 args.dino_prompt,
                                 args.box_thresh, args.text_thresh, device,
                                 short_edge=args.dino_short_edge,
                                 long_edge=args.dino_long_edge)

            # Remap crop-local bboxes to full-frame coords
            remapped_group: List[Dict] = []
            for det in dino_dets:
                x1, y1, x2, y2 = det["bbox_xyxy"]
                remapped_group.append({
                    "label": det["label"],
                    "raw_label": det.get("raw_label", ""),
                    "score": det["score"],
                    "bbox_xyxy": [x1 + gx1, y1 + gy1, x2 + gx1, y2 + gy1],
                    "source_tracks": list(source_tracks),
                })
            remapped_all.extend(remapped_group)

            labeled = draw_dino_on_crop(crop, dino_dets, font_sm)
            crop_images_composite.append(labeled)
            n_boats = sum(1 for d in dino_dets if d["label"] == "boat")
            n_buoys = sum(1 for d in dino_dets if d["label"] == "buoy")
            min_water = min(d["water_frac"] for d in group)
            crop_labels.append(
                f"{track_tag}  water>={min_water:.0%}  "
                f"boats:{n_boats} buoys:{n_buoys}")

            per_crop_records.append({
                "source_tracks": source_tracks,
                "radar_bboxes_xyxy": [d["bbox_xyxy"] for d in sorted_group],
                "padded_bboxes_xyxy": [d["padded_bbox"] for d in sorted_group],
                "merged_crop_xyxy": [gx1, gy1, gx2, gy2],
                "water_fracs": [float(d["water_frac"]) for d in sorted_group],
                "crop_file": crop_name,
                "dino_detections_crop": dino_dets,
                "dino_detections_remapped": remapped_group,
            })

        # ---- Cross-crop class-aware NMS ----
        deduped = class_aware_nms(remapped_all, args.nms_iou)

        # ---- vid3: composite of KEEP crops with DINO labels ----
        composite = make_crop_composite(crop_images_composite, crop_labels,
                                        args.crop_width, font_sm)
        composite.save(vid3_dir / f"{idx:06d}.jpg", quality=92)

        # ---- vid4: full frame with remapped + deduped DINO boxes ----
        remap_img = draw_dino_on_full_frame(img, deduped,
                                            keep_crop_bboxes, font_lg)
        remap_resized = remap_img.resize(
            (1920, int(1920 * img_h / img_w)), Image.LANCZOS)
        remap_resized.save(vid4_dir / f"{idx:06d}.jpg", quality=92)

        # ---- per-frame detection record ----
        det_file.write(json.dumps({
            "frame_idx": idx,
            "camera_frame": cf,
            "radar_frame": rf,
            "num_radar_dets": len(classified),
            "num_keep": sum(1 for c in classified if c["status"] == "keep"),
            "num_skip_land": sum(1 for c in classified if c["status"] == "skip_land"),
            "num_skip_shape": sum(1 for c in classified if c["status"] == "skip_shape"),
            "num_skip_empty": sum(1 for c in classified if c["status"] == "skip_empty"),
            "num_merged_groups": len(groups),
            "radar_status": [
                {"track_id": int(c["track_id"]),
                 "status": c["status"],
                 "water_frac": float(c.get("water_frac", 0.0)),
                 "bbox_xyxy": c["bbox_xyxy"]}
                for c in classified
            ],
            "crops": per_crop_records,
            "num_remapped_raw": len(remapped_all),
            "num_remapped_deduped": len(deduped),
            "remapped_deduped": deduped,
        }) + "\n")
        det_file.flush()

        kept = sum(1 for c in classified if c["status"] == "keep")
        print(f"{tag}: {len(classified)} radar dets "
              f"({kept} KEEP in {len(groups)} groups / "
              f"{len(classified) - kept} SKIP), "
              f"{len(remapped_all)} raw DINO -> {len(deduped)} after NMS")

        # free per-frame allocations
        del img, points_data, water_mask
        del crop_images_composite, crop_labels, composite
        del raw_resized, radar_img, radar_resized, remap_img, remap_resized
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    det_file.close()

    if args.skip_video:
        print("Skipping video encoding (--skip-video).")
        return

    run_encoding(out_dir, video_dir, args.fps)
    print(f"\nDone. All outputs in {out_dir}")


def run_encoding(out_dir: Path, video_dir: Path, fps: int) -> None:
    videos = [
        ("vid1_raw",        "01_input_raw.mp4",          False),
        ("vid2_radar",      "02_radar_keep_skip.mp4",    False),
        ("vid3_dino",       "03_dino_labeled_crops.mp4", True),
        ("vid4_dino_remap", "04_dino_remap_nms.mp4",     False),
    ]
    video_dir.mkdir(parents=True, exist_ok=True)
    for subdir, filename, pad_even in videos:
        src = out_dir / subdir
        dst = video_dir / filename
        frame_files = sorted(src.glob("[0-9]*.jpg"))
        if not frame_files:
            print(f"  (no frames in {src}, skipping {filename})")
            continue
        # Build a concat list so non-contiguous frame indices still encode.
        concat_txt = src / "_concat.txt"
        with concat_txt.open("w", encoding="utf-8") as fh:
            for p in frame_files:
                fh.write(f"file '{p.resolve()}'\n")
                fh.write(f"duration {1.0 / fps}\n")
            # ffmpeg concat demuxer needs the last file repeated without duration
            fh.write(f"file '{frame_files[-1].resolve()}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-fps_mode", "cfr",
            "-r", str(fps),
        ]
        if pad_even:
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        cmd += [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            "-preset", "medium",
            str(dst),
        ]
        print(f"Encoding {filename} ({len(frame_files)} frames)...")
        subprocess.run(cmd, check=True)
        print(f"  -> {dst}")


if __name__ == "__main__":
    main()
