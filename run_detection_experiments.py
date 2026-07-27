#!/usr/bin/env python3
"""Run corrected vision-only and radar-confidence-gated DINO experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_PROMPT = (
    "boat . vessel . ship . buoy . navigation buoy . channel marker . red buoy . green buoy ."
)
BOAT_SUBSTRINGS = ("boat", "vess", "ship", "watercraft", "barge", "yacht", "dinghy")
BUOY_SUBSTRINGS = ("buoy", "navig", "marker", "channel", "beacon", "daymark", "daybeacon")
BUOY_COLOR_TOKENS = frozenset({"red", "green", "orange", "yellow", "black"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-jsonl", type=Path, default=Path("data/frames.jsonl"))
    parser.add_argument("--rgb-dir", type=Path, default=Path("data/rgb_out"))
    parser.add_argument(
        "--radar-status-jsonl",
        type=Path,
        default=Path("out/detections.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--candidate-threshold", type=float, default=0.10)
    parser.add_argument("--vision-threshold", type=float, default=0.18)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    parser.add_argument("--tile-width", type=int, default=2048)
    parser.add_argument("--tile-height", type=int, default=1600)
    parser.add_argument("--tile-overlap", type=int, default=300)
    parser.add_argument("--short-edge", type=int, default=1024)
    parser.add_argument("--long-edge", type=int, default=1600)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--radar-padding", type=float, default=150)
    parser.add_argument("--radar-min-ioa", type=float, default=0.25)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--gpu-id", type=int, default=-1)
    return parser.parse_args()


def normalize_label(label: str) -> str:
    clean = str(label).strip().lower()
    if not clean:
        return "other"
    if any(value in clean for value in BOAT_SUBSTRINGS):
        return "boat"
    if any(value in clean for value in BUOY_SUBSTRINGS):
        return "buoy"
    tokens = set(clean.replace(".", " ").split())
    if tokens & BUOY_COLOR_TOKENS:
        return "buoy"
    if clean == "bu" or clean.endswith(" bu"):
        return "buoy"
    return clean


def axis_positions(length: int, tile_length: int, overlap: int) -> list[int]:
    if tile_length >= length:
        return [0]
    stride = tile_length - overlap
    if stride <= 0:
        raise ValueError("tile overlap must be smaller than both tile dimensions")
    steps = math.ceil((length - tile_length) / stride)
    return [round(index * (length - tile_length) / steps) for index in range(steps + 1)]


def tile_boxes(
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
) -> list[list[int]]:
    xs = axis_positions(width, tile_width, overlap)
    ys = axis_positions(height, tile_height, overlap)
    return [
        [x, y, min(width, x + tile_width), min(height, y + tile_height)] for y in ys for x in xs
    ]


def box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def box_ioa(box: list[float], region: list[float]) -> float:
    x1 = max(box[0], region[0])
    y1 = max(box[1], region[1])
    x2 = min(box[2], region[2])
    y2 = min(box[3], region[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return intersection / area if area else 0.0


def class_aware_nms(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept = []
    labels = sorted({str(detection["label"]) for detection in detections})
    for label in labels:
        candidates = sorted(
            (detection for detection in detections if detection["label"] == label),
            key=lambda detection: float(detection["score"]),
            reverse=True,
        )
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [
                candidate
                for candidate in candidates
                if box_iou(best["bbox_xyxy"], candidate["bbox_xyxy"]) < threshold
            ]
    return sorted(kept, key=lambda detection: float(detection["score"]), reverse=True)


def radar_supported(
    box: list[float],
    radar_regions: list[list[float]],
    minimum_ioa: float,
) -> bool:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    for region in radar_regions:
        center_inside = region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]
        if center_inside or box_ioa(box, region) >= minimum_ioa:
            return True
    return False


def load_radar_regions(path: Path, padding: float) -> dict[int, list[list[float]]]:
    regions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        frame_regions = []
        for radar in record.get("radar_status", []):
            if radar.get("status") != "keep":
                continue
            x1, y1, x2, y2 = (float(value) for value in radar["bbox_xyxy"])
            frame_regions.append([x1 - padding, y1 - padding, x2 + padding, y2 + padding])
        regions[int(record["camera_frame"])] = frame_regions
    return regions


class GroundingDino:
    def __init__(
        self,
        model_id: str,
        device: str,
        short_edge: int,
        long_edge: int,
    ) -> None:
        import torch  # ty: ignore[unresolved-import]
        from transformers import (  # ty: ignore[unresolved-import]
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self.torch = torch
        self.device = device
        self.short_edge = short_edge
        self.long_edge = long_edge
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(device).eval()

    def detect(
        self,
        image: Image.Image,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        size = {"shortest_edge": self.short_edge, "longest_edge": self.long_edge}
        try:
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                size=size,
            )
        except TypeError:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([image.size[::-1]], device=self.device)
        try:
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        labels = result.get("text_labels") or result.get("labels", [])
        return [
            {
                "label": normalize_label(str(label)),
                "raw_label": str(label).strip(),
                "score": float(score),
                "bbox_xyxy": [float(value) for value in box.tolist()],
            }
            for label, box, score in zip(
                labels,
                result.get("boxes", []),
                result.get("scores", []),
                strict=False,
            )
        ]


def tiled_candidates(
    detector: GroundingDino,
    image: Image.Image,
    prompt: str,
    threshold: float,
    text_threshold: float,
    boxes: list[list[int]],
    nms_iou: float,
) -> list[dict[str, Any]]:
    candidates = []
    for tile_index, (x1, y1, x2, y2) in enumerate(boxes):
        tile = image.crop((x1, y1, x2, y2))
        for detection in detector.detect(tile, prompt, threshold, text_threshold):
            if detection["label"] not in {"boat", "buoy"}:
                continue
            box = detection["bbox_xyxy"]
            candidates.append(
                {
                    **detection,
                    "bbox_xyxy": [
                        box[0] + x1,
                        box[1] + y1,
                        box[2] + x1,
                        box[3] + y1,
                    ],
                    "tile_index": tile_index,
                }
            )
    return class_aware_nms(candidates, nms_iou)


def main() -> None:
    args = parse_args()
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch  # ty: ignore[unresolved-import]

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this experiment")
    device = "cuda"

    frames = [
        json.loads(line) for line in args.frames_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    end_index = len(frames) if args.end_index < 0 else min(args.end_index, len(frames))
    selected = frames[args.start_index : end_index]
    radar_by_frame = load_radar_regions(args.radar_status_jsonl, args.radar_padding)
    detector = GroundingDino(
        args.model_id,
        device,
        args.short_edge,
        args.long_edge,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output:
        for local_index, frame_record in enumerate(selected, start=1):
            camera_frame = int(frame_record["camera_frame"])
            with Image.open(args.rgb_dir / f"{camera_frame}_rgb.png") as source:
                image = source.convert("RGB")
            boxes = tile_boxes(
                image.width,
                image.height,
                args.tile_width,
                args.tile_height,
                args.tile_overlap,
            )
            candidates = tiled_candidates(
                detector,
                image,
                args.prompt,
                args.candidate_threshold,
                args.text_threshold,
                boxes,
                args.nms_iou,
            )
            vision = [
                detection for detection in candidates if detection["score"] >= args.vision_threshold
            ]
            radar_regions = radar_by_frame.get(camera_frame, [])
            radar_gated = [
                {
                    **detection,
                    "radar_supported": radar_supported(
                        detection["bbox_xyxy"],
                        radar_regions,
                        args.radar_min_ioa,
                    ),
                }
                for detection in candidates
                if detection["score"] >= args.vision_threshold
                or radar_supported(
                    detection["bbox_xyxy"],
                    radar_regions,
                    args.radar_min_ioa,
                )
            ]
            output.write(
                json.dumps(
                    {
                        "frame_index": args.start_index + local_index - 1,
                        "camera_frame": camera_frame,
                        "tile_count": len(boxes),
                        "candidate_count": len(candidates),
                        "vision_detections": vision,
                        "radar_gated_detections": radar_gated,
                        "radar_regions": radar_regions,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            output.flush()
            print(
                f"[{local_index}/{len(selected)}] CF {camera_frame}: "
                f"{len(candidates)} candidates, {len(vision)} vision, "
                f"{len(radar_gated)} gated",
                flush=True,
            )


if __name__ == "__main__":
    main()
