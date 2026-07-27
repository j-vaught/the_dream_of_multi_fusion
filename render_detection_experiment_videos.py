#!/usr/bin/env python3
"""Render TP, FP, and FN overlays for the three detection experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from run_detection_experiments import box_iou, normalize_label

TP_BOAT_COLOR = "#FF00FF"
TP_BUOY_COLOR = "#FFFF00"
FP_COLOR = "#FF0000"
FN_COLOR = "#FF00FF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, default=Path("data/rgb_out"))
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("out/datasets/dream_fusion_yolo/manifest.jsonl"),
    )
    parser.add_argument(
        "--vision-only",
        type=Path,
        default=Path("out/experiments/detection_comparison/vision_only_tiled.jsonl"),
    )
    parser.add_argument(
        "--radar-gated",
        type=Path,
        default=Path("out/experiments/detection_comparison/radar_confidence_gated.jsonl"),
    )
    parser.add_argument(
        "--radar-bounded",
        type=Path,
        default=Path("out/detections.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/videos/detection_experiments"),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--line-width", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        default=None,
    )
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument(
        "--encoder",
        choices=("auto", "h264_nvenc", "libx264"),
        default="auto",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def predictions_by_frame(
    path: Path,
    detection_field: str = "detections",
) -> dict[int, list[dict[str, Any]]]:
    return {
        int(record["camera_frame"]): record.get(detection_field, []) for record in read_jsonl(path)
    }


def categorize_detections(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    eligible_predictions: list[dict[str, Any]] = [
        {
            **prediction,
            "label": normalize_label(str(prediction["label"])),
        }
        for prediction in predictions
        if normalize_label(str(prediction["label"])) in {"boat", "buoy"}
    ]
    unmatched_truth = set(range(len(ground_truth)))
    true_positives = []
    false_positives = []
    ordered_predictions = sorted(
        eligible_predictions,
        key=lambda prediction: float(prediction["score"]),
        reverse=True,
    )
    for prediction in ordered_predictions:
        label = prediction["label"]
        candidates = [
            index for index in unmatched_truth if ground_truth[index]["class_name"] == label
        ]
        best_iou, best_index = max(
            (
                (
                    box_iou(
                        prediction["bbox_xyxy"],
                        ground_truth[index]["bbox_xyxy"],
                    ),
                    index,
                )
                for index in candidates
            ),
            default=(0.0, None),
        )
        if best_index is not None and best_iou >= iou_threshold:
            true_positives.append(prediction)
            unmatched_truth.remove(best_index)
        else:
            false_positives.append(prediction)
    false_negatives = [ground_truth[index] for index in sorted(unmatched_truth)]
    return true_positives, false_positives, false_negatives


def draw_dashed_rectangle(
    drawing: ImageDraw.ImageDraw,
    box: list[float],
    color: str,
    width: int,
    dash_length: int = 24,
) -> None:
    x1, y1, x2, y2 = (round(value) for value in box)
    for start in range(x1, x2 + 1, dash_length * 2):
        drawing.line((start, y1, min(start + dash_length, x2), y1), fill=color, width=width)
        drawing.line((start, y2, min(start + dash_length, x2), y2), fill=color, width=width)
    for start in range(y1, y2 + 1, dash_length * 2):
        drawing.line((x1, start, x1, min(start + dash_length, y2)), fill=color, width=width)
        drawing.line((x2, start, x2, min(start + dash_length, y2)), fill=color, width=width)
    drawing.line((x1, y1, x2, y2), fill=color, width=max(2, width // 2))
    drawing.line((x1, y2, x2, y1), fill=color, width=max(2, width // 2))


def draw_evaluation(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
    line_width: int,
) -> None:
    true_positives, false_positives, false_negatives = categorize_detections(
        ground_truth,
        predictions,
        iou_threshold,
    )
    drawing = ImageDraw.Draw(image)
    for ground_truth_object in false_negatives:
        draw_dashed_rectangle(
            drawing,
            ground_truth_object["bbox_xyxy"],
            FN_COLOR,
            line_width,
        )
    for prediction in false_positives:
        drawing.rectangle(
            tuple(round(value) for value in prediction["bbox_xyxy"]),
            outline=FP_COLOR,
            width=line_width,
        )
    for prediction in true_positives:
        color = TP_BOAT_COLOR if prediction["label"] == "boat" else TP_BUOY_COLOR
        drawing.rectangle(
            tuple(round(value) for value in prediction["bbox_xyxy"]),
            outline=color,
            width=line_width,
        )


def choose_encoder(requested: str, width: int) -> str:
    if requested != "auto":
        return requested
    if width > 4096:
        return "libx264"
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    return "h264_nvenc" if "h264_nvenc" in result.stdout else "libx264"


def encode_video(
    frames_dir: Path,
    output: Path,
    frame_count: int,
    fps: int,
    encoder: str,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%06d.jpg"),
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        encoder,
    ]
    if encoder == "h264_nvenc":
        command.extend(["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0"])
    else:
        command.extend(["-preset", "fast", "-crf", "18"])
    command.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)])
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    ground_truth_records = read_jsonl(args.ground_truth)
    if args.max_frames:
        ground_truth_records = ground_truth_records[: args.max_frames]
    ground_truth = {
        int(record["camera_frame"]): record["objects"] for record in ground_truth_records
    }
    methods = {
        "01_vision_only_tiled": predictions_by_frame(args.vision_only),
        "02_radar_confidence_gated": predictions_by_frame(args.radar_gated),
        "03_radar_bounded_crops": predictions_by_frame(
            args.radar_bounded,
            "remapped_deduped",
        ),
    }
    camera_frames = [int(record["camera_frame"]) for record in ground_truth_records]
    if not camera_frames:
        raise ValueError("ground-truth manifest contains no frames")

    with Image.open(args.rgb_dir / f"{camera_frames[0]}_rgb.png") as first_image:
        output_width = args.crop[2] if args.crop else first_image.width
    output_width = round(output_width * args.scale_factor)
    encoder = choose_encoder(args.encoder, output_width)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="detection_video_frames_",
        dir=args.output_dir,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        frame_directories = {method_name: temporary_root / method_name for method_name in methods}
        for frame_directory in frame_directories.values():
            frame_directory.mkdir()

        for frame_index, camera_frame in enumerate(camera_frames):
            with Image.open(args.rgb_dir / f"{camera_frame}_rgb.png") as source:
                base_image = source.convert("RGB")
            for method_name, method_predictions in methods.items():
                image = base_image.copy()
                draw_evaluation(
                    image,
                    ground_truth[camera_frame],
                    method_predictions.get(camera_frame, []),
                    args.iou_threshold,
                    args.line_width,
                )
                if args.crop:
                    crop_x, crop_y, crop_width, crop_height = args.crop
                    image = image.crop(
                        (
                            crop_x,
                            crop_y,
                            crop_x + crop_width,
                            crop_y + crop_height,
                        )
                    )
                if args.scale_factor != 1.0:
                    image = image.resize(
                        (
                            round(image.width * args.scale_factor),
                            round(image.height * args.scale_factor),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                image.save(
                    frame_directories[method_name] / f"{frame_index:06d}.jpg",
                    quality=95,
                    subsampling=0,
                )
            if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(camera_frames):
                print(f"Rendered {frame_index + 1}/{len(camera_frames)} frames", flush=True)

        for method_name, frame_directory in frame_directories.items():
            output = args.output_dir / f"{method_name}.mp4"
            print(f"Encoding {output.name} with {encoder}", flush=True)
            encode_video(
                frame_directory,
                output,
                len(camera_frames),
                args.fps,
                encoder,
            )
            print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
