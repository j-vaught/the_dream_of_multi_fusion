#!/usr/bin/env python3
"""Build deterministic presentation media for the vision-only experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

FULL_VIDEO = REPOSITORY_ROOT / "08_exp1_vision_only_detections.mp4"
ZOOM_VIDEO = REPOSITORY_ROOT / "11_zoom_exp1_vision_only.mp4"
PREDICTIONS_JSONL = REPOSITORY_ROOT / "experiments/detection_comparison/vision_only_tiled.jsonl"
GROUND_TRUTH_JSONL = REPOSITORY_ROOT / "datasets/dream_fusion_yolo/manifest.jsonl"
CLEAN_FRAME_119 = REPOSITORY_ROOT / "presentation/stills/source/119_rgb.png"

CAMERA_FRAME_OFFSET = 119
SOURCE_FRAME_COUNT = 300
SOURCE_FPS = Fraction(20, 1)
OUTPUT_FPS = 60
FULL_DIMENSIONS = (5320, 3032)
ZOOM_DIMENSIONS = (1800, 1500)
ZOOM_CROP = (2550, 1300, 600, 500)
IOU_THRESHOLD = 0.5

BOAT_COLOR = "#FF00FF"
BUOY_COLOR = "#FFFF00"
FALSE_POSITIVE_COLOR = "#FF0000"
GROUND_TRUTH_COLOR = "#FFFFFF"
CLASS_COLORS = {"boat": BOAT_COLOR, "buoy": BUOY_COLOR}

CORRECT_CAMERA_FRAME = 119
EXPECTED_FALSE_POSITIVE_FRAMES = (341, 416)
FALSE_POSITIVE_WINDOWS = ((337, 345), (412, 418))
CONTEXT_REPETITIONS = 3
FALSE_POSITIVE_HOLD_FRAMES = 45
CORRECT_VIDEO_FRAMES = 300

CORRECT_POSTER = OUTPUT_DIRECTORY / "vision_only_correct_vs_ground_truth_poster.png"
CORRECT_VIDEO = OUTPUT_DIRECTORY / "vision_only_correct_vs_ground_truth.mp4"
FALSE_POSITIVE_POSTER = OUTPUT_DIRECTORY / "vision_only_false_positive_episodes_poster.png"
FALSE_POSITIVE_VIDEO = OUTPUT_DIRECTORY / "vision_only_false_positive_episodes.mp4"
MANIFEST = OUTPUT_DIRECTORY / "manifest.json"


@dataclass(frozen=True)
class Match:
    prediction: dict[str, Any]
    ground_truth: dict[str, Any]
    iou: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing outputs and their recorded hashes without rebuilding them.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rgb(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def normalize_label(label: str) -> str:
    lowered = label.strip().lower()
    if "buoy" in lowered:
        return "buoy"
    if "boat" in lowered or "ship" in lowered or "vessel" in lowered:
        return "boat"
    return lowered


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    intersection_x1 = max(box_a[0], box_b[0])
    intersection_y1 = max(box_a[1], box_b[1])
    intersection_x2 = min(box_a[2], box_b[2])
    intersection_y2 = min(box_a[3], box_b[3])
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_detections(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[list[Match], list[dict[str, Any]], list[dict[str, Any]]]:
    eligible_predictions = [
        {**prediction, "label": normalize_label(str(prediction["label"]))}
        for prediction in predictions
        if normalize_label(str(prediction["label"])) in CLASS_COLORS
    ]
    unmatched_truth = set(range(len(ground_truth)))
    matches: list[Match] = []
    false_positives: list[dict[str, Any]] = []
    ordered_predictions = sorted(
        eligible_predictions,
        key=lambda prediction: float(prediction["score"]),
        reverse=True,
    )
    for prediction in ordered_predictions:
        candidates = [
            index
            for index in unmatched_truth
            if ground_truth[index]["class_name"] == prediction["label"]
        ]
        best_iou, best_index = max(
            (
                (
                    box_iou(
                        cast(list[float], prediction["bbox_xyxy"]),
                        ground_truth[index]["bbox_xyxy"],
                    ),
                    index,
                )
                for index in candidates
            ),
            default=(0.0, None),
        )
        if best_index is not None and best_iou >= iou_threshold:
            matches.append(
                Match(
                    prediction=prediction,
                    ground_truth=ground_truth[best_index],
                    iou=best_iou,
                )
            )
            unmatched_truth.remove(best_index)
        else:
            false_positives.append(prediction)
    false_negatives = [ground_truth[index] for index in sorted(unmatched_truth)]
    return matches, false_positives, false_negatives


def predictions_by_frame(path: Path) -> dict[int, list[dict[str, Any]]]:
    return {
        int(record["camera_frame"]): record.get("detections", []) for record in read_jsonl(path)
    }


def ground_truth_by_frame(path: Path) -> dict[int, list[dict[str, Any]]]:
    return {int(record["camera_frame"]): record["objects"] for record in read_jsonl(path)}


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    video = {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream["r_frame_rate"],
        "avg_frame_rate": stream["avg_frame_rate"],
        "frame_count": int(stream["nb_frames"]),
        "duration_seconds": float(payload["format"]["duration"]),
        "size_bytes": int(payload["format"]["size"]),
    }
    return video


def validate_sources() -> dict[str, dict[str, Any]]:
    required = (
        FULL_VIDEO,
        ZOOM_VIDEO,
        PREDICTIONS_JSONL,
        GROUND_TRUTH_JSONL,
        CLEAN_FRAME_119,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required source files are missing. {missing}")

    full_probe = probe_video(FULL_VIDEO)
    zoom_probe = probe_video(ZOOM_VIDEO)
    expected_video_values = {
        "frame_count": SOURCE_FRAME_COUNT,
        "r_frame_rate": f"{SOURCE_FPS.numerator}/{SOURCE_FPS.denominator}",
        "avg_frame_rate": f"{SOURCE_FPS.numerator}/{SOURCE_FPS.denominator}",
    }
    for name, probe, dimensions in (
        ("full", full_probe, FULL_DIMENSIONS),
        ("zoom", zoom_probe, ZOOM_DIMENSIONS),
    ):
        actual_values = {
            "frame_count": probe["frame_count"],
            "r_frame_rate": probe["r_frame_rate"],
            "avg_frame_rate": probe["avg_frame_rate"],
        }
        if actual_values != expected_video_values:
            raise ValueError(f"Unexpected {name} video timing. {actual_values}")
        if (probe["width"], probe["height"]) != dimensions:
            raise ValueError(
                f"Unexpected {name} video dimensions. {probe['width']}x{probe['height']}"
            )

    sources: dict[str, dict[str, Any]] = {}
    for key, path in (
        ("full_video", FULL_VIDEO),
        ("zoom_video", ZOOM_VIDEO),
        ("predictions", PREDICTIONS_JSONL),
        ("ground_truth", GROUND_TRUTH_JSONL),
        ("clean_frame_119", CLEAN_FRAME_119),
    ):
        sources[key] = {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    sources["full_video"]["media"] = full_probe
    sources["zoom_video"]["media"] = zoom_probe
    return sources


def find_false_positive_frames(
    ground_truth: dict[int, list[dict[str, Any]]],
    predictions: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    false_positives: dict[int, list[dict[str, Any]]] = {}
    for camera_frame in sorted(ground_truth):
        _, frame_false_positives, _ = match_detections(
            ground_truth[camera_frame],
            predictions.get(camera_frame, []),
        )
        if frame_false_positives:
            false_positives[camera_frame] = frame_false_positives
    actual_frames = tuple(false_positives)
    if actual_frames != EXPECTED_FALSE_POSITIVE_FRAMES:
        raise ValueError(
            "The verified false-positive frames changed. "
            f"Expected {EXPECTED_FALSE_POSITIVE_FRAMES}, found {actual_frames}."
        )
    return false_positives


def draw_dashed_rectangle(
    drawing: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: str,
    width: int = 6,
    dash_length: int = 18,
    gap_length: int = 12,
) -> None:
    x1, y1, x2, y2 = (round(value) for value in box)
    step = dash_length + gap_length
    for start in range(x1, x2 + 1, step):
        end = min(start + dash_length, x2)
        drawing.line((start, y1, end, y1), fill=color, width=width)
        drawing.line((start, y2, end, y2), fill=color, width=width)
    for start in range(y1, y2 + 1, step):
        end = min(start + dash_length, y2)
        drawing.line((x1, start, x1, end), fill=color, width=width)
        drawing.line((x2, start, x2, end), fill=color, width=width)


def full_box_to_zoom(box: list[float]) -> tuple[float, float, float, float]:
    crop_x, crop_y, _, _ = ZOOM_CROP
    scale = 3.0
    return (
        (box[0] - crop_x) * scale,
        (box[1] - crop_y) * scale,
        (box[2] - crop_x) * scale,
        (box[3] - crop_y) * scale,
    )


def build_correct_poster(
    matches: list[Match],
) -> tuple[Image.Image, dict[str, Any]]:
    crop_x, crop_y, crop_width, crop_height = ZOOM_CROP
    with Image.open(CLEAN_FRAME_119) as source:
        clean_source = source.convert("RGB")
        clean_source_rgb_sha256 = sha256_rgb(clean_source)
        image = clean_source.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    image = image.resize(ZOOM_DIMENSIONS, Image.Resampling.LANCZOS)
    drawing = ImageDraw.Draw(image)
    for match in matches:
        draw_dashed_rectangle(
            drawing,
            full_box_to_zoom(match.ground_truth["bbox_xyxy"]),
            GROUND_TRUTH_COLOR,
        )
    for match in matches:
        color = CLASS_COLORS[match.prediction["label"]]
        drawing.rectangle(
            tuple(round(value) for value in full_box_to_zoom(match.prediction["bbox_xyxy"])),
            outline=color,
            width=6,
        )
    image.save(CORRECT_POSTER, format="PNG", optimize=False, compress_level=9)
    record = {
        "camera_frame": CORRECT_CAMERA_FRAME,
        "source_video_frame": CORRECT_CAMERA_FRAME - CAMERA_FRAME_OFFSET,
        "clean_source_rgb_sha256": clean_source_rgb_sha256,
        "poster_rgb_sha256": sha256_rgb(image),
        "matches": [
            {
                "label": match.prediction["label"],
                "score": match.prediction["score"],
                "prediction_bbox_xyxy": match.prediction["bbox_xyxy"],
                "ground_truth_bbox_xyxy": match.ground_truth["bbox_xyxy"],
                "iou": match.iou,
            }
            for match in matches
        ],
    }
    return image, record


def extract_frames(
    video: Path,
    source_indices: list[int],
    destination: Path,
) -> dict[int, Path]:
    unique_indices = sorted(set(source_indices))
    if not unique_indices:
        return {}
    expression = "+".join(f"eq(n\\,{index})" for index in unique_indices)
    pattern = destination / "selected_%06d.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select='{expression}'",
            "-fps_mode",
            "passthrough",
            str(pattern),
        ],
        check=True,
    )
    extracted = sorted(destination.glob("selected_*.png"))
    if len(extracted) != len(unique_indices):
        raise RuntimeError(
            f"Expected {len(unique_indices)} selected frames, found {len(extracted)}."
        )
    return dict(zip(unique_indices, extracted, strict=True))


def false_positive_timeline() -> list[int]:
    first_before = list(range(337, 341))
    first_after = list(range(342, 346))
    second_before = list(range(412, 416))
    second_after = list(range(417, 419))
    timeline: list[int] = []
    for camera_frame in first_before:
        timeline.extend([camera_frame] * CONTEXT_REPETITIONS)
    timeline.extend([341] * FALSE_POSITIVE_HOLD_FRAMES)
    for camera_frame in first_after:
        timeline.extend([camera_frame] * CONTEXT_REPETITIONS)
    for camera_frame in second_before:
        timeline.extend([camera_frame] * CONTEXT_REPETITIONS)
    timeline.extend([416] * FALSE_POSITIVE_HOLD_FRAMES)
    for camera_frame in second_after:
        timeline.extend([camera_frame] * CONTEXT_REPETITIONS)
    return timeline


def encode_image_sequence(
    frames_directory: Path,
    output: Path,
    frame_count: int,
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(OUTPUT_FPS),
            "-i",
            str(frames_directory / "%06d.png"),
            "-frames:v",
            str(frame_count),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(OUTPUT_FPS),
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-movflags",
            "+faststart",
            output,
        ],
        check=True,
    )


def link_timeline_frames(
    source_frames: dict[int, Path],
    camera_timeline: list[int],
    destination: Path,
) -> None:
    destination.mkdir()
    for output_index, camera_frame in enumerate(camera_timeline):
        source_index = camera_frame - CAMERA_FRAME_OFFSET
        os.link(source_frames[source_index], destination / f"{output_index:06d}.png")


def build_static_video(poster: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="correct_hold_", dir=OUTPUT_DIRECTORY) as temporary:
        frames_directory = Path(temporary)
        for output_index in range(CORRECT_VIDEO_FRAMES):
            os.link(poster, frames_directory / f"{output_index:06d}.png")
        encode_image_sequence(frames_directory, output, CORRECT_VIDEO_FRAMES)


def build_false_positive_assets(
    false_positives: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timeline = false_positive_timeline()
    selected_camera_frames = sorted(set(timeline))
    selected_source_indices = [
        camera_frame - CAMERA_FRAME_OFFSET for camera_frame in selected_camera_frames
    ]
    with tempfile.TemporaryDirectory(prefix="false_positive_", dir=OUTPUT_DIRECTORY) as temporary:
        temporary_directory = Path(temporary)
        zoom_frames_directory = temporary_directory / "zoom"
        zoom_frames_directory.mkdir()
        source_frames = extract_frames(
            ZOOM_VIDEO,
            selected_source_indices,
            zoom_frames_directory,
        )
        full_frames_directory = temporary_directory / "full"
        full_frames_directory.mkdir()
        full_source_frames = extract_frames(
            FULL_VIDEO,
            [camera_frame - CAMERA_FRAME_OFFSET for camera_frame in EXPECTED_FALSE_POSITIVE_FRAMES],
            full_frames_directory,
        )

        timeline_directory = temporary_directory / "timeline"
        link_timeline_frames(source_frames, timeline, timeline_directory)
        encode_image_sequence(timeline_directory, FALSE_POSITIVE_VIDEO, len(timeline))

        poster_source_index = EXPECTED_FALSE_POSITIVE_FRAMES[0] - CAMERA_FRAME_OFFSET
        with Image.open(source_frames[poster_source_index]) as poster_source:
            poster = poster_source.convert("RGB")
        poster.save(
            FALSE_POSITIVE_POSTER,
            format="PNG",
            optimize=False,
            compress_level=9,
        )

        selected_records = []
        for camera_frame in selected_camera_frames:
            source_index = camera_frame - CAMERA_FRAME_OFFSET
            with Image.open(source_frames[source_index]) as image:
                decoded_hash = sha256_rgb(image)
            selected_records.append(
                {
                    "camera_frame": camera_frame,
                    "source_video_frame": source_index,
                    "zoom_decoded_rgb_sha256": decoded_hash,
                    "output_occurrences": timeline.count(camera_frame),
                    "is_false_positive": camera_frame in false_positives,
                    "false_positives": false_positives.get(camera_frame, []),
                }
            )
            if camera_frame in false_positives:
                with Image.open(full_source_frames[source_index]) as full_image:
                    selected_records[-1]["full_decoded_rgb_sha256"] = sha256_rgb(full_image)

        poster_record = {
            "camera_frame": EXPECTED_FALSE_POSITIVE_FRAMES[0],
            "source_video_frame": (EXPECTED_FALSE_POSITIVE_FRAMES[0] - CAMERA_FRAME_OFFSET),
            "source": "zoom_video",
            "poster_rgb_sha256": sha256_rgb(poster),
        }
    timeline_records = [
        {
            "start_camera_frame": start,
            "end_camera_frame": end,
            "source_video_start_frame": start - CAMERA_FRAME_OFFSET,
            "source_video_end_frame": end - CAMERA_FRAME_OFFSET,
        }
        for start, end in FALSE_POSITIVE_WINDOWS
    ]
    return selected_records, [*timeline_records, poster_record]


def media_record(path: Path) -> dict[str, Any]:
    probe = probe_video(path)
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "media": probe,
    }


def image_record(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
    }


def write_manifest(
    sources: dict[str, dict[str, Any]],
    correct_record: dict[str, Any],
    false_positive_records: list[dict[str, Any]],
    false_positive_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "frame_numbering": {
            "camera_frame_offset": CAMERA_FRAME_OFFSET,
            "source_video_frame_base": 0,
            "output_video_frame_base": 0,
        },
        "style": {
            "burned_in_text": False,
            "false_positive": {
                "color": FALSE_POSITIVE_COLOR,
                "line_style": "solid",
            },
            "predictions": {
                "boat": {"color": BOAT_COLOR, "line_style": "solid"},
                "buoy": {"color": BUOY_COLOR, "line_style": "solid"},
            },
            "ground_truth": {
                "boat": {"color": GROUND_TRUTH_COLOR, "line_style": "dashed"},
                "buoy": {"color": GROUND_TRUTH_COLOR, "line_style": "dashed"},
            },
        },
        "selection": {
            "iou_threshold": IOU_THRESHOLD,
            "correct_camera_frame": CORRECT_CAMERA_FRAME,
            "false_positive_camera_frames": list(EXPECTED_FALSE_POSITIVE_FRAMES),
        },
        "sources": sources,
        "assets": {
            "correct_comparison": {
                "video": media_record(CORRECT_VIDEO),
                "poster": image_record(CORRECT_POSTER),
                "selected_frame": correct_record,
                "timeline": {
                    "output_frame_count": CORRECT_VIDEO_FRAMES,
                    "output_fps": OUTPUT_FPS,
                    "duration_seconds": CORRECT_VIDEO_FRAMES / OUTPUT_FPS,
                    "camera_frame": CORRECT_CAMERA_FRAME,
                    "hold_frames": CORRECT_VIDEO_FRAMES,
                },
            },
            "false_positive_episodes": {
                "video": media_record(FALSE_POSITIVE_VIDEO),
                "poster": image_record(FALSE_POSITIVE_POSTER),
                "selected_frames": false_positive_records,
                "verified_windows": false_positive_windows,
                "timeline": {
                    "output_frame_count": len(false_positive_timeline()),
                    "output_fps": OUTPUT_FPS,
                    "duration_seconds": len(false_positive_timeline()) / OUTPUT_FPS,
                    "context_repetitions": CONTEXT_REPETITIONS,
                    "false_positive_hold_frames": FALSE_POSITIVE_HOLD_FRAMES,
                    "camera_frames": false_positive_timeline(),
                },
            },
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for asset_group in manifest["assets"].values():
        for asset_type in ("video", "poster"):
            record = asset_group[asset_type]
            path = REPOSITORY_ROOT / record["path"]
            actual_hash = sha256_file(path)
            if actual_hash != record["sha256"]:
                raise ValueError(
                    f"Hash mismatch for {record['path']}. "
                    f"Expected {record['sha256']}, found {actual_hash}."
                )
    correct_probe = probe_video(CORRECT_VIDEO)
    if (
        correct_probe["frame_count"] != CORRECT_VIDEO_FRAMES
        or correct_probe["r_frame_rate"] != f"{OUTPUT_FPS}/1"
        or correct_probe["duration_seconds"] != CORRECT_VIDEO_FRAMES / OUTPUT_FPS
    ):
        raise ValueError(f"Incorrect comparison video timing. {correct_probe}")
    false_positive_probe = probe_video(FALSE_POSITIVE_VIDEO)
    expected_false_positive_frames = len(false_positive_timeline())
    if (
        false_positive_probe["frame_count"] != expected_false_positive_frames
        or false_positive_probe["r_frame_rate"] != f"{OUTPUT_FPS}/1"
        or false_positive_probe["duration_seconds"] != expected_false_positive_frames / OUTPUT_FPS
    ):
        raise ValueError(f"Incorrect false-positive video timing. {false_positive_probe}")


def build() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sources = validate_sources()
    predictions = predictions_by_frame(PREDICTIONS_JSONL)
    ground_truth = ground_truth_by_frame(GROUND_TRUTH_JSONL)
    false_positives = find_false_positive_frames(ground_truth, predictions)
    correct_matches, correct_false_positives, _ = match_detections(
        ground_truth[CORRECT_CAMERA_FRAME],
        predictions[CORRECT_CAMERA_FRAME],
    )
    if correct_false_positives:
        raise ValueError(f"Correct comparison frame has false positives. {correct_false_positives}")
    if {match.prediction["label"] for match in correct_matches} != {"boat", "buoy"}:
        raise ValueError("Correct comparison frame must match both a boat and a buoy.")

    _, correct_record = build_correct_poster(correct_matches)
    build_static_video(CORRECT_POSTER, CORRECT_VIDEO)
    false_positive_records, false_positive_windows = build_false_positive_assets(false_positives)
    write_manifest(
        sources,
        correct_record,
        false_positive_records,
        false_positive_windows,
    )
    verify_manifest()


def main() -> None:
    arguments = parse_arguments()
    if arguments.verify_only:
        verify_manifest()
    else:
        build()


if __name__ == "__main__":
    main()
