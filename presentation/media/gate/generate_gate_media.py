#!/usr/bin/env python3
"""Build deterministic, text-free media for the radar confidence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_detection_experiments import (  # noqa: E402
    box_ioa,
    box_iou,
    normalize_label,
    radar_supported,
)

OUTPUT_DIR = ROOT / "presentation" / "media" / "gate"
POSTER_SOURCE = OUTPUT_DIR / "poster.typ"
RAW_VIDEO = ROOT / "01_input_raw.mp4"
SHARDS = (
    ROOT / "experiments" / "detection_comparison" / "shard_000_149.jsonl",
    ROOT / "experiments" / "detection_comparison" / "shard_150_299.jsonl",
)
FINAL_PREDICTIONS = ROOT / "experiments" / "detection_comparison" / "radar_confidence_gated.jsonl"
GROUND_TRUTH = ROOT / "datasets" / "dream_fusion_yolo" / "manifest.jsonl"
EXISTING_PREDICTION_ONLY_VIDEO = ROOT / "08_exp1_vision_only_detections.mp4"

SOURCE_WIDTH = 5320.0
SOURCE_HEIGHT = 3032.0
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1094
POSTER_WIDTH = 1920
POSTER_HEIGHT = 1080
LAYOUT_WIDTH = 960.0
LAYOUT_HEIGHT = 540.0
FPS = 60
FIRST_CAMERA_FRAME = 119
GLOBAL_THRESHOLD = 0.18
RADAR_THRESHOLD = 0.16
MINIMUM_IOA = 0.25
IOU_THRESHOLD = 0.5

COLORS = {
    "garnet": "#73000A",
    "black": "#000000",
    "white": "#FFFFFF",
    "dark": "#363636",
    "neutral": "#A2A2A2",
    "light": "#ECECEC",
    "rose": "#CC2E40",
    "atlantic": "#466A9F",
    "congaree": "#1F414D",
    "horseshoe": "#65780B",
    "grass": "#CED318",
    "honeycomb": "#A49137",
}

ASSET_FILES = (
    "support_map_poster.png",
    "center_inside_example.png",
    "ioa_25_and_touching_rejection.png",
    "gate_acceptance_animation.mp4",
    "false_positive_episode_montage.png",
    "correct_predictions_over_dashed_ground_truth.mp4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--keep-build", action="store_true")
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


def support_reason(
    box: Sequence[float],
    radar_regions: Sequence[Sequence[float]],
    minimum_ioa: float = MINIMUM_IOA,
) -> tuple[str | None, int | None, float]:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    best_ioa = 0.0
    best_index: int | None = None
    for index, region in enumerate(radar_regions):
        ioa = box_ioa(list(box), list(region))
        if ioa > best_ioa:
            best_ioa = ioa
            best_index = index
        if region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]:
            return "center_inside", index, ioa
    if best_ioa >= minimum_ioa:
        return "minimum_ioa", best_index, best_ioa
    return None, best_index, best_ioa


def gate_decision(score: float, supported: bool) -> str:
    if score >= GLOBAL_THRESHOLD:
        return "accepted_global"
    if score >= RADAR_THRESHOLD and supported:
        return "accepted_radar"
    return "rejected"


def select_examples(
    shards: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    center_case: dict[str, Any] | None = None
    high_case: dict[str, Any] | None = None
    low_case: dict[str, Any] | None = None
    for record in sorted(shards, key=lambda item: int(item["camera_frame"])):
        regions = record["radar_regions"]
        for detection in record["radar_gated_detections"]:
            score = float(detection["score"])
            reason, region_index, ioa = support_reason(detection["bbox_xyxy"], regions)
            candidate = {
                "camera_frame": int(record["camera_frame"]),
                "detection": detection,
                "radar_regions": regions,
                "support_reason": reason,
                "support_region_index": region_index,
                "maximum_ioa": ioa,
            }
            if high_case is None and score >= GLOBAL_THRESHOLD:
                high_case = candidate
            if (
                center_case is None
                and RADAR_THRESHOLD <= score < GLOBAL_THRESHOLD
                and reason == "center_inside"
            ):
                center_case = candidate
            if low_case is None and score < RADAR_THRESHOLD and reason is not None:
                low_case = candidate
        if high_case and center_case and low_case:
            break
    if high_case is None or center_case is None or low_case is None:
        raise ValueError("committed shards do not contain all required gate examples")
    return {
        "high": high_case,
        "center": center_case,
        "low": low_case,
    }


def categorize_detections(
    ground_truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any], float]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    unmatched_truth = set(range(len(ground_truth)))
    true_positives: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    false_positives: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = [
        {**prediction, "label": normalize_label(str(prediction["label"]))}
        for prediction in predictions
        if normalize_label(str(prediction["label"])) in {"boat", "buoy"}
    ]
    for prediction in sorted(
        eligible,
        key=lambda item: float(item["score"]),
        reverse=True,
    ):
        candidates = [
            index
            for index in unmatched_truth
            if ground_truth[index]["class_name"] == prediction["label"]
        ]
        best_iou, best_index = max(
            (
                (
                    box_iou(
                        list(prediction["bbox_xyxy"]),
                        list(ground_truth[index]["bbox_xyxy"]),
                    ),
                    index,
                )
                for index in candidates
            ),
            default=(0.0, None),
        )
        if best_index is not None and best_iou >= IOU_THRESHOLD:
            true_positives.append((prediction, ground_truth[best_index], best_iou))
            unmatched_truth.remove(best_index)
        else:
            false_positives.append(prediction)
    false_negatives = [ground_truth[index] for index in sorted(unmatched_truth)]
    return true_positives, false_positives, false_negatives


def choose_false_positive_episode(
    ground_truth_by_frame: dict[int, list[dict[str, Any]]],
    predictions_by_frame: dict[int, list[dict[str, Any]]],
) -> list[int]:
    frames_with_false_positives = []
    for camera_frame in sorted(ground_truth_by_frame):
        _, false_positives, _ = categorize_detections(
            ground_truth_by_frame[camera_frame],
            predictions_by_frame.get(camera_frame, []),
        )
        if false_positives:
            frames_with_false_positives.append(camera_frame)

    episodes: list[list[int]] = []
    for camera_frame in frames_with_false_positives:
        if not episodes or camera_frame != episodes[-1][-1] + 1:
            episodes.append([camera_frame])
        else:
            episodes[-1].append(camera_frame)
    if not episodes:
        raise ValueError("final predictions contain no false-positive episode")
    longest = max(episodes, key=lambda values: (len(values), -values[0]))
    start = max(FIRST_CAMERA_FRAME, longest[0] - 1)
    end = min(FIRST_CAMERA_FRAME + 299, longest[-1] + 1)
    return list(range(start, end + 1))


def extract_video_frames(
    camera_frames: Iterable[int],
    destination: Path,
) -> dict[int, Path]:
    selected = sorted(set(camera_frames))
    destination.mkdir(parents=True, exist_ok=True)
    frame_indices = [camera_frame - FIRST_CAMERA_FRAME for camera_frame in selected]
    if any(index < 0 or index >= 300 for index in frame_indices):
        raise ValueError("camera frame lies outside the committed 300-frame source video")
    expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(RAW_VIDEO),
            "-vf",
            f"select='{expression}'",
            "-vsync",
            "0",
            str(destination / "%06d.png"),
        ],
        check=True,
    )
    extracted = sorted(destination.glob("*.png"))
    if len(extracted) != len(selected):
        raise RuntimeError(f"expected {len(selected)} extracted frames, found {len(extracted)}")
    return dict(zip(selected, extracted, strict=True))


def crop_frame(
    source: Path,
    source_crop: Sequence[float],
    destination: Path,
    width: int,
    height: int,
) -> None:
    sx = VIDEO_WIDTH / SOURCE_WIDTH
    sy = VIDEO_HEIGHT / SOURCE_HEIGHT
    crop = (
        round(source_crop[0] * sx),
        round(source_crop[1] * sy),
        round(source_crop[2] * sx),
        round(source_crop[3] * sy),
    )
    with Image.open(source) as image:
        prepared = image.convert("RGB").crop(crop)
        prepared = prepared.resize((width, height), Image.Resampling.LANCZOS)
        prepared.save(destination, format="PNG", optimize=False, compress_level=9)


def project_box(
    box: Sequence[float],
    source_crop: Sequence[float],
    panel: Sequence[float],
) -> list[float]:
    crop_width = source_crop[2] - source_crop[0]
    crop_height = source_crop[3] - source_crop[1]
    panel_x, panel_y, panel_width, panel_height = panel
    projected = [
        panel_x + (box[0] - source_crop[0]) * panel_width / crop_width,
        panel_y + (box[1] - source_crop[1]) * panel_height / crop_height,
        panel_x + (box[2] - source_crop[0]) * panel_width / crop_width,
        panel_y + (box[3] - source_crop[1]) * panel_height / crop_height,
    ]
    return [
        max(panel_x, min(panel_x + panel_width, projected[0])),
        max(panel_y, min(panel_y + panel_height, projected[1])),
        max(panel_x, min(panel_x + panel_width, projected[2])),
        max(panel_y, min(panel_y + panel_height, projected[3])),
    ]


def rectangle(
    box: Sequence[float],
    stroke: str,
    width: float,
    *,
    fill: str | None = None,
    alpha: float = 0.0,
    dash: str = "solid",
) -> dict[str, Any]:
    return {
        "x1": box[0],
        "y1": box[1],
        "x2": box[2],
        "y2": box[3],
        "stroke": stroke,
        "width": width,
        "fill": fill,
        "alpha": alpha,
        "dash": dash,
    }


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str,
    width: float,
    dash: str = "solid",
) -> dict[str, Any]:
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "stroke": stroke,
        "width": width,
        "dash": dash,
    }


def circle(
    cx: float,
    cy: float,
    radius: float,
    stroke: str,
    width: float,
    *,
    fill: str | None = None,
    alpha: float = 0.0,
) -> dict[str, Any]:
    return {
        "cx": cx,
        "cy": cy,
        "radius": radius,
        "stroke": stroke,
        "width": width,
        "fill": fill,
        "alpha": alpha,
    }


def empty_figure(background: str = COLORS["black"]) -> dict[str, Any]:
    return {
        "background": background,
        "images": [],
        "rectangles": [],
        "lines": [],
        "circles": [],
    }


def add_x(
    figure: dict[str, Any],
    box: Sequence[float],
    color: str = COLORS["rose"],
    width: float = 5.0,
) -> None:
    figure["lines"].extend(
        [
            line(box[0], box[1], box[2], box[3], color, width),
            line(box[0], box[3], box[2], box[1], color, width),
        ]
    )


def add_check(
    figure: dict[str, Any],
    x: float,
    y: float,
    scale: float,
    color: str,
) -> None:
    figure["lines"].extend(
        [
            line(x, y, x + 0.34 * scale, y + 0.34 * scale, color, 6.0),
            line(
                x + 0.34 * scale,
                y + 0.34 * scale,
                x + scale,
                y - 0.42 * scale,
                color,
                6.0,
            ),
        ]
    )


def write_figure_data(path: Path, figure: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(figure, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def compile_figure(
    figure: dict[str, Any],
    output: Path,
    build_dir: Path,
    name: str,
) -> None:
    data_path = build_dir / f"{name}.json"
    write_figure_data(data_path, figure)
    data_argument = data_path.relative_to(OUTPUT_DIR).as_posix()
    subprocess.run(
        [
            "typst",
            "compile",
            "--input",
            f"data={data_argument}",
            "--ppi",
            "144",
            str(POSTER_SOURCE),
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    with Image.open(output) as image:
        if image.size != (POSTER_WIDTH, POSTER_HEIGHT):
            raise RuntimeError(f"{output.name} rendered at unexpected size {image.size}")


def prepare_panel_image(
    raw_frame: Path,
    crop: Sequence[float],
    build_dir: Path,
    name: str,
    panel: Sequence[float],
) -> Path:
    destination = build_dir / f"{name}.png"
    width = max(2, round(panel[2] * 2))
    height = max(2, round(panel[3] * 2))
    crop_frame(raw_frame, crop, destination, width, height)
    return destination


def image_item(path: Path, panel: Sequence[float]) -> dict[str, Any]:
    return {
        "path": path.relative_to(OUTPUT_DIR).as_posix(),
        "x": panel[0],
        "y": panel[1],
        "width": panel[2],
        "height": panel[3],
    }


def support_map_figure(
    raw_frame: Path,
    record: dict[str, Any],
    build_dir: Path,
) -> dict[str, Any]:
    crop = [0.0, 0.0, SOURCE_WIDTH, SOURCE_HEIGHT]
    panel = [0.0, 0.0, LAYOUT_WIDTH, LAYOUT_HEIGHT]
    prepared = prepare_panel_image(
        raw_frame,
        crop,
        build_dir,
        "support_map_source",
        panel,
    )
    figure = empty_figure()
    figure["images"].append(image_item(prepared, panel))
    for region in record["radar_regions"]:
        figure["rectangles"].append(
            rectangle(
                project_box(region, crop, panel),
                COLORS["atlantic"],
                3.0,
                fill=COLORS["atlantic"],
                alpha=0.72,
            )
        )
    for detection in record["radar_gated_detections"]:
        score = float(detection["score"])
        supported = radar_supported(
            detection["bbox_xyxy"],
            record["radar_regions"],
            MINIMUM_IOA,
        )
        decision = gate_decision(score, supported)
        color = (
            COLORS["grass"]
            if decision == "accepted_global"
            else COLORS["honeycomb"]
            if decision == "accepted_radar"
            else COLORS["rose"]
        )
        box = project_box(detection["bbox_xyxy"], crop, panel)
        figure["rectangles"].append(rectangle(box, color, 3.0))
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        figure["circles"].append(
            circle(
                center_x,
                center_y,
                3.5,
                COLORS["white"],
                1.5,
                fill=color,
            )
        )
        if decision == "rejected":
            add_x(figure, box, width=2.5)
    return figure


def center_inside_figure(
    raw_frame: Path,
    example: dict[str, Any],
    build_dir: Path,
) -> dict[str, Any]:
    detection = example["detection"]
    box = detection["bbox_xyxy"]
    crop = [
        min(region[0] for region in example["radar_regions"]) - 80,
        min(region[1] for region in example["radar_regions"]) - 80,
        max(region[2] for region in example["radar_regions"]) + 80,
        max(region[3] for region in example["radar_regions"]) + 80,
    ]
    panel = [0.0, 0.0, LAYOUT_WIDTH, LAYOUT_HEIGHT]
    prepared = prepare_panel_image(
        raw_frame,
        crop,
        build_dir,
        "center_inside_source",
        panel,
    )
    figure = empty_figure()
    figure["images"].append(image_item(prepared, panel))
    for region in example["radar_regions"]:
        figure["rectangles"].append(
            rectangle(
                project_box(region, crop, panel),
                COLORS["atlantic"],
                3.0,
                fill=COLORS["atlantic"],
                alpha=0.76,
            )
        )
    projected = project_box(box, crop, panel)
    figure["rectangles"].append(rectangle(projected, COLORS["honeycomb"], 5.0))
    center_x = (projected[0] + projected[2]) / 2
    center_y = (projected[1] + projected[3]) / 2
    figure["circles"].extend(
        [
            circle(
                center_x,
                center_y,
                10.0,
                COLORS["white"],
                2.0,
                fill=COLORS["honeycomb"],
            ),
            circle(
                center_x,
                center_y,
                24.0,
                COLORS["white"],
                2.0,
            ),
        ]
    )
    add_check(
        figure,
        min(LAYOUT_WIDTH - 90, projected[2] + 24),
        max(50, projected[1] - 28),
        44,
        COLORS["grass"],
    )
    return figure


def ioa_touching_figure() -> dict[str, Any]:
    figure = empty_figure(COLORS["dark"])
    left_panel = [30.0, 30.0, 435.0, 480.0]
    right_panel = [495.0, 30.0, 435.0, 480.0]
    for panel in (left_panel, right_panel):
        figure["rectangles"].append(
            rectangle(panel_to_box(panel), COLORS["white"], 2.0, fill=COLORS["black"])
        )

    left_region = [95.0, 150.0, 330.0, 390.0]
    left_candidate = [271.25, 150.0, 506.25, 390.0]
    left_intersection = [271.25, 150.0, 330.0, 390.0]
    figure["rectangles"].extend(
        [
            rectangle(
                left_region,
                COLORS["atlantic"],
                5.0,
                fill=COLORS["atlantic"],
                alpha=0.70,
            ),
            rectangle(
                left_intersection,
                COLORS["grass"],
                1.0,
                fill=COLORS["grass"],
                alpha=0.18,
            ),
            rectangle(left_candidate, COLORS["white"], 5.0),
        ]
    )
    figure["circles"].append(
        circle(
            (left_candidate[0] + left_candidate[2]) / 2,
            (left_candidate[1] + left_candidate[3]) / 2,
            8.0,
            COLORS["white"],
            2.0,
            fill=COLORS["honeycomb"],
        )
    )
    add_check(figure, 375.0, 98.0, 48.0, COLORS["grass"])

    right_region = [555.0, 150.0, 790.0, 390.0]
    right_candidate = [790.0, 150.0, 925.0, 390.0]
    figure["rectangles"].extend(
        [
            rectangle(
                right_region,
                COLORS["atlantic"],
                5.0,
                fill=COLORS["atlantic"],
                alpha=0.70,
            ),
            rectangle(right_candidate, COLORS["white"], 5.0),
        ]
    )
    figure["circles"].append(
        circle(
            (right_candidate[0] + right_candidate[2]) / 2,
            (right_candidate[1] + right_candidate[3]) / 2,
            8.0,
            COLORS["white"],
            2.0,
            fill=COLORS["rose"],
        )
    )
    add_x(figure, [810.0, 72.0, 890.0, 132.0], width=8.0)
    return figure


def panel_to_box(panel: Sequence[float]) -> list[float]:
    return [panel[0], panel[1], panel[0] + panel[2], panel[1] + panel[3]]


def acceptance_state_figure(
    raw_frames: dict[int, Path],
    examples: dict[str, dict[str, Any]],
    build_dir: Path,
    state: int,
) -> dict[str, Any]:
    figure = empty_figure(COLORS["dark"])
    keys = ("high", "center", "low")
    panels = (
        [18.0, 92.0, 290.0, 410.0],
        [335.0, 92.0, 290.0, 410.0],
        [652.0, 92.0, 290.0, 410.0],
    )
    crops: dict[str, list[float]] = {}
    for key, panel in zip(keys, panels, strict=True):
        example = examples[key]
        box = example["detection"]["bbox_xyxy"]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        crop_width = 720.0
        crop_height = crop_width * panel[3] / panel[2]
        crop = [
            center_x - crop_width / 2,
            center_y - crop_height / 2,
            center_x + crop_width / 2,
            center_y + crop_height / 2,
        ]
        crops[key] = crop
        prepared = prepare_panel_image(
            raw_frames[example["camera_frame"]],
            crop,
            build_dir,
            f"acceptance_{key}",
            panel,
        )
        figure["images"].append(image_item(prepared, panel))
        figure["rectangles"].append(rectangle(panel_to_box(panel), COLORS["white"], 2.0))
        if key != "high":
            for region in example["radar_regions"]:
                figure["rectangles"].append(
                    rectangle(
                        project_box(region, crop, panel),
                        COLORS["atlantic"],
                        2.5,
                        fill=COLORS["atlantic"],
                        alpha=0.78,
                    )
                )

    upper_y = 38.0
    lower_y = 66.0
    figure["lines"].extend(
        [
            line(18.0, upper_y, 942.0, upper_y, COLORS["white"], 2.0, "dashed"),
            line(18.0, lower_y, 942.0, lower_y, COLORS["neutral"], 2.0, "dashed"),
        ]
    )
    meter_y = {"high": 22.0, "center": 51.0, "low": 78.0}

    for index, (key, panel) in enumerate(zip(keys, panels, strict=True)):
        example = examples[key]
        projected = project_box(
            example["detection"]["bbox_xyxy"],
            crops[key],
            panel,
        )
        active = index <= state
        decision = gate_decision(
            float(example["detection"]["score"]),
            example["support_reason"] is not None,
        )
        color = (
            COLORS["grass"] if decision in {"accepted_global", "accepted_radar"} else COLORS["rose"]
        )
        stroke = color if active else COLORS["white"]
        figure["rectangles"].append(rectangle(projected, stroke, 4.0))
        dot_x = panel[0] + panel[2] / 2
        figure["circles"].append(
            circle(
                dot_x,
                meter_y[key],
                9.0,
                COLORS["white"],
                2.0,
                fill=stroke,
            )
        )
        if active:
            if decision == "rejected":
                add_x(
                    figure,
                    [panel[0] + 104, 14, panel[0] + 186, 82],
                    color,
                    7.0,
                )
            else:
                add_check(
                    figure,
                    panel[0] + 112,
                    50,
                    54,
                    color,
                )
    return figure


def false_positive_montage_figure(
    raw_frames: dict[int, Path],
    camera_frames: Sequence[int],
    ground_truth_by_frame: dict[int, list[dict[str, Any]]],
    predictions_by_frame: dict[int, list[dict[str, Any]]],
    build_dir: Path,
) -> dict[str, Any]:
    figure = empty_figure(COLORS["black"])
    panels = (
        [8.0, 8.0, 468.0, 258.0],
        [484.0, 8.0, 468.0, 258.0],
        [8.0, 274.0, 468.0, 258.0],
        [484.0, 274.0, 468.0, 258.0],
    )
    crop = [2500.0, 1300.0, 3250.0, 1800.0]
    for camera_frame, panel in zip(camera_frames, panels, strict=True):
        prepared = prepare_panel_image(
            raw_frames[camera_frame],
            crop,
            build_dir,
            f"false_positive_{camera_frame}",
            panel,
        )
        figure["images"].append(image_item(prepared, panel))
        figure["rectangles"].append(rectangle(panel_to_box(panel), COLORS["white"], 2.0))
        true_positives, false_positives, _ = categorize_detections(
            ground_truth_by_frame[camera_frame],
            predictions_by_frame.get(camera_frame, []),
        )
        for _, truth, _ in true_positives:
            if truth["class_name"] == "boat":
                figure["rectangles"].append(
                    rectangle(
                        project_box(truth["bbox_xyxy"], crop, panel),
                        COLORS["white"],
                        3.0,
                        dash="dashed",
                    )
                )
        for prediction in false_positives:
            figure["rectangles"].append(
                rectangle(
                    project_box(prediction["bbox_xyxy"], crop, panel),
                    COLORS["rose"],
                    5.0,
                )
            )
    return figure


def correct_prediction_figure(
    raw_frame: Path,
    camera_frame: int,
    ground_truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    build_dir: Path,
) -> dict[str, Any]:
    crop = [2500.0, 1320.0, 3150.0, 1830.0]
    panel = [0.0, 0.0, LAYOUT_WIDTH, LAYOUT_HEIGHT]
    prepared = prepare_panel_image(
        raw_frame,
        crop,
        build_dir,
        f"correct_source_{camera_frame}",
        panel,
    )
    figure = empty_figure()
    figure["images"].append(image_item(prepared, panel))
    true_positives, _, _ = categorize_detections(ground_truth, predictions)
    for prediction, truth, _ in true_positives:
        truth_box = project_box(truth["bbox_xyxy"], crop, panel)
        prediction_box = project_box(prediction["bbox_xyxy"], crop, panel)
        color = COLORS["rose"] if prediction["label"] == "boat" else COLORS["grass"]
        figure["rectangles"].extend(
            [
                rectangle(
                    truth_box,
                    COLORS["white"],
                    3.0,
                    dash="dashed",
                ),
                rectangle(prediction_box, color, 4.0),
            ]
        )
    return figure


def encode_sequence(
    frames_dir: Path,
    output: Path,
    input_fps: int,
    output_fps: int,
    frame_count: int,
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(input_fps),
            "-i",
            str(frames_dir / "%06d.png"),
            "-frames:v",
            str(frame_count),
            "-r",
            str(output_fps),
            "-an",
            "-map_metadata",
            "-1",
            "-metadata",
            "creation_time=1970-01-01T00:00:00Z",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-x264-params",
            "keyint=60:min-keyint=60:scenecut=0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt",
            "-show_entries",
            "format=duration",
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
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream["r_frame_rate"],
        "avg_frame_rate": stream["avg_frame_rate"],
        "frame_count": int(stream["nb_frames"]),
        "pixel_format": stream["pix_fmt"],
        "duration_seconds": float(payload["format"]["duration"]),
    }


def build_manifest(
    output_dir: Path,
    examples: dict[str, dict[str, Any]],
    support_record: dict[str, Any],
    false_positive_frames: Sequence[int],
    correct_frames: Sequence[int],
) -> dict[str, Any]:
    source_paths = [RAW_VIDEO, *SHARDS, FINAL_PREDICTIONS, GROUND_TRUTH]
    sources = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in source_paths
    ]
    assets = []
    frame_selection = {
        "support_map_poster.png": [int(support_record["camera_frame"])],
        "center_inside_example.png": [examples["center"]["camera_frame"]],
        "ioa_25_and_touching_rejection.png": [],
        "gate_acceptance_animation.mp4": [
            examples[key]["camera_frame"] for key in ("high", "center", "low")
        ],
        "false_positive_episode_montage.png": list(false_positive_frames),
        "correct_predictions_over_dashed_ground_truth.mp4": list(correct_frames),
    }
    for filename in ASSET_FILES:
        path = output_dir / filename
        record: dict[str, Any] = {
            "path": filename,
            "sha256": sha256_file(path),
            "camera_frames": frame_selection[filename],
            "burned_in_text": False,
        }
        if path.suffix == ".png":
            with Image.open(path) as image:
                record.update(
                    {
                        "media_type": "image/png",
                        "width": image.width,
                        "height": image.height,
                    }
                )
        else:
            record.update(
                {
                    "media_type": "video/mp4",
                    **probe_video(path),
                }
            )
        assets.append(record)

    center = examples["center"]
    return {
        "schema_version": 1,
        "generator": "generate_gate_media.py",
        "deterministic": True,
        "source_coordinate_system": [int(SOURCE_WIDTH), int(SOURCE_HEIGHT)],
        "thresholds": {
            "global_acceptance_score": GLOBAL_THRESHOLD,
            "radar_acceptance_score": RADAR_THRESHOLD,
            "radar_minimum_ioa": MINIMUM_IOA,
            "evaluation_iou": IOU_THRESHOLD,
        },
        "sources": sources,
        "existing_prediction_only_base_video": {
            "path": EXISTING_PREDICTION_ONLY_VIDEO.relative_to(ROOT).as_posix(),
            "generated_or_duplicated": False,
            "sha256": (
                sha256_file(EXISTING_PREDICTION_ONLY_VIDEO)
                if EXISTING_PREDICTION_ONLY_VIDEO.exists()
                else None
            ),
        },
        "selections": {
            "center_inside": {
                "camera_frame": center["camera_frame"],
                "label": center["detection"]["label"],
                "score": center["detection"]["score"],
                "bbox_xyxy": center["detection"]["bbox_xyxy"],
                "radar_region_index": center["support_region_index"],
                "support_reason": center["support_reason"],
            },
            "ioa_geometry": {
                "provenance": "geometry_only",
                "accepted_candidate_area_fraction": 0.25,
                "accepted_center_inside": False,
                "touching_candidate_area_fraction": 0.0,
                "touching_center_inside": False,
            },
            "acceptance_animation": {
                key: {
                    "camera_frame": example["camera_frame"],
                    "score": example["detection"]["score"],
                    "support_reason": example["support_reason"],
                    "decision": gate_decision(
                        float(example["detection"]["score"]),
                        example["support_reason"] is not None,
                    ),
                }
                for key, example in examples.items()
            },
            "false_positive_episode": list(false_positive_frames),
            "correct_prediction_sequence": list(correct_frames),
        },
        "assets": assets,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir != OUTPUT_DIR.resolve():
        raise ValueError("output directory must remain presentation/media/gate")
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    for path in SHARDS:
        shards.extend(read_jsonl(path))
    shards_by_frame = {int(record["camera_frame"]): record for record in shards}
    examples = select_examples(shards)
    ground_truth_records = read_jsonl(GROUND_TRUTH)
    ground_truth_by_frame = {
        int(record["camera_frame"]): record["objects"] for record in ground_truth_records
    }
    prediction_records = read_jsonl(FINAL_PREDICTIONS)
    predictions_by_frame = {
        int(record["camera_frame"]): record["detections"] for record in prediction_records
    }
    false_positive_frames = choose_false_positive_episode(
        ground_truth_by_frame,
        predictions_by_frame,
    )
    if len(false_positive_frames) != 4:
        raise ValueError("false-positive montage requires a four-frame episode window")
    correct_frames = list(range(119, 179))
    support_record = shards_by_frame[203]
    required_frames = {
        support_record["camera_frame"],
        *(example["camera_frame"] for example in examples.values()),
        *false_positive_frames,
        *correct_frames,
    }

    temporary_context = tempfile.TemporaryDirectory(
        prefix=".gate_build_",
        dir=output_dir,
    )
    build_dir = Path(temporary_context.name)
    if args.keep_build:
        temporary_context._finalizer.detach()  # ty: ignore[unresolved-attribute]
    try:
        raw_frames = extract_video_frames(required_frames, build_dir / "raw")

        compile_figure(
            support_map_figure(
                raw_frames[int(support_record["camera_frame"])],
                support_record,
                build_dir,
            ),
            output_dir / "support_map_poster.png",
            build_dir,
            "support_map_poster",
        )
        compile_figure(
            center_inside_figure(
                raw_frames[examples["center"]["camera_frame"]],
                examples["center"],
                build_dir,
            ),
            output_dir / "center_inside_example.png",
            build_dir,
            "center_inside_example",
        )
        compile_figure(
            ioa_touching_figure(),
            output_dir / "ioa_25_and_touching_rejection.png",
            build_dir,
            "ioa_25_and_touching_rejection",
        )

        animation_keyframes = build_dir / "gate_animation"
        animation_keyframes.mkdir()
        for state in range(3):
            compile_figure(
                acceptance_state_figure(
                    raw_frames,
                    examples,
                    build_dir,
                    state,
                ),
                animation_keyframes / f"{state:06d}.png",
                build_dir,
                f"gate_animation_state_{state}",
            )
        animation_sequence = build_dir / "gate_animation_sequence"
        animation_sequence.mkdir()
        output_index = 0
        for state in range(3):
            source = animation_keyframes / f"{state:06d}.png"
            for _ in range(FPS):
                (animation_sequence / f"{output_index:06d}.png").symlink_to(source)
                output_index += 1
        encode_sequence(
            animation_sequence,
            output_dir / "gate_acceptance_animation.mp4",
            FPS,
            FPS,
            FPS * 3,
        )

        compile_figure(
            false_positive_montage_figure(
                raw_frames,
                false_positive_frames,
                ground_truth_by_frame,
                predictions_by_frame,
                build_dir,
            ),
            output_dir / "false_positive_episode_montage.png",
            build_dir,
            "false_positive_episode_montage",
        )

        correct_sequence = build_dir / "correct_sequence"
        correct_sequence.mkdir()
        for sequence_index, camera_frame in enumerate(correct_frames):
            compile_figure(
                correct_prediction_figure(
                    raw_frames[camera_frame],
                    camera_frame,
                    ground_truth_by_frame[camera_frame],
                    predictions_by_frame.get(camera_frame, []),
                    build_dir,
                ),
                correct_sequence / f"{sequence_index:06d}.png",
                build_dir,
                f"correct_prediction_{camera_frame}",
            )
        encode_sequence(
            correct_sequence,
            output_dir / "correct_predictions_over_dashed_ground_truth.mp4",
            20,
            FPS,
            len(correct_frames) * 3,
        )

        manifest = build_manifest(
            output_dir,
            examples,
            support_record,
            false_positive_frames,
            correct_frames,
        )
        (output_dir / "asset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        if args.keep_build:
            print(f"Kept build directory {build_dir}")
        else:
            temporary_context.cleanup()

    print(f"Wrote {len(ASSET_FILES)} assets and asset_manifest.json")


if __name__ == "__main__":
    main()
