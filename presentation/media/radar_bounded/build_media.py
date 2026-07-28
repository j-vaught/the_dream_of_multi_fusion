#!/usr/bin/env python3
"""Build deterministic, text-free presentation videos for radar-bounded detection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from PIL import Image, ImageDraw

ORIGINAL_SIZE = (5320, 3032)
FPS = 60
FRAME_COUNT = 300
FULL_SIZE = (1920, 1080)
ZOOM_SIZE = (1920, 540)
ZOOM_CROP = (2550, 1300, 3150, 1800)
IOU_THRESHOLD = 0.5

GARNET = "#73000A"
BLACK = "#000000"
WHITE = "#FFFFFF"
ATLANTIC = "#466A9F"
ROSE = "#CC2E40"
GRASS = "#CED318"
HONEYCOMB = "#A49137"

METHOD_ORDER = ("vision_only", "radar_confidence_gated", "radar_bounded")
METHOD_COLORS = {
    "vision_only": ATLANTIC,
    "radar_confidence_gated": HONEYCOMB,
    "radar_bounded": GARNET,
}


class AssetSpec(TypedDict):
    filename: str
    size: tuple[int, int]


ASSET_SPECS: dict[str, AssetSpec] = {
    "radar_inspection": {
        "filename": "01_radar_inspection_zones.mp4",
        "size": FULL_SIZE,
    },
    "full_crop": {
        "filename": "02_full_frame_extracted_crop_composite.mp4",
        "size": FULL_SIZE,
    },
    "false_positive": {
        "filename": "03_radar_bounded_false_positive_episodes.mp4",
        "size": FULL_SIZE,
    },
    "correct_prediction": {
        "filename": "04_radar_bounded_correct_predictions_over_ground_truth.mp4",
        "size": FULL_SIZE,
    },
    "full_methods": {
        "filename": "05_three_method_full_frame_composite.mp4",
        "size": FULL_SIZE,
    },
    "zoom_methods": {
        "filename": "06_three_method_zoom_composite.mp4",
        "size": ZOOM_SIZE,
    },
    "disagreement": {
        "filename": "07_method_disagreement_montage.mp4",
        "size": FULL_SIZE,
    },
}


@dataclass(frozen=True)
class Episode:
    """A fixed source-frame interval and its stable presentation crop."""

    center_frame: int
    start_frame: int
    end_frame: int
    score: int
    crop_xyxy: tuple[int, int, int, int]

    @property
    def camera_frames(self) -> list[int]:
        return list(range(self.start_frame, self.end_frame + 1))


@dataclass(frozen=True)
class MatchResult:
    """One-to-one prediction matching for a single frame."""

    predictions: tuple[dict[str, Any], ...]
    matched_truth_indices: frozenset[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-video", type=Path, default=Path("01_input_raw.mp4"))
    parser.add_argument("--rgb-dir", type=Path)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("datasets/dream_fusion_yolo/manifest.jsonl"),
    )
    parser.add_argument(
        "--radar-frames",
        type=Path,
        default=Path("data/frames.jsonl"),
    )
    parser.add_argument(
        "--vision-only",
        type=Path,
        default=Path("experiments/detection_comparison/vision_only_tiled.jsonl"),
    )
    parser.add_argument(
        "--radar-gated",
        type=Path,
        default=Path("experiments/detection_comparison/radar_confidence_gated.jsonl"),
    )
    parser.add_argument(
        "--radar-bounded",
        type=Path,
        default=Path("experiments/detection_comparison/radar_bounded_full.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("presentation/media/radar_bounded/assets"),
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="fast")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON Lines records in file order."""

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def records_by_camera(
    path: Path,
    *,
    detection_field: str | None = None,
) -> dict[int, Any]:
    """Index JSONL records or one detection field by camera frame."""

    records = read_jsonl(path)
    if detection_field is None:
        return {int(record["camera_frame"]): record for record in records}
    return {int(record["camera_frame"]): record.get(detection_field, []) for record in records}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using canonical compact serialization."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_label(label: str) -> str:
    """Map model vocabulary variants to the two evaluated classes."""

    lowered = label.lower().strip()
    if "boat" in lowered or "vessel" in lowered:
        return "boat"
    if "buoy" in lowered or lowered in {"green", "red"}:
        return "buoy"
    return lowered


def box_iou(
    first: Sequence[int | float],
    second: Sequence[int | float],
) -> float:
    """Compute intersection over union for two XYXY boxes."""

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_predictions(
    ground_truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    iou_threshold: float = IOU_THRESHOLD,
) -> MatchResult:
    """Match score-ordered predictions to same-class truth exactly once."""

    eligible = []
    for prediction in predictions:
        label = normalize_label(str(prediction.get("label", "")))
        if label in {"boat", "buoy"}:
            eligible.append({**prediction, "label": label})
    ordered = sorted(
        eligible,
        key=lambda item: (
            -float(item.get("score", 0.0)),
            item["label"],
            tuple(float(value) for value in item["bbox_xyxy"]),
        ),
    )
    unmatched_truth = set(range(len(ground_truth)))
    annotated = []
    matched_truth: set[int] = set()
    for prediction in ordered:
        candidates = [
            index
            for index in unmatched_truth
            if ground_truth[index]["class_name"] == prediction["label"]
        ]
        best_iou, best_index = max(
            (
                (
                    box_iou(
                        cast(Sequence[int | float], prediction["bbox_xyxy"]),
                        cast(Sequence[int | float], ground_truth[index]["bbox_xyxy"]),
                    ),
                    index,
                )
                for index in candidates
            ),
            default=(0.0, None),
        )
        is_match = best_index is not None and best_iou >= iou_threshold
        annotated.append(
            {
                **prediction,
                "_is_true_positive": is_match,
                "_matched_truth_index": best_index if is_match else None,
                "_iou": best_iou,
            }
        )
        if is_match:
            unmatched_truth.remove(best_index)
            matched_truth.add(best_index)
    return MatchResult(tuple(annotated), frozenset(matched_truth))


def primary_crop(record: dict[str, Any]) -> dict[str, Any] | None:
    """Select one radar crop by detections, area, then coordinates."""

    crops = record.get("crops", [])
    if not crops:
        return None

    def key(crop: dict[str, Any]) -> tuple[int, float, tuple[float, ...]]:
        box = crop["merged_crop_xyxy"]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        detections = len(crop.get("dino_detections_crop", []))
        return detections, area, tuple(-float(value) for value in box)

    return max(crops, key=key)


def clamp_crop(
    focus_box: Sequence[int | float],
    *,
    crop_size: tuple[int, int] = (1150, 650),
    image_size: tuple[int, int] = ORIGINAL_SIZE,
) -> tuple[int, int, int, int]:
    """Center a fixed crop on a box and clamp it to image bounds."""

    crop_width = min(crop_size[0], image_size[0])
    crop_height = min(crop_size[1], image_size[1])
    center_x = (float(focus_box[0]) + float(focus_box[2])) / 2
    center_y = (float(focus_box[1]) + float(focus_box[3])) / 2
    left = round(center_x - crop_width / 2)
    top = round(center_y - crop_height / 2)
    left = min(max(0, left), image_size[0] - crop_width)
    top = min(max(0, top), image_size[1] - crop_height)
    return left, top, left + crop_width, top + crop_height


def union_boxes(
    boxes: Sequence[Sequence[int | float]],
) -> tuple[float, float, float, float]:
    """Return the minimal box containing all supplied boxes."""

    if not boxes:
        raise ValueError("at least one box is required")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def select_episodes(
    scores: dict[int, int],
    focus_boxes: Mapping[int, Sequence[Sequence[int | float]]],
    *,
    first_frame: int,
    last_frame: int,
    episode_count: int = 3,
    episode_length: int = 20,
) -> list[Episode]:
    """Select high-scoring, non-overlapping intervals with deterministic ties."""

    if episode_count * episode_length > last_frame - first_frame + 1:
        raise ValueError("requested episodes do not fit in the frame range")
    ranked_frames = sorted(scores, key=lambda frame: (-scores[frame], frame))
    selected: list[Episode] = []
    for center in ranked_frames:
        if scores[center] <= 0 or not focus_boxes.get(center):
            continue
        start = center - episode_length // 2
        start = min(max(first_frame, start), last_frame - episode_length + 1)
        end = start + episode_length - 1
        if any(
            not (end < episode.start_frame or start > episode.end_frame) for episode in selected
        ):
            continue
        focus = union_boxes(focus_boxes[center])
        selected.append(
            Episode(
                center_frame=center,
                start_frame=start,
                end_frame=end,
                score=scores[center],
                crop_xyxy=clamp_crop(focus),
            )
        )
        if len(selected) == episode_count:
            break
    if len(selected) != episode_count:
        raise ValueError(f"could select only {len(selected)} of {episode_count} episodes")
    return sorted(selected, key=lambda episode: episode.start_frame)


def episode_output_selection(
    episodes: Sequence[Episode],
    *,
    repeat_each: int = 5,
) -> list[int]:
    """Expand three 20-frame episodes into a 300-frame presentation sequence."""

    selection = []
    for episode in episodes:
        for camera_frame in episode.camera_frames:
            selection.extend([camera_frame] * repeat_each)
    return selection


def disagreement_score(signatures: Sequence[tuple[int, int, int]]) -> int:
    """Score count disagreement among methods for TP, FP, and FN."""

    return sum(max(values) - min(values) for values in zip(*signatures, strict=True))


def select_separated_frames(
    scores: dict[int, int],
    *,
    count: int,
    minimum_separation: int,
) -> list[int]:
    """Select top-scoring frames while preserving temporal separation."""

    selected = []
    for frame in sorted(scores, key=lambda value: (-scores[value], value)):
        if scores[frame] <= 0:
            continue
        if all(abs(frame - existing) >= minimum_separation for existing in selected):
            selected.append(frame)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"could select only {len(selected)} of {count} frames")
    return sorted(selected)


def scale_box(
    box: Sequence[int | float],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Scale an XYXY box between image coordinate systems."""

    scale_x = to_size[0] / from_size[0]
    scale_y = to_size[1] / from_size[1]
    if len(box) != 4:
        raise ValueError(f"box must have four coordinates, got {len(box)}")
    scaled = tuple(
        round(float(value) * (scale_x if index % 2 == 0 else scale_y))
        for index, value in enumerate(box)
    )
    return scaled[0], scaled[1], scaled[2], scaled[3]


def draw_dashed_rectangle(
    drawing: ImageDraw.ImageDraw,
    box: Sequence[int | float],
    *,
    color: str,
    width: int,
    dash: int,
) -> None:
    """Draw a square-cornered dashed rectangle."""

    left, top, right, bottom = (round(value) for value in box)
    for start in range(left, right + 1, dash * 2):
        drawing.line((start, top, min(start + dash, right), top), fill=color, width=width)
        drawing.line((start, bottom, min(start + dash, right), bottom), fill=color, width=width)
    for start in range(top, bottom + 1, dash * 2):
        drawing.line((left, start, left, min(start + dash, bottom)), fill=color, width=width)
        drawing.line((right, start, right, min(start + dash, bottom)), fill=color, width=width)


def draw_inspection_zones(image: Image.Image, record: dict[str, Any]) -> None:
    """Draw raw radar support and merged detector inspection zones."""

    drawing = ImageDraw.Draw(image)
    for crop in record.get("crops", []):
        for radar_box in crop.get("radar_bboxes_xyxy", []):
            drawing.rectangle(
                scale_box(radar_box, ORIGINAL_SIZE, image.size),
                outline=ATLANTIC,
                width=3,
            )
        drawing.rectangle(
            scale_box(crop["merged_crop_xyxy"], ORIGINAL_SIZE, image.size),
            outline=GARNET,
            width=7,
        )


def draw_evaluation(
    image: Image.Image,
    truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    *,
    truth_mode: str = "all",
    prediction_mode: str = "all",
    line_width: int = 5,
) -> MatchResult:
    """Draw dashed truth and solid predictions without any text."""

    result = match_predictions(truth, predictions)
    drawing = ImageDraw.Draw(image)
    if truth_mode == "all":
        truth_indices: Iterable[int] = range(len(truth))
    elif truth_mode == "matched":
        truth_indices = sorted(result.matched_truth_indices)
    elif truth_mode == "none":
        truth_indices = ()
    else:
        raise ValueError(f"unknown truth mode {truth_mode!r}")
    for index in truth_indices:
        draw_dashed_rectangle(
            drawing,
            scale_box(truth[index]["bbox_xyxy"], ORIGINAL_SIZE, image.size),
            color=WHITE,
            width=line_width,
            dash=max(8, line_width * 3),
        )
    for prediction in result.predictions:
        is_true_positive = bool(prediction["_is_true_positive"])
        if prediction_mode == "true_positive" and not is_true_positive:
            continue
        if prediction_mode == "false_positive" and is_true_positive:
            continue
        if prediction_mode == "none":
            continue
        if prediction_mode not in {"all", "true_positive", "false_positive", "none"}:
            raise ValueError(f"unknown prediction mode {prediction_mode!r}")
        if is_true_positive:
            color = GARNET if prediction["label"] == "boat" else GRASS
        else:
            color = ROSE
        drawing.rectangle(
            scale_box(prediction["bbox_xyxy"], ORIGINAL_SIZE, image.size),
            outline=color,
            width=line_width,
        )
    return result


def clip_box_to_crop(
    box: Sequence[int | float],
    crop_xyxy: Sequence[int | float],
) -> tuple[float, float, float, float] | None:
    """Clip one source-coordinate box to a crop, or reject it if disjoint."""

    left = max(float(box[0]), float(crop_xyxy[0]))
    top = max(float(box[1]), float(crop_xyxy[1]))
    right = min(float(box[2]), float(crop_xyxy[2]))
    bottom = min(float(box[3]), float(crop_xyxy[3]))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def map_box_from_crop(
    box: Sequence[int | float],
    crop_xyxy: Sequence[int | float],
    output_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Map a clipped source-coordinate box into a resized crop panel."""

    clipped = clip_box_to_crop(box, crop_xyxy)
    if clipped is None:
        return None
    crop_width = float(crop_xyxy[2]) - float(crop_xyxy[0])
    crop_height = float(crop_xyxy[3]) - float(crop_xyxy[1])
    return (
        round((clipped[0] - float(crop_xyxy[0])) / crop_width * output_size[0]),
        round((clipped[1] - float(crop_xyxy[1])) / crop_height * output_size[1]),
        round((clipped[2] - float(crop_xyxy[0])) / crop_width * output_size[0]),
        round((clipped[3] - float(crop_xyxy[1])) / crop_height * output_size[1]),
    )


def draw_cropped_evaluation(
    image: Image.Image,
    truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    crop_xyxy: Sequence[int | float],
    *,
    truth_mode: str = "all",
    prediction_mode: str = "all",
    line_width: int = 5,
) -> MatchResult:
    """Draw evaluation overlays after a crop has reached display resolution."""

    result = match_predictions(truth, predictions)
    drawing = ImageDraw.Draw(image)
    if truth_mode == "all":
        truth_indices: Iterable[int] = range(len(truth))
    elif truth_mode == "matched":
        truth_indices = sorted(result.matched_truth_indices)
    elif truth_mode == "none":
        truth_indices = ()
    else:
        raise ValueError(f"unknown truth mode {truth_mode!r}")
    for index in truth_indices:
        mapped = map_box_from_crop(truth[index]["bbox_xyxy"], crop_xyxy, image.size)
        if mapped is not None:
            draw_dashed_rectangle(
                drawing,
                mapped,
                color=WHITE,
                width=line_width,
                dash=max(8, line_width * 3),
            )
    for prediction in result.predictions:
        is_true_positive = bool(prediction["_is_true_positive"])
        if prediction_mode == "true_positive" and not is_true_positive:
            continue
        if prediction_mode == "false_positive" and is_true_positive:
            continue
        if prediction_mode == "none":
            continue
        if prediction_mode not in {"all", "true_positive", "false_positive", "none"}:
            raise ValueError(f"unknown prediction mode {prediction_mode!r}")
        mapped = map_box_from_crop(prediction["bbox_xyxy"], crop_xyxy, image.size)
        if mapped is None:
            continue
        if is_true_positive:
            color = GARNET if prediction["label"] == "boat" else GRASS
        else:
            color = ROSE
        drawing.rectangle(mapped, outline=color, width=line_width)
    return result


def contain(
    image: Image.Image,
    size: tuple[int, int],
    *,
    background: str = BLACK,
) -> Image.Image:
    """Fit an image inside a fixed square-cornered canvas."""

    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, background)
    position = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    canvas.paste(resized, position)
    return canvas


def crop_original_coordinates(
    image: Image.Image,
    crop_xyxy: Sequence[int | float],
) -> Image.Image:
    """Crop a working image using full-resolution source coordinates."""

    return image.crop(scale_box(crop_xyxy, ORIGINAL_SIZE, image.size))


def render_radar_inspection(base: Image.Image, radar_record: dict[str, Any]) -> Image.Image:
    image = base.copy()
    draw_inspection_zones(image, radar_record)
    return contain(image, FULL_SIZE)


def render_full_crop(
    base: Image.Image,
    radar_record: dict[str, Any],
    truth: Sequence[dict[str, Any]],
    bounded_predictions: Sequence[dict[str, Any]],
) -> Image.Image:
    """Render one full frame beside its deterministically selected radar crop."""

    full = base.copy()
    draw_inspection_zones(full, radar_record)
    canvas = Image.new("RGB", FULL_SIZE, BLACK)
    left_panel = contain(full, (1264, 1048))
    canvas.paste(left_panel, (16, 16))
    crop = primary_crop(radar_record)
    if crop is not None:
        detailed = base.copy()
        draw_evaluation(
            detailed,
            truth,
            bounded_predictions,
            truth_mode="all",
            prediction_mode="all",
            line_width=5,
        )
        extracted = crop_original_coordinates(detailed, crop["merged_crop_xyxy"])
        right_panel = contain(extracted, (608, 1048))
        canvas.paste(right_panel, (1296, 16))
        ImageDraw.Draw(canvas).rectangle((1296, 16, 1903, 1063), outline=GARNET, width=6)
    ImageDraw.Draw(canvas).rectangle((16, 16, 1279, 1063), outline=ATLANTIC, width=4)
    return canvas


def render_method_panel(
    base: Image.Image,
    truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    method: str,
    *,
    panel_size: tuple[int, int],
    crop_xyxy: Sequence[int | float] | None = None,
) -> Image.Image:
    if crop_xyxy is None:
        image = base.copy()
        draw_evaluation(image, truth, predictions, line_width=5)
        panel = contain(image, panel_size)
    else:
        panel = crop_original_coordinates(base, crop_xyxy).resize(
            panel_size,
            Image.Resampling.LANCZOS,
        )
        draw_cropped_evaluation(
            panel,
            truth,
            predictions,
            crop_xyxy,
            line_width=4,
        )
    ImageDraw.Draw(panel).rectangle(
        (0, 0, panel.width - 1, panel.height - 1),
        outline=METHOD_COLORS[method],
        width=6,
    )
    return panel


def render_three_method_full(
    base: Image.Image,
    truth: Sequence[dict[str, Any]],
    methods: dict[str, Sequence[dict[str, Any]]],
) -> Image.Image:
    canvas = Image.new("RGB", FULL_SIZE, BLACK)
    for index, method in enumerate(METHOD_ORDER):
        panel = render_method_panel(
            base,
            truth,
            methods[method],
            method,
            panel_size=(624, 1048),
        )
        canvas.paste(panel, (16 + index * 640, 16))
    return canvas


def render_three_method_zoom(
    base: Image.Image,
    truth: Sequence[dict[str, Any]],
    methods: dict[str, Sequence[dict[str, Any]]],
) -> Image.Image:
    canvas = Image.new("RGB", ZOOM_SIZE, BLACK)
    for index, method in enumerate(METHOD_ORDER):
        panel = render_method_panel(
            base,
            truth,
            methods[method],
            method,
            panel_size=(632, 532),
            crop_xyxy=ZOOM_CROP,
        )
        canvas.paste(panel, (4 + index * 640, 4))
    return canvas


def render_event_frame(
    base: Image.Image,
    truth: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    crop_xyxy: Sequence[int | float],
    *,
    event_kind: str,
) -> Image.Image:
    rendered = crop_original_coordinates(base, crop_xyxy).resize(
        FULL_SIZE,
        Image.Resampling.LANCZOS,
    )
    if event_kind == "false_positive":
        draw_cropped_evaluation(
            rendered,
            truth,
            predictions,
            crop_xyxy,
            truth_mode="all",
            prediction_mode="false_positive",
            line_width=5,
        )
        border_color = ROSE
    elif event_kind == "correct_prediction":
        draw_cropped_evaluation(
            rendered,
            truth,
            predictions,
            crop_xyxy,
            truth_mode="matched",
            prediction_mode="true_positive",
            line_width=5,
        )
        border_color = GRASS
    else:
        raise ValueError(f"unknown event kind {event_kind!r}")
    ImageDraw.Draw(rendered).rectangle(
        (4, 4, rendered.width - 5, rendered.height - 5),
        outline=border_color,
        width=8,
    )
    return rendered


def disagreement_focus(
    truth: Sequence[dict[str, Any]],
    method_results: dict[str, MatchResult],
) -> tuple[int, int, int, int]:
    """Choose the most explanatory local region for a method disagreement."""

    bounded_false_positives = [
        prediction["bbox_xyxy"]
        for prediction in method_results["radar_bounded"].predictions
        if not prediction["_is_true_positive"]
    ]
    if bounded_false_positives:
        return clamp_crop(bounded_false_positives[0], crop_size=(500, 840))
    for truth_index, truth_object in enumerate(truth):
        matched_count = sum(
            truth_index in result.matched_truth_indices for result in method_results.values()
        )
        if 0 < matched_count < len(method_results):
            return clamp_crop(truth_object["bbox_xyxy"], crop_size=(500, 840))
    boxes = [
        prediction["bbox_xyxy"]
        for result in method_results.values()
        for prediction in result.predictions
    ]
    if boxes:
        return clamp_crop(boxes[0], crop_size=(500, 840))
    return clamp_crop(ZOOM_CROP, crop_size=(500, 840))


def render_disagreement_tile(
    base: Image.Image,
    truth: Sequence[dict[str, Any]],
    methods: dict[str, Sequence[dict[str, Any]]],
    focus_crop: Sequence[int | float],
) -> Image.Image:
    tile = Image.new("RGB", (960, 540), BLACK)
    for index, method in enumerate(METHOD_ORDER):
        panel = render_method_panel(
            base,
            truth,
            methods[method],
            method,
            panel_size=(312, 524),
            crop_xyxy=focus_crop,
        )
        tile.paste(panel, (8 + index * 320, 8))
    return tile


def render_disagreement_montage(tiles: Sequence[Image.Image]) -> Image.Image:
    if len(tiles) != 4:
        raise ValueError("disagreement montage requires exactly four tiles")
    canvas = Image.new("RGB", FULL_SIZE, BLACK)
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 2) * 960, (index // 2) * 540))
    return canvas


def probe_video(path: Path) -> dict[str, Any]:
    """Return compact ffprobe metadata for the encoded presentation asset."""

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


class FrameEncoder:
    """Stream RGB frames to a deterministic H.264 MP4 encoder."""

    def __init__(
        self,
        output: Path,
        size: tuple[int, int],
        *,
        crf: int,
        preset: str,
    ) -> None:
        self.output = output
        self.size = size
        self.frame_hashes: list[str] = []
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{size[0]}x{size[1]}",
            "-framerate",
            str(FPS),
            "-i",
            "-",
            "-frames:v",
            str(FRAME_COUNT),
            "-map_metadata",
            "-1",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, image: Image.Image) -> None:
        if image.size != self.size:
            raise ValueError(f"frame is {image.size}, expected {self.size}")
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        frame_bytes = rgb.tobytes()
        self.frame_hashes.append(hashlib.sha256(frame_bytes).hexdigest())
        if self.process.stdin is None:
            raise RuntimeError("encoder input is closed")
        self.process.stdin.write(frame_bytes)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, self.process.args)
        if len(self.frame_hashes) != FRAME_COUNT:
            raise ValueError(
                f"{self.output.name} received {len(self.frame_hashes)} frames, "
                f"expected {FRAME_COUNT}"
            )


def video_frame_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)


def iter_video_frames(path: Path) -> Iterator[Image.Image]:
    """Decode a clean source video to RGB frames in display order."""

    size = video_frame_size(path)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("decoder output is unavailable")
    frame_bytes = size[0] * size[1] * 3
    try:
        while True:
            data = process.stdout.read(frame_bytes)
            if not data:
                break
            if len(data) != frame_bytes:
                raise ValueError("source video ended with a partial RGB frame")
            yield Image.frombytes("RGB", size, data)
    finally:
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, process.args)


def iter_png_frames(rgb_dir: Path, camera_frames: Sequence[int]) -> Iterator[Image.Image]:
    """Read original clean PNG frames and normalize them to the local working size."""

    for camera_frame in camera_frames:
        with Image.open(rgb_dir / f"{camera_frame}_rgb.png") as source:
            image = source.convert("RGB")
        if image.size != ORIGINAL_SIZE:
            raise ValueError(f"{camera_frame}_rgb.png has unexpected size {image.size}")
        yield image.resize((1920, 1094), Image.Resampling.LANCZOS)


def frame_source(
    clean_video: Path,
    rgb_dir: Path | None,
    camera_frames: Sequence[int],
) -> Iterator[Image.Image]:
    if rgb_dir is not None:
        return iter_png_frames(rgb_dir, camera_frames)
    return iter_video_frames(clean_video)


def event_data(
    ground_truth: dict[int, Sequence[dict[str, Any]]],
    predictions: dict[int, Sequence[dict[str, Any]]],
) -> tuple[
    dict[int, MatchResult],
    dict[int, int],
    dict[int, list[Sequence[int | float]]],
    dict[int, int],
    dict[int, list[Sequence[int | float]]],
]:
    """Compute matching plus focus candidates for FP and TP episode selection."""

    results = {}
    fp_scores = {}
    fp_boxes = {}
    tp_scores = {}
    tp_boxes = {}
    for camera_frame in sorted(ground_truth):
        result = match_predictions(ground_truth[camera_frame], predictions.get(camera_frame, []))
        results[camera_frame] = result
        false_positives = [
            prediction["bbox_xyxy"]
            for prediction in result.predictions
            if not prediction["_is_true_positive"]
        ]
        true_positives = [
            prediction["bbox_xyxy"]
            for prediction in result.predictions
            if prediction["_is_true_positive"]
        ]
        fp_scores[camera_frame] = len(false_positives)
        fp_boxes[camera_frame] = false_positives
        tp_scores[camera_frame] = len(true_positives)
        tp_boxes[camera_frame] = true_positives
    return results, fp_scores, fp_boxes, tp_scores, tp_boxes


def build_manifest(
    *,
    asset_key: str,
    output: Path,
    encoder: FrameEncoder,
    selection: Any,
    source_files: dict[str, dict[str, str]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Create and write one self-contained reproducibility manifest."""

    spec = ASSET_SPECS[asset_key]
    manifest = {
        "schema_version": 1,
        "asset": asset_key,
        "file": output.name,
        "sha256": sha256_file(output),
        "video": probe_video(output),
        "encoding": {
            "fps": FPS,
            "frame_count": FRAME_COUNT,
            "duration_seconds": FRAME_COUNT / FPS,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "audio": False,
            "burned_in_text": False,
            "width": spec["size"][0],
            "height": spec["size"][1],
        },
        "selection": selection,
        "selection_sha256": canonical_sha256(selection),
        "rendered_frame_sha256": encoder.frame_hashes,
        "rendered_frame_hashes_sha256": canonical_sha256(encoder.frame_hashes),
        "source_files": source_files,
        "method_order": list(METHOD_ORDER),
        "method_border_colors": METHOD_COLORS,
        "overlay_colors": {
            "ground_truth_dashed": WHITE,
            "true_positive_boat": GARNET,
            "true_positive_buoy": GRASS,
            "false_positive": ROSE,
            "radar_support": ATLANTIC,
            "radar_inspection_zone": GARNET,
        },
        **extra,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def source_file_manifest(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def summarize_episode(episode: Episode) -> dict[str, Any]:
    return {
        "center_frame": episode.center_frame,
        "start_frame": episode.start_frame,
        "end_frame": episode.end_frame,
        "score": episode.score,
        "crop_xyxy": list(episode.crop_xyxy),
    }


def validate_inputs(
    camera_frames: Sequence[int],
    mappings: dict[str, dict[int, Any]],
) -> None:
    if len(camera_frames) != FRAME_COUNT:
        raise ValueError(f"ground truth has {len(camera_frames)} frames, expected {FRAME_COUNT}")
    if camera_frames != list(range(camera_frames[0], camera_frames[0] + FRAME_COUNT)):
        raise ValueError("camera frames must be one contiguous 300-frame sequence")
    expected = set(camera_frames)
    for name, mapping in mappings.items():
        if set(mapping) != expected:
            missing = sorted(expected - set(mapping))
            extra = sorted(set(mapping) - expected)
            raise ValueError(f"{name} frame mismatch, missing={missing[:5]}, extra={extra[:5]}")


def build_assets(args: argparse.Namespace) -> None:
    """Build all seven videos and their deterministic manifests."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_records = records_by_camera(args.ground_truth)
    ground_truth = {frame: record["objects"] for frame, record in ground_truth_records.items()}
    radar_records = records_by_camera(args.radar_bounded)
    radar_frames = records_by_camera(args.radar_frames)
    vision = records_by_camera(args.vision_only, detection_field="detections")
    gated = records_by_camera(args.radar_gated, detection_field="detections")
    bounded = records_by_camera(args.radar_bounded, detection_field="remapped_deduped")
    camera_frames = sorted(ground_truth)
    validate_inputs(
        camera_frames,
        {
            "radar frames": radar_frames,
            "radar bounded": radar_records,
            "vision only": vision,
            "radar gated": gated,
        },
    )
    bounded_results, fp_scores, fp_boxes, tp_scores, tp_boxes = event_data(
        ground_truth,
        bounded,
    )
    first_frame, last_frame = camera_frames[0], camera_frames[-1]
    fp_episodes = select_episodes(
        fp_scores,
        fp_boxes,
        first_frame=first_frame,
        last_frame=last_frame,
    )
    tp_episodes = select_episodes(
        tp_scores,
        tp_boxes,
        first_frame=first_frame,
        last_frame=last_frame,
    )
    fp_selection = episode_output_selection(fp_episodes)
    tp_selection = episode_output_selection(tp_episodes)
    if len(fp_selection) != FRAME_COUNT or len(tp_selection) != FRAME_COUNT:
        raise ValueError("episode selections must expand to exactly 300 frames")
    fp_episode_by_frame = {
        camera_frame: episode for episode in fp_episodes for camera_frame in episode.camera_frames
    }
    tp_episode_by_frame = {
        camera_frame: episode for episode in tp_episodes for camera_frame in episode.camera_frames
    }

    all_results: dict[int, dict[str, MatchResult]] = {}
    disagreement_scores = {}
    disagreement_crops = {}
    for camera_frame in camera_frames:
        results = {
            "vision_only": match_predictions(ground_truth[camera_frame], vision[camera_frame]),
            "radar_confidence_gated": match_predictions(
                ground_truth[camera_frame], gated[camera_frame]
            ),
            "radar_bounded": bounded_results[camera_frame],
        }
        all_results[camera_frame] = results
        signatures = [
            (
                len(result.matched_truth_indices),
                sum(not item["_is_true_positive"] for item in result.predictions),
                len(ground_truth[camera_frame]) - len(result.matched_truth_indices),
            )
            for result in results.values()
        ]
        disagreement_scores[camera_frame] = disagreement_score(signatures)
        disagreement_crops[camera_frame] = disagreement_focus(ground_truth[camera_frame], results)
    disagreement_frames = select_separated_frames(
        disagreement_scores,
        count=4,
        minimum_separation=20,
    )

    source_paths = {
        "clean_video": args.clean_video,
        "ground_truth": args.ground_truth,
        "radar_frames": args.radar_frames,
        "vision_only": args.vision_only,
        "radar_confidence_gated": args.radar_gated,
        "radar_bounded": args.radar_bounded,
    }
    if args.rgb_dir is not None:
        source_paths.pop("clean_video")
    source_files = source_file_manifest(source_paths)

    encoders = {
        key: FrameEncoder(
            args.output_dir / spec["filename"],
            spec["size"],
            crf=args.crf,
            preset=args.preset,
        )
        for key, spec in ASSET_SPECS.items()
    }
    captured_disagreement: dict[int, Image.Image] = {}
    try:
        source_iterator = frame_source(args.clean_video, args.rgb_dir, camera_frames)
        decoded_count = 0
        for camera_frame, base in zip(camera_frames, source_iterator, strict=True):
            decoded_count += 1
            truth = ground_truth[camera_frame]
            methods = {
                "vision_only": vision[camera_frame],
                "radar_confidence_gated": gated[camera_frame],
                "radar_bounded": bounded[camera_frame],
            }
            encoders["radar_inspection"].write(
                render_radar_inspection(base, radar_records[camera_frame])
            )
            encoders["full_crop"].write(
                render_full_crop(
                    base,
                    radar_records[camera_frame],
                    truth,
                    bounded[camera_frame],
                )
            )
            encoders["full_methods"].write(render_three_method_full(base, truth, methods))
            encoders["zoom_methods"].write(render_three_method_zoom(base, truth, methods))
            if camera_frame in fp_episode_by_frame:
                rendered = render_event_frame(
                    base,
                    truth,
                    bounded[camera_frame],
                    fp_episode_by_frame[camera_frame].crop_xyxy,
                    event_kind="false_positive",
                )
                for _ in range(5):
                    encoders["false_positive"].write(rendered)
            if camera_frame in tp_episode_by_frame:
                rendered = render_event_frame(
                    base,
                    truth,
                    bounded[camera_frame],
                    tp_episode_by_frame[camera_frame].crop_xyxy,
                    event_kind="correct_prediction",
                )
                for _ in range(5):
                    encoders["correct_prediction"].write(rendered)
            if camera_frame in disagreement_frames:
                captured_disagreement[camera_frame] = base.copy()
            if decoded_count % 25 == 0 or decoded_count == FRAME_COUNT:
                print(f"Rendered {decoded_count}/{FRAME_COUNT} source frames", flush=True)
        if decoded_count != FRAME_COUNT:
            raise ValueError(f"decoded {decoded_count} clean frames, expected {FRAME_COUNT}")

        tiles = []
        for camera_frame in disagreement_frames:
            methods = {
                "vision_only": vision[camera_frame],
                "radar_confidence_gated": gated[camera_frame],
                "radar_bounded": bounded[camera_frame],
            }
            tiles.append(
                render_disagreement_tile(
                    captured_disagreement[camera_frame],
                    ground_truth[camera_frame],
                    methods,
                    disagreement_crops[camera_frame],
                )
            )
        montage = render_disagreement_montage(tiles)
        for _ in range(FRAME_COUNT):
            encoders["disagreement"].write(montage)
    finally:
        close_errors = []
        for encoder in encoders.values():
            try:
                encoder.close()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                close_errors.append(error)
        if close_errors:
            raise close_errors[0]

    selections = {
        "radar_inspection": {
            "kind": "complete_sequence",
            "camera_frames": camera_frames,
        },
        "full_crop": {
            "kind": "complete_sequence",
            "camera_frames": camera_frames,
            "primary_crop_rule": "most detections, then area, then top-left coordinates",
        },
        "false_positive": {
            "kind": "three_episodes",
            "episodes": [summarize_episode(episode) for episode in fp_episodes],
            "output_camera_frames": fp_selection,
            "source_frame_repeat": 5,
        },
        "correct_prediction": {
            "kind": "three_episodes",
            "episodes": [summarize_episode(episode) for episode in tp_episodes],
            "output_camera_frames": tp_selection,
            "source_frame_repeat": 5,
        },
        "full_methods": {
            "kind": "complete_sequence",
            "camera_frames": camera_frames,
        },
        "zoom_methods": {
            "kind": "complete_sequence",
            "camera_frames": camera_frames,
            "crop_xyxy": list(ZOOM_CROP),
        },
        "disagreement": {
            "kind": "static_four_frame_montage",
            "camera_frames": disagreement_frames,
            "scores": {str(frame): disagreement_scores[frame] for frame in disagreement_frames},
            "crop_xyxy": {
                str(frame): list(disagreement_crops[frame]) for frame in disagreement_frames
            },
            "output_repetitions": FRAME_COUNT,
        },
    }
    manifests = {}
    for key, encoder in encoders.items():
        output = args.output_dir / ASSET_SPECS[key]["filename"]
        manifests[key] = build_manifest(
            asset_key=key,
            output=output,
            encoder=encoder,
            selection=selections[key],
            source_files=source_files,
            extra={
                "layout": {
                    "full_crop": "full frame left, selected crop right",
                    "full_methods": "vision, radar-gated, radar-bounded left to right",
                    "zoom_methods": "vision, radar-gated, radar-bounded left to right",
                    "disagreement": "four disagreement frames in reading order; "
                    "each tile uses method order left to right",
                }.get(key, "single composite stream"),
            },
        )
    index = {
        "schema_version": 1,
        "assets": {
            key: {
                "file": manifest["file"],
                "sha256": manifest["sha256"],
                "manifest": Path(manifest["file"]).with_suffix(".manifest.json").name,
                "manifest_sha256": sha256_file(
                    args.output_dir / Path(manifest["file"]).with_suffix(".manifest.json").name
                ),
            }
            for key, manifest in manifests.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifests)} videos and manifests to {args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    build_assets(args)


if __name__ == "__main__":
    main()
