#!/usr/bin/env python3
"""Mine and render deterministic object-detection error evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw

Box = Sequence[float]
JsonObject = dict[str, Any]

IOU_THRESHOLD = 0.5
MAX_PER_CATEGORY = 3
OUTPUT_SIZE = (1600, 1000)
MIN_SOURCE_CROP = (600.0, 375.0)
FRAME_SEPARATION = 5

FP_COLOR = "#FF0000"
FN_COLOR = "#FF00FF"
CORRECT_COLORS = {
    "boat": "#FF00FF",
    "buoy": "#FFFF00",
}

CATEGORY_ORDER = (
    "tiny_misses",
    "localization_below_0_5",
    "duplicate_hypotheses",
    "radar_supported_clutter_shoreline",
    "ambiguous_omitted_objects",
    "method_specific_recovery_disagreement",
)

METHOD_ORDER = {
    "vision_only_tiled": 0,
    "radar_confidence_gated": 1,
    "radar_bounded_crops": 2,
    "radar_bounded_raw": 3,
}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "presentation" / "errors" / "evidence"
DEFAULT_REMOTE_ROOT = "/home/j-vaught/dream_fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Manifest JSONL or a .tar.gz containing manifest.jsonl.",
    )
    parser.add_argument(
        "--vision-only",
        type=Path,
        default=ROOT / "experiments" / "detection_comparison" / "vision_only_tiled.jsonl",
    )
    parser.add_argument(
        "--radar-gated",
        type=Path,
        default=ROOT / "experiments" / "detection_comparison" / "radar_confidence_gated.jsonl",
    )
    parser.add_argument(
        "--radar-bounded",
        type=Path,
        default=None,
        help="Recovered radar-bounded JSONL. A narrow remote fetch is used if absent.",
    )
    parser.add_argument("--rgb-dir", type=Path, default=ROOT / "data" / "rgb_out")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--remote-host", default="comech-2422")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--max-per-category", type=int, default=MAX_PER_CATEGORY)
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_bytes(payload: bytes) -> list[JsonObject]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]


def read_jsonl(path: Path) -> list[JsonObject]:
    return read_jsonl_bytes(path.read_bytes())


def read_ground_truth(path: Path) -> tuple[list[JsonObject], str]:
    if path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as archive:
            member = archive.getmember("manifest.jsonl")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read manifest.jsonl from {path}")
            payload = extracted.read()
        return read_jsonl_bytes(payload), sha256_bytes(payload)
    payload = path.read_bytes()
    return read_jsonl_bytes(payload), sha256_bytes(payload)


def resolve_ground_truth(requested: Path | None) -> Path:
    candidates = [
        requested,
        ROOT / "out" / "datasets" / "dream_fusion_yolo" / "manifest.jsonl",
        ROOT / "dream_fusion_yolo_labels_and_manifest.tar.gz",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "ground truth was not found; pass --ground-truth as JSONL or a tar.gz archive"
    )


def atomic_remote_copy(host: str, remote_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        subprocess.run(
            ["scp", "-q", f"{host}:{remote_path}", str(temporary_path)],
            check=True,
        )
        if temporary_path.stat().st_size == 0:
            raise ValueError(f"remote file was empty: {host}:{remote_path}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def resolve_radar_bounded(
    requested: Path | None,
    output_dir: Path,
    remote_host: str,
    remote_root: str,
) -> Path:
    recovered = ROOT / "experiments" / "detection_comparison" / "radar_bounded_full.jsonl"
    candidates = [requested, recovered]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    cached = output_dir / "cache" / "radar_bounded_full.jsonl"
    if not cached.is_file():
        atomic_remote_copy(
            remote_host,
            f"{remote_root}/out/detections.jsonl",
            cached,
        )
    return cached


def normalize_class(label: object) -> str:
    value = str(label).strip().lower()
    if value in {"boat", "vessel", "ship"}:
        return "boat"
    if value in {"buoy", "navigation buoy", "red buoy", "green buoy", "red", "green"}:
        return "buoy"
    if "buoy" in value or value.endswith("oy"):
        return "buoy"
    return value


def box_area(box: Box) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_iou(first: Box, second: Box) -> float:
    intersection_width = max(
        0.0,
        min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])),
    )
    intersection_height = max(
        0.0,
        min(float(first[3]), float(second[3])) - max(float(first[1]), float(second[1])),
    )
    intersection = intersection_width * intersection_height
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def rounded_box(box: Box) -> list[float]:
    return [round(float(value), 6) for value in box]


def prediction_sort_key(item: tuple[int, JsonObject]) -> tuple[Any, ...]:
    index, prediction = item
    return (
        -float(prediction.get("score", 0.0)),
        normalize_class(prediction.get("label", "")),
        tuple(float(value) for value in prediction["bbox_xyxy"]),
        index,
    )


def match_detections(
    ground_truth: Sequence[JsonObject],
    predictions: Sequence[JsonObject],
    iou_threshold: float = IOU_THRESHOLD,
) -> JsonObject:
    """Greedily class-match scored predictions to truth with deterministic ties."""
    unmatched_truth = set(range(len(ground_truth)))
    true_positives: list[JsonObject] = []
    false_positives: list[JsonObject] = []
    normalized = [
        {
            **prediction,
            "label": normalize_class(prediction.get("label", "")),
        }
        for prediction in predictions
        if normalize_class(prediction.get("label", "")) in {"boat", "buoy"}
    ]
    for prediction_index, prediction in sorted(
        enumerate(normalized),
        key=prediction_sort_key,
    ):
        prediction_box = cast(Box, prediction["bbox_xyxy"])
        same_class_all = [
            truth_index
            for truth_index, truth in enumerate(ground_truth)
            if normalize_class(truth.get("class_name", "")) == prediction["label"]
        ]
        same_class_unmatched = [
            truth_index for truth_index in same_class_all if truth_index in unmatched_truth
        ]
        best_iou, best_truth_index = max(
            (
                (
                    box_iou(
                        prediction_box,
                        ground_truth[truth_index]["bbox_xyxy"],
                    ),
                    -truth_index,
                    truth_index,
                )
                for truth_index in same_class_unmatched
            ),
            default=(0.0, 0, None),
        )[::2]
        nearest_iou, nearest_truth_index = max(
            (
                (
                    box_iou(
                        prediction_box,
                        ground_truth[truth_index]["bbox_xyxy"],
                    ),
                    -truth_index,
                    truth_index,
                )
                for truth_index in same_class_all
            ),
            default=(0.0, 0, None),
        )[::2]
        result = {
            "prediction_index": prediction_index,
            "prediction": prediction,
            "iou": best_iou,
            "matched_truth_index": best_truth_index,
            "nearest_same_class_iou": nearest_iou,
            "nearest_same_class_truth_index": nearest_truth_index,
        }
        if best_truth_index is not None and best_iou >= iou_threshold:
            unmatched_truth.remove(best_truth_index)
            true_positives.append(result)
        else:
            result["matched_truth_index"] = None
            result["iou"] = 0.0
            false_positives.append(result)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": [
            {
                "truth_index": index,
                "truth": ground_truth[index],
            }
            for index in sorted(unmatched_truth)
        ],
    }


def records_by_frame(
    records: Iterable[JsonObject],
    field: str,
) -> dict[int, list[JsonObject]]:
    return {int(record["camera_frame"]): list(record.get(field, [])) for record in records}


def raw_radar_predictions(record: JsonObject) -> list[JsonObject]:
    predictions: list[JsonObject] = []
    for crop_index, crop in enumerate(record.get("crops", [])):
        for prediction in crop.get("dino_detections_remapped", []):
            predictions.append(
                {
                    **prediction,
                    "crop_index": crop_index,
                    "source_tracks": list(
                        prediction.get("source_tracks", crop.get("source_tracks", []))
                    ),
                }
            )
    return predictions


def annotation(
    kind: str,
    box: Box,
    class_name: str,
    method: str,
    *,
    score: float | None = None,
    iou: float | None = None,
) -> JsonObject:
    result: JsonObject = {
        "kind": kind,
        "bbox_xyxy": rounded_box(box),
        "class_name": class_name,
        "method": method,
    }
    if score is not None:
        result["score"] = round(float(score), 6)
    if iou is not None:
        result["iou"] = round(float(iou), 6)
    return result


def candidate(
    category: str,
    frame: int,
    rank: tuple[Any, ...],
    rationale: str,
    annotations: list[JsonObject],
    focus_boxes: list[Box],
    primary_method: str,
    identity: str,
    details: JsonObject | None = None,
) -> JsonObject:
    return {
        "category": category,
        "camera_frame": frame,
        "_rank": rank,
        "selection_rationale": rationale,
        "annotations": annotations,
        "focus_boxes": [rounded_box(box) for box in focus_boxes],
        "primary_method": primary_method,
        "_identity": identity,
        "details": details or {},
    }


def index_matches(match_result: JsonObject) -> tuple[dict[int, JsonObject], set[int]]:
    matched = {int(item["matched_truth_index"]): item for item in match_result["true_positives"]}
    missed = {int(item["truth_index"]) for item in match_result["false_negatives"]}
    return matched, missed


def mine_candidates(
    ground_truth_by_frame: dict[int, list[JsonObject]],
    predictions: dict[str, dict[int, list[JsonObject]]],
    radar_records: dict[int, JsonObject],
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, list[JsonObject]]:
    matches: dict[str, dict[int, JsonObject]] = {
        method: {
            frame: match_detections(
                truth,
                method_predictions.get(frame, []),
                iou_threshold,
            )
            for frame, truth in ground_truth_by_frame.items()
        }
        for method, method_predictions in predictions.items()
    }
    candidates: dict[str, list[JsonObject]] = {category: [] for category in CATEGORY_ORDER}
    final_methods = tuple(predictions)

    for frame in sorted(ground_truth_by_frame):
        truth = ground_truth_by_frame[frame]
        method_indexes = {method: index_matches(matches[method][frame]) for method in final_methods}

        for truth_index, truth_object in enumerate(truth):
            missed_by = [
                method for method in final_methods if truth_index in method_indexes[method][1]
            ]
            if len(missed_by) == len(final_methods):
                area = box_area(truth_object["bbox_xyxy"])
                class_name = normalize_class(truth_object["class_name"])
                candidates["tiny_misses"].append(
                    candidate(
                        "tiny_misses",
                        frame,
                        (area, frame, truth_index),
                        "Smallest ground-truth instance missed by every final method.",
                        [
                            annotation(
                                "false_negative",
                                truth_object["bbox_xyxy"],
                                class_name,
                                "all_final_methods",
                            )
                        ],
                        [truth_object["bbox_xyxy"]],
                        "all_final_methods",
                        str(truth_object.get("track_id", truth_index)),
                        {
                            "ground_truth_index": truth_index,
                            "track_id": truth_object.get("track_id"),
                            "area_px2": round(area, 6),
                            "missed_by": missed_by,
                        },
                    )
                )

            recovered_by = [
                method for method in final_methods if truth_index in method_indexes[method][0]
            ]
            if recovered_by and missed_by:
                preferred_recovery = sorted(
                    recovered_by,
                    key=lambda method: (
                        method == "vision_only_tiled",
                        METHOD_ORDER[method],
                    ),
                )[0]
                matched_prediction = method_indexes[preferred_recovery][0][truth_index]
                prediction = matched_prediction["prediction"]
                class_name = normalize_class(truth_object["class_name"])
                candidates["method_specific_recovery_disagreement"].append(
                    candidate(
                        "method_specific_recovery_disagreement",
                        frame,
                        (
                            -len(recovered_by),
                            METHOD_ORDER[preferred_recovery],
                            box_area(truth_object["bbox_xyxy"]),
                            frame,
                            truth_index,
                        ),
                        "One final method class-matches this object while another misses it.",
                        [
                            annotation(
                                "true_positive",
                                prediction["bbox_xyxy"],
                                class_name,
                                preferred_recovery,
                                score=float(prediction.get("score", 0.0)),
                                iou=float(matched_prediction["iou"]),
                            )
                        ],
                        [truth_object["bbox_xyxy"], prediction["bbox_xyxy"]],
                        preferred_recovery,
                        str(truth_object.get("track_id", truth_index)),
                        {
                            "ground_truth_index": truth_index,
                            "track_id": truth_object.get("track_id"),
                            "recovered_by": recovered_by,
                            "missed_by": missed_by,
                            "matched_iou": round(float(matched_prediction["iou"]), 6),
                        },
                    )
                )

        for method in final_methods:
            for false_positive in matches[method][frame]["false_positives"]:
                nearest_iou = float(false_positive["nearest_same_class_iou"])
                nearest_index = false_positive["nearest_same_class_truth_index"]
                if nearest_index is None or not 0.0 < nearest_iou < iou_threshold:
                    continue
                nearest_truth = truth[int(nearest_index)]
                prediction = false_positive["prediction"]
                class_name = normalize_class(prediction["label"])
                candidates["localization_below_0_5"].append(
                    candidate(
                        "localization_below_0_5",
                        frame,
                        (
                            -nearest_iou,
                            METHOD_ORDER[method],
                            -float(prediction.get("score", 0.0)),
                            frame,
                        ),
                        "Highest same-class overlap that remains below the 0.5 match threshold.",
                        [
                            annotation(
                                "false_positive",
                                prediction["bbox_xyxy"],
                                class_name,
                                method,
                                score=float(prediction.get("score", 0.0)),
                                iou=nearest_iou,
                            ),
                            annotation(
                                "false_negative",
                                nearest_truth["bbox_xyxy"],
                                class_name,
                                method,
                                iou=nearest_iou,
                            ),
                        ],
                        [prediction["bbox_xyxy"], nearest_truth["bbox_xyxy"]],
                        method,
                        f"{method}:{nearest_truth.get('track_id', nearest_index)}",
                        {
                            "ground_truth_index": nearest_index,
                            "track_id": nearest_truth.get("track_id"),
                            "subthreshold_iou": round(nearest_iou, 6),
                        },
                    )
                )

        raw_predictions = raw_radar_predictions(radar_records.get(frame, {}))
        raw_by_truth: dict[int, list[tuple[float, JsonObject]]] = defaultdict(list)
        for prediction in raw_predictions:
            label = normalize_class(prediction.get("label", ""))
            for truth_index, truth_object in enumerate(truth):
                if normalize_class(truth_object["class_name"]) != label:
                    continue
                overlap = box_iou(prediction["bbox_xyxy"], truth_object["bbox_xyxy"])
                if overlap >= iou_threshold:
                    raw_by_truth[truth_index].append((overlap, prediction))
        for truth_index, hypotheses in raw_by_truth.items():
            if len(hypotheses) < 2:
                continue
            hypotheses.sort(
                key=lambda item: (
                    -float(item[1].get("score", 0.0)),
                    -item[0],
                    tuple(float(value) for value in item[1]["bbox_xyxy"]),
                )
            )
            truth_object = truth[truth_index]
            class_name = normalize_class(truth_object["class_name"])
            duplicate_annotations = [
                annotation(
                    "true_positive" if index == 0 else "duplicate_false_positive",
                    prediction["bbox_xyxy"],
                    class_name,
                    "radar_bounded_raw",
                    score=float(prediction.get("score", 0.0)),
                    iou=overlap,
                )
                for index, (overlap, prediction) in enumerate(hypotheses)
            ]
            candidates["duplicate_hypotheses"].append(
                candidate(
                    "duplicate_hypotheses",
                    frame,
                    (
                        -len(hypotheses),
                        -max(overlap for overlap, _ in hypotheses),
                        frame,
                        truth_index,
                    ),
                    "Multiple pre-NMS radar-crop hypotheses class-match the same object.",
                    duplicate_annotations,
                    [
                        truth_object["bbox_xyxy"],
                        *(prediction["bbox_xyxy"] for _, prediction in hypotheses),
                    ],
                    "radar_bounded_raw",
                    str(truth_object.get("track_id", truth_index)),
                    {
                        "ground_truth_index": truth_index,
                        "track_id": truth_object.get("track_id"),
                        "hypothesis_count": len(hypotheses),
                        "hypothesis_ious": [round(overlap, 6) for overlap, _ in hypotheses],
                    },
                )
            )

        radar_record = radar_records.get(frame, {})
        radar_status = {
            int(status["track_id"]): status for status in radar_record.get("radar_status", [])
        }
        for false_positive in matches["radar_bounded_crops"][frame]["false_positives"]:
            prediction = false_positive["prediction"]
            source_tracks = [int(value) for value in prediction.get("source_tracks", [])]
            support_boxes = [
                radar_status[track_id]["bbox_xyxy"]
                for track_id in source_tracks
                if track_id in radar_status
            ]
            if not support_boxes:
                continue
            aspects = [
                (float(box[2]) - float(box[0])) / max(1.0, float(box[3]) - float(box[1]))
                for box in support_boxes
            ]
            nearest_iou = float(false_positive["nearest_same_class_iou"])
            if nearest_iou > 0.1:
                continue
            class_name = normalize_class(prediction["label"])
            candidates["radar_supported_clutter_shoreline"].append(
                candidate(
                    "radar_supported_clutter_shoreline",
                    frame,
                    (
                        -max(aspects),
                        -float(prediction.get("score", 0.0)),
                        nearest_iou,
                        frame,
                    ),
                    (
                        "Radar-track-supported detection has no class match and is "
                        "ranked by support-region elongation."
                    ),
                    [
                        annotation(
                            "false_positive",
                            prediction["bbox_xyxy"],
                            class_name,
                            "radar_bounded_crops",
                            score=float(prediction.get("score", 0.0)),
                            iou=nearest_iou,
                        )
                    ],
                    [prediction["bbox_xyxy"], *support_boxes],
                    "radar_bounded_crops",
                    f"{class_name}:{','.join(map(str, source_tracks))}",
                    {
                        "source_tracks": source_tracks,
                        "radar_support_bboxes_xyxy": [rounded_box(box) for box in support_boxes],
                        "radar_support_aspect_ratios": [round(value, 6) for value in aspects],
                        "nearest_same_class_iou": round(nearest_iou, 6),
                    },
                )
            )

        whole_fps = matches["vision_only_tiled"][frame]["false_positives"]
        bounded_fps = matches["radar_bounded_crops"][frame]["false_positives"]
        for whole_fp in whole_fps:
            whole_prediction = whole_fp["prediction"]
            whole_label = normalize_class(whole_prediction["label"])
            for bounded_fp in bounded_fps:
                bounded_prediction = bounded_fp["prediction"]
                if normalize_class(bounded_prediction["label"]) != whole_label:
                    continue
                agreement_iou = box_iou(
                    whole_prediction["bbox_xyxy"],
                    bounded_prediction["bbox_xyxy"],
                )
                if agreement_iou < iou_threshold:
                    continue
                nearest_gt_iou = max(
                    float(whole_fp["nearest_same_class_iou"]),
                    float(bounded_fp["nearest_same_class_iou"]),
                )
                candidates["ambiguous_omitted_objects"].append(
                    candidate(
                        "ambiguous_omitted_objects",
                        frame,
                        (
                            -agreement_iou,
                            nearest_gt_iou,
                            -min(
                                float(whole_prediction.get("score", 0.0)),
                                float(bounded_prediction.get("score", 0.0)),
                            ),
                            frame,
                        ),
                        (
                            "Whole-frame and radar-bounded methods agree, but neither "
                            "hypothesis class-matches YOLO ground truth."
                        ),
                        [
                            annotation(
                                "false_positive",
                                whole_prediction["bbox_xyxy"],
                                whole_label,
                                "vision_only_tiled",
                                score=float(whole_prediction.get("score", 0.0)),
                                iou=nearest_gt_iou,
                            ),
                            annotation(
                                "false_positive",
                                bounded_prediction["bbox_xyxy"],
                                whole_label,
                                "radar_bounded_crops",
                                score=float(bounded_prediction.get("score", 0.0)),
                                iou=nearest_gt_iou,
                            ),
                        ],
                        [
                            whole_prediction["bbox_xyxy"],
                            bounded_prediction["bbox_xyxy"],
                        ],
                        "vision_only_tiled+radar_bounded_crops",
                        f"{whole_label}:{round(float(whole_prediction['bbox_xyxy'][0]), 1)}",
                        {
                            "cross_method_iou": round(agreement_iou, 6),
                            "nearest_ground_truth_iou": round(nearest_gt_iou, 6),
                        },
                    )
                )

    add_temporally_verified_omission_candidates(
        candidates,
        matches["radar_bounded_crops"],
    )
    return candidates


def add_temporally_verified_omission_candidates(
    candidates: dict[str, list[JsonObject]],
    bounded_matches: dict[int, JsonObject],
    minimum_frames: int = 3,
    lookahead: int = 4,
) -> None:
    """Add stable unmatched hypotheses as possible ground-truth omissions."""
    available_frames = set(bounded_matches)
    for frame in sorted(available_frames):
        for false_positive in bounded_matches[frame]["false_positives"]:
            if float(false_positive["nearest_same_class_iou"]) > 0.05:
                continue
            prediction = false_positive["prediction"]
            label = normalize_class(prediction["label"])
            base_box = prediction["bbox_xyxy"]
            support_frames = [frame]
            support_ious = [1.0]
            support_scores = [float(prediction.get("score", 0.0))]
            for offset in range(1, lookahead + 1):
                next_frame = frame + offset
                if next_frame not in available_frames:
                    break
                compatible = [
                    item
                    for item in bounded_matches[next_frame]["false_positives"]
                    if normalize_class(item["prediction"]["label"]) == label
                    and float(item["nearest_same_class_iou"]) <= 0.05
                ]
                best_overlap, best_item = max(
                    (
                        (
                            box_iou(base_box, item["prediction"]["bbox_xyxy"]),
                            item,
                        )
                        for item in compatible
                    ),
                    key=lambda value: (
                        value[0],
                        float(value[1]["prediction"].get("score", 0.0)),
                    ),
                    default=(0.0, None),
                )
                if best_item is None or best_overlap < IOU_THRESHOLD:
                    break
                support_frames.append(next_frame)
                support_ious.append(best_overlap)
                support_scores.append(float(best_item["prediction"].get("score", 0.0)))
            if len(support_frames) < minimum_frames:
                continue
            center_x = (float(base_box[0]) + float(base_box[2])) / 2.0
            center_y = (float(base_box[1]) + float(base_box[3])) / 2.0
            identity = f"{label}:{round(center_x / 100)}:{round(center_y / 100)}"
            candidates["ambiguous_omitted_objects"].append(
                candidate(
                    "ambiguous_omitted_objects",
                    frame,
                    (
                        -len(support_frames),
                        -sum(support_scores) / len(support_scores),
                        frame,
                    ),
                    (
                        "A YOLO-unmatched radar-bounded hypothesis persists with "
                        "same-class spatial agreement across consecutive frames."
                    ),
                    [
                        annotation(
                            "false_positive",
                            base_box,
                            label,
                            "radar_bounded_crops",
                            score=float(prediction.get("score", 0.0)),
                            iou=float(false_positive["nearest_same_class_iou"]),
                        )
                    ],
                    [base_box],
                    "radar_bounded_crops",
                    identity,
                    {
                        "temporal_support_frames": support_frames,
                        "temporal_support_ious": [round(value, 6) for value in support_ious],
                        "nearest_ground_truth_iou": round(
                            float(false_positive["nearest_same_class_iou"]),
                            6,
                        ),
                    },
                )
            )


def select_candidates(
    candidates: dict[str, list[JsonObject]],
    max_per_category: int = MAX_PER_CATEGORY,
    frame_separation: int = FRAME_SEPARATION,
) -> dict[str, list[JsonObject]]:
    if not 1 <= max_per_category <= MAX_PER_CATEGORY:
        raise ValueError(f"max_per_category must be between 1 and {MAX_PER_CATEGORY}")
    selected: dict[str, list[JsonObject]] = {}
    for category in CATEGORY_ORDER:
        ranked = sorted(candidates[category], key=lambda item: item["_rank"])
        chosen: list[JsonObject] = []
        used_identities: set[str] = set()
        for item in ranked:
            identity = str(item["_identity"])
            frame = int(item["camera_frame"])
            if identity in used_identities:
                continue
            if any(
                abs(frame - int(existing["camera_frame"])) < frame_separation for existing in chosen
            ):
                continue
            chosen.append(item)
            used_identities.add(identity)
            if len(chosen) == max_per_category:
                break
        for rank, item in enumerate(chosen, start=1):
            item["rank"] = rank
            item["rank_key"] = [
                round(value, 6) if isinstance(value, float) else value
                for value in item.pop("_rank")
            ]
            item.pop("_identity")
        selected[category] = chosen
    return selected


def enclosing_crop(
    boxes: Sequence[Box],
    image_size: tuple[int, int],
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> list[int]:
    image_width, image_height = image_size
    x1 = min(float(box[0]) for box in boxes)
    y1 = min(float(box[1]) for box in boxes)
    x2 = max(float(box[2]) for box in boxes)
    y2 = max(float(box[3]) for box in boxes)
    content_width = max(1.0, x2 - x1)
    content_height = max(1.0, y2 - y1)
    width = max(MIN_SOURCE_CROP[0], content_width + 200.0)
    height = max(MIN_SOURCE_CROP[1], content_height + 160.0)
    target_aspect = output_size[0] / output_size[1]
    if width / height < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect
    width = min(width, float(image_width))
    height = min(height, float(image_height))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = min(max(0.0, center_x - width / 2.0), image_width - width)
    crop_y1 = min(max(0.0, center_y - height / 2.0), image_height - height)
    return [
        round(crop_x1),
        round(crop_y1),
        round(crop_x1 + width),
        round(crop_y1 + height),
    ]


def transform_box(box: Box, crop: Box, output_size: tuple[int, int]) -> list[float]:
    scale_x = output_size[0] / (float(crop[2]) - float(crop[0]))
    scale_y = output_size[1] / (float(crop[3]) - float(crop[1]))
    return [
        (float(box[0]) - float(crop[0])) * scale_x,
        (float(box[1]) - float(crop[1])) * scale_y,
        (float(box[2]) - float(crop[0])) * scale_x,
        (float(box[3]) - float(crop[1])) * scale_y,
    ]


def draw_dashed_rectangle(
    drawing: ImageDraw.ImageDraw,
    box: Box,
    color: str,
    width: int,
    dash_length: int = 18,
) -> None:
    x1, y1, x2, y2 = (round(float(value)) for value in box)
    for start in range(x1, x2 + 1, dash_length * 2):
        end = min(start + dash_length, x2)
        drawing.line((start, y1, end, y1), fill=color, width=width)
        drawing.line((start, y2, end, y2), fill=color, width=width)
    for start in range(y1, y2 + 1, dash_length * 2):
        end = min(start + dash_length, y2)
        drawing.line((x1, start, x1, end), fill=color, width=width)
        drawing.line((x2, start, x2, end), fill=color, width=width)


def render_candidate(
    source_path: Path,
    item: JsonObject,
    destination: Path,
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> None:
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    crop = enclosing_crop(item["focus_boxes"], image.size, output_size)
    crop_tuple = (crop[0], crop[1], crop[2], crop[3])
    rendered = image.crop(crop_tuple).resize(output_size, Image.Resampling.LANCZOS)
    drawing = ImageDraw.Draw(rendered)
    transformed_annotations = []
    for evidence_annotation in item["annotations"]:
        transformed = transform_box(
            evidence_annotation["bbox_xyxy"],
            crop,
            output_size,
        )
        kind = evidence_annotation["kind"]
        if kind == "false_negative":
            draw_dashed_rectangle(drawing, transformed, FN_COLOR, width=8)
        elif kind in {"false_positive", "duplicate_false_positive"}:
            drawing.rectangle(
                tuple(round(value) for value in transformed),
                outline=FP_COLOR,
                width=8,
            )
        elif kind == "true_positive":
            drawing.rectangle(
                tuple(round(value) for value in transformed),
                outline=CORRECT_COLORS[evidence_annotation["class_name"]],
                width=8,
            )
        else:
            raise ValueError(f"unsupported annotation kind: {kind}")
        transformed_annotations.append(
            {
                **evidence_annotation,
                "rendered_bbox_xyxy": rounded_box(transformed),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(destination, format="PNG", optimize=False, compress_level=9)
    item["source_crop_xyxy"] = crop
    item["output_size"] = list(output_size)
    item["annotations"] = transformed_annotations


def local_source_candidates(frame: int, rgb_dir: Path) -> list[Path]:
    return [
        rgb_dir / f"{frame}_rgb.png",
        ROOT / "presentation" / "stills" / "source" / f"{frame}_rgb.png",
    ]


def ensure_source_image(
    frame: int,
    rgb_dir: Path,
    output_dir: Path,
    remote_host: str,
    remote_root: str,
) -> Path:
    destination = output_dir / "source" / f"{frame}_rgb.png"
    if destination.is_file():
        return destination
    for source in local_source_candidates(frame, rgb_dir):
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            return destination
    atomic_remote_copy(
        remote_host,
        f"{remote_root}/data/rgb_out/{frame}_rgb.png",
        destination,
    )
    return destination


def safe_method_slug(method: str) -> str:
    return method.replace("+", "_and_")


def clean_previous_outputs(output_dir: Path) -> None:
    for category in CATEGORY_ORDER:
        category_dir = output_dir / category
        if category_dir.is_dir():
            shutil.rmtree(category_dir)
    for filename in ("manifest.json", "montage.typ", "montage.png"):
        (output_dir / filename).unlink(missing_ok=True)


def prune_unselected_sources(output_dir: Path, selected_frames: Sequence[int]) -> None:
    selected_names = {f"{frame}_rgb.png" for frame in selected_frames}
    source_dir = output_dir / "source"
    if not source_dir.is_dir():
        return
    for path in source_dir.iterdir():
        if path.is_file() and path.name not in selected_names:
            path.unlink()


def build_manifest(args: argparse.Namespace) -> JsonObject:
    ground_truth_path = resolve_ground_truth(args.ground_truth)
    radar_bounded_path = resolve_radar_bounded(
        args.radar_bounded,
        args.output_dir,
        args.remote_host,
        args.remote_root,
    )
    ground_truth_records, ground_truth_hash = read_ground_truth(ground_truth_path)
    vision_records = read_jsonl(args.vision_only)
    gated_records = read_jsonl(args.radar_gated)
    radar_records_list = read_jsonl(radar_bounded_path)
    ground_truth = {
        int(record["camera_frame"]): list(record["objects"]) for record in ground_truth_records
    }
    predictions = {
        "vision_only_tiled": records_by_frame(vision_records, "detections"),
        "radar_confidence_gated": records_by_frame(gated_records, "detections"),
        "radar_bounded_crops": records_by_frame(
            radar_records_list,
            "remapped_deduped",
        ),
    }
    radar_records = {int(record["camera_frame"]): record for record in radar_records_list}
    expected_frames = sorted(ground_truth)
    for method, method_predictions in predictions.items():
        missing = sorted(set(expected_frames) - set(method_predictions))
        if missing:
            raise ValueError(f"{method} is missing {len(missing)} ground-truth frames")

    mined = mine_candidates(
        ground_truth,
        predictions,
        radar_records,
        args.iou_threshold,
    )
    selected = select_candidates(mined, args.max_per_category)
    clean_previous_outputs(args.output_dir)

    selected_frames = sorted(
        {int(item["camera_frame"]) for items in selected.values() for item in items}
    )
    prune_unselected_sources(args.output_dir, selected_frames)
    source_paths = {
        frame: ensure_source_image(
            frame,
            args.rgb_dir,
            args.output_dir,
            args.remote_host,
            args.remote_root,
        )
        for frame in selected_frames
    }
    for category in CATEGORY_ORDER:
        for item in selected[category]:
            frame = int(item["camera_frame"])
            filename = (
                f"{int(item['rank']):02d}_cf_{frame:06d}_"
                f"{safe_method_slug(str(item['primary_method']))}.png"
            )
            destination = args.output_dir / category / filename
            render_candidate(source_paths[frame], item, destination)
            item["source_image"] = str(source_paths[frame].relative_to(args.output_dir))
            item["source_sha256"] = sha256_file(source_paths[frame])
            item["evidence_image"] = str(destination.relative_to(args.output_dir))
            item["evidence_sha256"] = sha256_file(destination)
            item.pop("focus_boxes")

    manifest = {
        "schema_version": 1,
        "matching": {
            "class_aware": True,
            "iou_threshold": args.iou_threshold,
            "prediction_order": "score_desc_then_class_box_and_input_index",
            "ground_truth_assignment": "greedy_one_to_one",
        },
        "rendering": {
            "max_per_category": args.max_per_category,
            "output_size": list(OUTPUT_SIZE),
            "no_text_overlays": True,
            "colors": {
                "false_positive": FP_COLOR,
                "false_negative": FN_COLOR,
                "correct_by_class": CORRECT_COLORS,
            },
            "false_negative_line_style": "dashed",
        },
        "inputs": {
            "ground_truth": {
                "path": str(ground_truth_path.relative_to(ROOT)),
                "content_sha256": ground_truth_hash,
            },
            "vision_only": {
                "path": str(args.vision_only.relative_to(ROOT)),
                "sha256": sha256_file(args.vision_only),
            },
            "radar_confidence_gated": {
                "path": str(args.radar_gated.relative_to(ROOT)),
                "sha256": sha256_file(args.radar_gated),
            },
            "radar_bounded": {
                "path": str(radar_bounded_path.relative_to(ROOT)),
                "sha256": sha256_file(radar_bounded_path),
            },
        },
        "selected_source_frames": selected_frames,
        "categories": selected,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    if not math.isclose(args.iou_threshold, IOU_THRESHOLD):
        raise ValueError("presentation evidence is verified only at IoU 0.5")
    manifest = build_manifest(args)
    counts = {category: len(manifest["categories"][category]) for category in CATEGORY_ORDER}
    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
