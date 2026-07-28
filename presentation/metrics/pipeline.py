#!/usr/bin/env python3
"""Recompute deterministic detection metrics and figure-ready CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_IOU_THRESHOLD = 0.5
EXPECTED_FRAMES = tuple(range(119, 419))
CLASS_NAMES = ("boat", "buoy")
METHODS = (
    "vision_only_tiled",
    "radar_confidence_gated",
    "radar_bounded_crops",
)
DETAIL_SOURCE = "prediction_jsonl"
FALLBACK_SOURCE = "committed_metrics_json"
OUTPUT_FILES = (
    "method_metrics.csv",
    "class_metrics.csv",
    "track_metrics.csv",
    "frame_metrics.csv",
    "frame_class_metrics.csv",
    "track_frame_matches.csv",
    "match_events.csv",
    "frame_rolling_metrics.csv",
    "figure_01_overall_quality.csv",
    "figure_02_class_recall.csv",
    "figure_03_detection_outcomes.csv",
    "figure_04_track_recall.csv",
    "figure_05_frame_trends.csv",
    "figure_06_cumulative_false_positives.csv",
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUND_TRUTH = ROOT / "datasets/dream_fusion_yolo/manifest.jsonl"
DEFAULT_VISION = ROOT / "experiments/detection_comparison/vision_only_tiled.jsonl"
DEFAULT_RADAR_GATED = ROOT / "experiments/detection_comparison/radar_confidence_gated.jsonl"
DEFAULT_RADAR_BOUNDED = ROOT / "experiments/detection_comparison/radar_bounded_full.jsonl"
DEFAULT_FALLBACK_METRICS = ROOT / "experiments/detection_comparison/metrics.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_BUILD_MANIFEST = Path(__file__).resolve().parent / "metrics_manifest.json"


@dataclass(frozen=True)
class Match:
    prediction_index: int
    truth_index: int
    track_id: str
    class_name: str
    score: float
    iou: float


@dataclass(frozen=True)
class FrameMatch:
    matches: tuple[Match, ...]
    false_positive_indices: tuple[int, ...]
    false_negative_indices: tuple[int, ...]

    @property
    def tp(self) -> int:
        return len(self.matches)

    @property
    def fp(self) -> int:
        return len(self.false_positive_indices)

    @property
    def fn(self) -> int:
        return len(self.false_negative_indices)


@dataclass(frozen=True)
class SourceState:
    method: str
    requested_path: Path
    selected_path: Path
    mode: str
    detail_available: bool


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(record)
    return records


def _validate_frame_records(records: Sequence[Mapping[str, Any]], source: Path) -> None:
    frames = [int(record["camera_frame"]) for record in records]
    if len(frames) != len(EXPECTED_FRAMES) or sorted(frames) != list(EXPECTED_FRAMES):
        raise ValueError(f"{source} must contain camera frames 119 through 418 exactly once")


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _box(value: Any, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} must be a four-value box")
    x1, y1, x2, y2 = (_number(item, field) for item in value)
    box = (x1, y1, x2, y2)
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError(f"{field} has inverted coordinates")
    return box


def normalize_label(value: Any) -> str:
    clean = str(value).strip().lower()
    if any(token in clean for token in ("boat", "vess", "ship", "watercraft", "yacht")):
        return "boat"
    if any(token in clean for token in ("buoy", "navig", "marker", "channel", "beacon", "daymark")):
        return "buoy"
    if clean in {"red", "green", "orange", "yellow", "black", "bu"}:
        return "buoy"
    return clean


def _normalized_prediction(
    prediction: Mapping[str, Any],
    source: Path,
    frame: int,
) -> dict[str, Any]:
    label = normalize_label(prediction.get("label", prediction.get("raw_label", "")))
    if label not in CLASS_NAMES:
        raise ValueError(f"{source} frame {frame} has unsupported label {label!r}")
    return {
        **prediction,
        "label": label,
        "score": _number(prediction.get("score"), "score"),
        "bbox_xyxy": _box(prediction.get("bbox_xyxy"), "bbox_xyxy"),
    }


def load_ground_truth(
    path: Path,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, str],
]:
    records = _read_jsonl(path)
    _validate_frame_records(records, path)
    objects_by_frame: dict[int, list[dict[str, Any]]] = {}
    splits: dict[int, str] = {}
    for record in records:
        frame = int(record["camera_frame"])
        objects: list[dict[str, Any]] = []
        raw_objects = record.get("objects")
        if not isinstance(raw_objects, list):
            raise ValueError(f"{path} frame {frame} objects must be a list")
        for raw_object in raw_objects:
            if not isinstance(raw_object, dict):
                raise ValueError(f"{path} frame {frame} has a non-object annotation")
            class_name = normalize_label(raw_object.get("class_name", ""))
            if class_name not in CLASS_NAMES:
                raise ValueError(f"{path} frame {frame} has unsupported class {class_name!r}")
            objects.append(
                {
                    **raw_object,
                    "track_id": str(raw_object["track_id"]),
                    "class_name": class_name,
                    "bbox_xyxy": _box(raw_object.get("bbox_xyxy"), "bbox_xyxy"),
                }
            )
        objects_by_frame[frame] = objects
        splits[frame] = str(record["split"])
    return objects_by_frame, splits


def load_predictions(path: Path, detections_field: str) -> dict[int, list[dict[str, Any]]]:
    records = _read_jsonl(path)
    _validate_frame_records(records, path)
    result: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        frame = int(record["camera_frame"])
        detections = record.get(detections_field)
        if not isinstance(detections, list):
            raise ValueError(f"{path} frame {frame} field {detections_field!r} must be a list")
        result[frame] = [_normalized_prediction(detection, path, frame) for detection in detections]
    return result


def load_radar_bounded_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load the expected radar-bounded remapped_deduped prediction schema."""

    return load_predictions(path, "remapped_deduped")


def box_iou(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_frame(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    class_name: str,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> FrameMatch:
    truth_indices = [
        index
        for index, truth in enumerate(ground_truth)
        if normalize_label(truth["class_name"]) == class_name
    ]
    prediction_indices = [
        index
        for index, prediction in enumerate(predictions)
        if normalize_label(prediction["label"]) == class_name
    ]
    prediction_indices.sort(
        key=lambda index: (
            -float(predictions[index]["score"]),
            tuple(float(value) for value in predictions[index]["bbox_xyxy"]),
            index,
        )
    )
    unmatched_truth = set(truth_indices)
    matches: list[Match] = []
    false_positives: list[int] = []
    for prediction_index in prediction_indices:
        candidates = sorted(
            (
                -box_iou(
                    predictions[prediction_index]["bbox_xyxy"],
                    ground_truth[truth_index]["bbox_xyxy"],
                ),
                str(ground_truth[truth_index]["track_id"]),
                truth_index,
            )
            for truth_index in unmatched_truth
        )
        if not candidates:
            false_positives.append(prediction_index)
            continue
        negative_iou, _, truth_index = candidates[0]
        iou = -negative_iou
        if iou < iou_threshold:
            false_positives.append(prediction_index)
            continue
        unmatched_truth.remove(truth_index)
        matches.append(
            Match(
                prediction_index=prediction_index,
                truth_index=truth_index,
                track_id=str(ground_truth[truth_index]["track_id"]),
                class_name=class_name,
                score=float(predictions[prediction_index]["score"]),
                iou=iou,
            )
        )
    matches.sort(key=lambda match: (match.track_id, match.prediction_index))
    return FrameMatch(
        matches=tuple(matches),
        false_positive_indices=tuple(sorted(false_positives)),
        false_negative_indices=tuple(
            sorted(unmatched_truth, key=lambda index: str(ground_truth[index]["track_id"]))
        ),
    )


def _ratios(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _score_detailed_method(
    method: str,
    truth_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    predictions_by_frame: Mapping[int, Sequence[Mapping[str, Any]]],
    splits: Mapping[int, str],
    iou_threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    frame_class_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    track_frame_rows: list[dict[str, Any]] = []
    for frame in EXPECTED_FRAMES:
        truth = truth_by_frame[frame]
        predictions = predictions_by_frame[frame]
        for class_name in CLASS_NAMES:
            outcome = match_frame(truth, predictions, class_name, iou_threshold)
            prediction_count = sum(
                normalize_label(prediction["label"]) == class_name for prediction in predictions
            )
            truth_count = sum(normalize_label(item["class_name"]) == class_name for item in truth)
            counts = {
                "tp": outcome.tp,
                "fp": outcome.fp,
                "fn": outcome.fn,
            }
            frame_class_rows.append(
                {
                    "method": method,
                    "camera_frame": frame,
                    "split": splits[frame],
                    "class_name": class_name,
                    "ground_truth_count": truth_count,
                    "prediction_count": prediction_count,
                    **counts,
                    **_ratios(**counts),
                    "source_mode": DETAIL_SOURCE,
                    "detail_available": True,
                }
            )
            matched_by_truth = {match.truth_index: match for match in outcome.matches}
            for truth_index, item in enumerate(truth):
                if normalize_label(item["class_name"]) != class_name:
                    continue
                matched = matched_by_truth.get(truth_index)
                track_frame_rows.append(
                    {
                        "method": method,
                        "camera_frame": frame,
                        "split": splits[frame],
                        "class_name": class_name,
                        "track_id": str(item["track_id"]),
                        "matched": matched is not None,
                        "iou": matched.iou if matched else "",
                        "score": matched.score if matched else "",
                        "source_mode": DETAIL_SOURCE,
                        "detail_available": True,
                    }
                )
            for match in outcome.matches:
                match_rows.append(
                    {
                        "method": method,
                        "camera_frame": frame,
                        "split": splits[frame],
                        "class_name": class_name,
                        "outcome": "tp",
                        "track_id": match.track_id,
                        "score": match.score,
                        "iou": match.iou,
                        "source_mode": DETAIL_SOURCE,
                    }
                )
            for prediction_index in outcome.false_positive_indices:
                match_rows.append(
                    {
                        "method": method,
                        "camera_frame": frame,
                        "split": splits[frame],
                        "class_name": class_name,
                        "outcome": "fp",
                        "track_id": "",
                        "score": float(predictions[prediction_index]["score"]),
                        "iou": "",
                        "source_mode": DETAIL_SOURCE,
                    }
                )
            for truth_index in outcome.false_negative_indices:
                match_rows.append(
                    {
                        "method": method,
                        "camera_frame": frame,
                        "split": splits[frame],
                        "class_name": class_name,
                        "outcome": "fn",
                        "track_id": str(truth[truth_index]["track_id"]),
                        "score": "",
                        "iou": "",
                        "source_mode": DETAIL_SOURCE,
                    }
                )
    return {
        "frame_class": frame_class_rows,
        "matches": match_rows,
        "track_frame": track_frame_rows,
    }


def _sum_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    result = {"tp": 0, "fp": 0, "fn": 0}
    for row in rows:
        for key in result:
            result[key] += int(row[key])
    return result


def _aggregate_detailed(
    detailed: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    frame_class_rows = list(detailed["frame_class"])
    method_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []

    methods = sorted({str(row["method"]) for row in frame_class_rows})
    for method in methods:
        selected = [row for row in frame_class_rows if row["method"] == method]
        counts = _sum_counts(selected)
        method_rows.append(
            {
                "method": method,
                "class_name": "overall",
                **counts,
                **_ratios(**counts),
                "source_mode": DETAIL_SOURCE,
                "detail_available": True,
            }
        )
        for class_name in CLASS_NAMES:
            class_selected = [row for row in selected if row["class_name"] == class_name]
            class_counts = _sum_counts(class_selected)
            class_rows.append(
                {
                    "method": method,
                    "class_name": class_name,
                    **class_counts,
                    **_ratios(**class_counts),
                    "source_mode": DETAIL_SOURCE,
                    "detail_available": True,
                }
            )
        for frame in EXPECTED_FRAMES:
            frame_selected = [row for row in selected if int(row["camera_frame"]) == frame]
            frame_counts = _sum_counts(frame_selected)
            frame_rows.append(
                {
                    "method": method,
                    "camera_frame": frame,
                    "split": frame_selected[0]["split"],
                    "ground_truth_count": frame_counts["tp"] + frame_counts["fn"],
                    "prediction_count": frame_counts["tp"] + frame_counts["fp"],
                    **frame_counts,
                    **_ratios(**frame_counts),
                    "source_mode": DETAIL_SOURCE,
                    "detail_available": True,
                }
            )

    grouped_tracks: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in detailed["track_frame"]:
        grouped_tracks[(str(row["method"]), str(row["class_name"]), str(row["track_id"]))].append(
            row
        )
    for (method, class_name, track_id), rows in sorted(grouped_tracks.items()):
        matched = [row for row in rows if bool(row["matched"])]
        ious = [float(row["iou"]) for row in matched]
        scores = [float(row["score"]) for row in matched]
        tp = len(matched)
        fn = len(rows) - tp
        track_rows.append(
            {
                "method": method,
                "class_name": class_name,
                "track_id": track_id,
                "ground_truth_instances": len(rows),
                "tp": tp,
                "fn": fn,
                "recall": tp / len(rows),
                "mean_iou": statistics.fmean(ious) if ious else "",
                "median_iou": statistics.median(ious) if ious else "",
                "mean_score": statistics.fmean(scores) if scores else "",
                "source_mode": DETAIL_SOURCE,
                "detail_available": True,
            }
        )
    return method_rows, class_rows, frame_rows, track_rows


def _load_fallback_rows(
    metrics_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    all_metrics = metrics["methods"]["radar_bounded_crops"]["all"]
    method_value = all_metrics["overall"]
    method_rows = [
        {
            "method": "radar_bounded_crops",
            "class_name": "overall",
            **{key: method_value[key] for key in ("tp", "fp", "fn")},
            **{key: method_value[key] for key in ("precision", "recall", "f1")},
            "source_mode": FALLBACK_SOURCE,
            "detail_available": False,
        }
    ]
    class_rows = []
    for class_name in CLASS_NAMES:
        value = all_metrics[class_name]
        class_rows.append(
            {
                "method": "radar_bounded_crops",
                "class_name": class_name,
                **{key: value[key] for key in ("tp", "fp", "fn")},
                **{key: value[key] for key in ("precision", "recall", "f1")},
                "source_mode": FALLBACK_SOURCE,
                "detail_available": False,
            }
        )
    return metrics, method_rows, class_rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _rolling_frame_rows(
    frame_rows: Sequence[Mapping[str, Any]],
    window: int = 15,
) -> list[dict[str, Any]]:
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_method[str(row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for method, rows in sorted(by_method.items()):
        ordered = sorted(rows, key=lambda row: int(row["camera_frame"]))
        cumulative_fp = 0
        for index, row in enumerate(ordered):
            start = max(0, index - window + 1)
            selected = ordered[start : index + 1]
            counts = _sum_counts(selected)
            cumulative_fp += int(row["fp"])
            output.append(
                {
                    "method": method,
                    "camera_frame": int(row["camera_frame"]),
                    "window_frames": len(selected),
                    "rolling_precision": _ratios(**counts)["precision"],
                    "rolling_recall": _ratios(**counts)["recall"],
                    "rolling_f1": _ratios(**counts)["f1"],
                    "cumulative_fp": cumulative_fp,
                    "source_mode": DETAIL_SOURCE,
                    "detail_available": True,
                }
            )
    return output


def _figure_inputs(
    output_dir: Path,
    method_rows: Sequence[Mapping[str, Any]],
    class_rows: Sequence[Mapping[str, Any]],
    track_rows: Sequence[Mapping[str, Any]],
    rolling_rows: Sequence[Mapping[str, Any]],
) -> None:
    quality_rows = [
        {
            "method": row["method"],
            "metric": metric,
            "value": row[metric],
            "source_mode": row["source_mode"],
            "detail_available": row["detail_available"],
        }
        for row in method_rows
        for metric in ("precision", "recall", "f1")
    ]
    recall_rows = [
        {
            "method": row["method"],
            "class_name": row["class_name"],
            "recall": row["recall"],
            "source_mode": row["source_mode"],
            "detail_available": row["detail_available"],
        }
        for row in class_rows
    ]
    outcome_rows = [
        {
            "method": row["method"],
            "tp": row["tp"],
            "fn": row["fn"],
            "fp": row["fp"],
            "source_mode": row["source_mode"],
            "detail_available": row["detail_available"],
        }
        for row in method_rows
    ]
    heatmap_rows = [
        {
            "method": row["method"],
            "track_id": row["track_id"],
            "class_name": row["class_name"],
            "ground_truth_instances": row["ground_truth_instances"],
            "recall": row["recall"],
        }
        for row in track_rows
    ]
    trend_rows = [
        {
            "method": row["method"],
            "camera_frame": row["camera_frame"],
            "rolling_recall": row["rolling_recall"],
            "rolling_f1": row["rolling_f1"],
        }
        for row in rolling_rows
    ]
    cumulative_rows = [
        {
            "method": row["method"],
            "camera_frame": row["camera_frame"],
            "cumulative_fp": row["cumulative_fp"],
        }
        for row in rolling_rows
    ]
    _write_csv(
        output_dir / "figure_01_overall_quality.csv",
        quality_rows,
        ("method", "metric", "value", "source_mode", "detail_available"),
    )
    _write_csv(
        output_dir / "figure_02_class_recall.csv",
        recall_rows,
        ("method", "class_name", "recall", "source_mode", "detail_available"),
    )
    _write_csv(
        output_dir / "figure_03_detection_outcomes.csv",
        outcome_rows,
        ("method", "tp", "fn", "fp", "source_mode", "detail_available"),
    )
    _write_csv(
        output_dir / "figure_04_track_recall.csv",
        heatmap_rows,
        ("method", "track_id", "class_name", "ground_truth_instances", "recall"),
    )
    _write_csv(
        output_dir / "figure_05_frame_trends.csv",
        trend_rows,
        ("method", "camera_frame", "rolling_recall", "rolling_f1"),
    )
    _write_csv(
        output_dir / "figure_06_cumulative_false_positives.csv",
        cumulative_rows,
        ("method", "camera_frame", "cumulative_fp"),
    )


def build_metrics(
    *,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH,
    vision_path: Path = DEFAULT_VISION,
    radar_gated_path: Path = DEFAULT_RADAR_GATED,
    radar_bounded_path: Path = DEFAULT_RADAR_BOUNDED,
    fallback_metrics_path: Path = DEFAULT_FALLBACK_METRICS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_BUILD_MANIFEST,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be between zero and one")
    truth_by_frame, splits = load_ground_truth(ground_truth_path)
    prediction_sources = {
        "vision_only_tiled": load_predictions(vision_path, "detections"),
        "radar_confidence_gated": load_predictions(radar_gated_path, "detections"),
    }
    source_states = [
        SourceState(
            method="vision_only_tiled",
            requested_path=vision_path,
            selected_path=vision_path,
            mode=DETAIL_SOURCE,
            detail_available=True,
        ),
        SourceState(
            method="radar_confidence_gated",
            requested_path=radar_gated_path,
            selected_path=radar_gated_path,
            mode=DETAIL_SOURCE,
            detail_available=True,
        ),
    ]
    fallback_metrics: dict[str, Any] | None = None
    fallback_method_rows: list[dict[str, Any]] = []
    fallback_class_rows: list[dict[str, Any]] = []
    if radar_bounded_path.exists():
        prediction_sources["radar_bounded_crops"] = load_radar_bounded_predictions(
            radar_bounded_path
        )
        source_states.append(
            SourceState(
                method="radar_bounded_crops",
                requested_path=radar_bounded_path,
                selected_path=radar_bounded_path,
                mode=DETAIL_SOURCE,
                detail_available=True,
            )
        )
    else:
        if not fallback_metrics_path.exists():
            raise FileNotFoundError(
                f"{radar_bounded_path} is absent and fallback metrics do not exist"
            )
        fallback_metrics, fallback_method_rows, fallback_class_rows = _load_fallback_rows(
            fallback_metrics_path
        )
        source_states.append(
            SourceState(
                method="radar_bounded_crops",
                requested_path=radar_bounded_path,
                selected_path=fallback_metrics_path,
                mode=FALLBACK_SOURCE,
                detail_available=False,
            )
        )

    detail_parts = [
        _score_detailed_method(
            method,
            truth_by_frame,
            predictions,
            splits,
            iou_threshold,
        )
        for method, predictions in sorted(prediction_sources.items())
    ]
    combined_detail = {
        key: [row for part in detail_parts for row in part[key]]
        for key in ("frame_class", "matches", "track_frame")
    }
    method_rows, class_rows, frame_rows, track_rows = _aggregate_detailed(combined_detail)
    method_rows.extend(fallback_method_rows)
    class_rows.extend(fallback_class_rows)
    method_rows.sort(key=lambda row: METHODS.index(str(row["method"])))
    class_rows.sort(
        key=lambda row: (
            METHODS.index(str(row["method"])),
            CLASS_NAMES.index(str(row["class_name"])),
        )
    )
    frame_rows.sort(key=lambda row: (METHODS.index(str(row["method"])), int(row["camera_frame"])))
    track_rows.sort(
        key=lambda row: (
            METHODS.index(str(row["method"])),
            0 if str(row["track_id"]) == "boat" else 1,
            str(row["track_id"]),
        )
    )
    rolling_rows = _rolling_frame_rows(frame_rows)

    _write_csv(
        output_dir / "method_metrics.csv",
        method_rows,
        (
            "method",
            "class_name",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "source_mode",
            "detail_available",
        ),
    )
    _write_csv(
        output_dir / "class_metrics.csv",
        class_rows,
        (
            "method",
            "class_name",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "source_mode",
            "detail_available",
        ),
    )
    _write_csv(
        output_dir / "track_metrics.csv",
        track_rows,
        (
            "method",
            "class_name",
            "track_id",
            "ground_truth_instances",
            "tp",
            "fn",
            "recall",
            "mean_iou",
            "median_iou",
            "mean_score",
            "source_mode",
            "detail_available",
        ),
    )
    _write_csv(
        output_dir / "frame_metrics.csv",
        frame_rows,
        (
            "method",
            "camera_frame",
            "split",
            "ground_truth_count",
            "prediction_count",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "source_mode",
            "detail_available",
        ),
    )
    _write_csv(
        output_dir / "frame_class_metrics.csv",
        combined_detail["frame_class"],
        (
            "method",
            "camera_frame",
            "split",
            "class_name",
            "ground_truth_count",
            "prediction_count",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "source_mode",
            "detail_available",
        ),
    )
    _write_csv(
        output_dir / "track_frame_matches.csv",
        combined_detail["track_frame"],
        (
            "method",
            "camera_frame",
            "split",
            "class_name",
            "track_id",
            "matched",
            "iou",
            "score",
            "source_mode",
            "detail_available",
        ),
    )
    _write_csv(
        output_dir / "match_events.csv",
        combined_detail["matches"],
        (
            "method",
            "camera_frame",
            "split",
            "class_name",
            "outcome",
            "track_id",
            "score",
            "iou",
            "source_mode",
        ),
    )
    _write_csv(
        output_dir / "frame_rolling_metrics.csv",
        rolling_rows,
        (
            "method",
            "camera_frame",
            "window_frames",
            "rolling_precision",
            "rolling_recall",
            "rolling_f1",
            "cumulative_fp",
            "source_mode",
            "detail_available",
        ),
    )
    _figure_inputs(
        output_dir,
        method_rows,
        class_rows,
        track_rows,
        rolling_rows,
    )

    source_manifest = []
    for state in source_states:
        source_manifest.append(
            {
                "method": state.method,
                "requested_path": _relative_or_absolute(state.requested_path),
                "selected_path": _relative_or_absolute(state.selected_path),
                "sha256": _sha256(state.selected_path),
                "mode": state.mode,
                "detail_available": state.detail_available,
            }
        )
    manifest = {
        "schema_version": "1.0",
        "frame_range": {"first": 119, "last": 418, "count": 300},
        "iou_threshold": iou_threshold,
        "matching": {
            "class_aware": True,
            "one_to_one": True,
            "prediction_order": "score_descending_then_bbox_then_source_index",
            "truth_tie_break": "iou_descending_then_track_id_then_source_index",
            "threshold_inclusive": True,
        },
        "ground_truth": {
            "path": _relative_or_absolute(ground_truth_path),
            "sha256": _sha256(ground_truth_path),
        },
        "prediction_sources": source_manifest,
        "fallback_used": fallback_metrics is not None,
        "fallback_notice": (
            "radar-bounded_full.jsonl is absent. Aggregate radar-bounded method and "
            "class metrics use the committed metrics JSON. Per-frame and per-track "
            "radar-bounded rows are unavailable and are not imputed."
            if fallback_metrics is not None
            else ""
        ),
        "outputs": list(OUTPUT_FILES),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--vision", type=Path, default=DEFAULT_VISION)
    parser.add_argument("--radar-gated", type=Path, default=DEFAULT_RADAR_GATED)
    parser.add_argument("--radar-bounded", type=Path, default=DEFAULT_RADAR_BOUNDED)
    parser.add_argument(
        "--fallback-metrics",
        type=Path,
        default=DEFAULT_FALLBACK_METRICS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_BUILD_MANIFEST)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_metrics(
        ground_truth_path=args.ground_truth,
        vision_path=args.vision,
        radar_gated_path=args.radar_gated,
        radar_bounded_path=args.radar_bounded,
        fallback_metrics_path=args.fallback_metrics,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
