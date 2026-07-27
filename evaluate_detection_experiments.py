#!/usr/bin/env python3
"""Merge experiment shards and score all three detection methods."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_detection_experiments import box_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("out/datasets/dream_fusion_yolo/manifest.jsonl"),
    )
    parser.add_argument(
        "--radar-bounded",
        type=Path,
        default=Path("out/detections.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/experiments/detection_comparison"),
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def match_frame(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    class_name: str,
    iou_threshold: float,
) -> dict[str, int]:
    truth = [obj["bbox_xyxy"] for obj in ground_truth if obj["class_name"] == class_name]
    predicted = sorted(
        (
            prediction
            for prediction in predictions
            if str(prediction["label"]).lower() == class_name
        ),
        key=lambda prediction: float(prediction["score"]),
        reverse=True,
    )
    unmatched = set(range(len(truth)))
    true_positive = 0
    false_positive = 0
    for prediction in predicted:
        best_iou, best_index = max(
            ((box_iou(prediction["bbox_xyxy"], truth[index]), index) for index in unmatched),
            default=(0.0, None),
        )
        if best_index is not None and best_iou >= iou_threshold:
            true_positive += 1
            unmatched.remove(best_index)
        else:
            false_positive += 1
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": len(unmatched),
    }


def score_method(
    ground_truth_by_frame: dict[int, list[dict[str, Any]]],
    predictions_by_frame: dict[int, list[dict[str, Any]]],
    iou_threshold: float,
) -> dict[str, Any]:
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for camera_frame, truth in ground_truth_by_frame.items():
        predictions = predictions_by_frame.get(camera_frame, [])
        for class_name in ("boat", "buoy"):
            frame_counts = match_frame(
                truth,
                predictions,
                class_name,
                iou_threshold,
            )
            for key, value in frame_counts.items():
                counts[class_name][key] += value

    result = {}
    combined = {"tp": 0, "fp": 0, "fn": 0}
    for class_name in ("boat", "buoy"):
        class_counts = counts[class_name]
        precision = (
            class_counts["tp"] / (class_counts["tp"] + class_counts["fp"])
            if class_counts["tp"] + class_counts["fp"]
            else 0.0
        )
        recall = (
            class_counts["tp"] / (class_counts["tp"] + class_counts["fn"])
            if class_counts["tp"] + class_counts["fn"]
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[class_name] = {
            **class_counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        for key in combined:
            combined[key] += class_counts[key]
    precision = combined["tp"] / (combined["tp"] + combined["fp"])
    recall = combined["tp"] / (combined["tp"] + combined["fn"])
    combined["precision"] = precision
    combined["recall"] = recall
    combined["f1"] = 2 * precision * recall / (precision + recall)
    result["overall"] = combined
    return result


def write_predictions(path: Path, records: list[dict[str, Any]], field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "camera_frame": record["camera_frame"],
                    "detections": record[field],
                },
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def filter_predictions(
    predictions_by_frame: dict[int, list[dict[str, Any]]],
    confidence_threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    return {
        frame: [
            prediction
            for prediction in predictions
            if float(prediction["score"]) >= confidence_threshold
        ]
        for frame, predictions in predictions_by_frame.items()
    }


def score_by_split(
    ground_truth: dict[int, list[dict[str, Any]]],
    predictions: dict[int, list[dict[str, Any]]],
    frame_splits: dict[int, str],
    iou_threshold: float,
) -> dict[str, Any]:
    scores = {"all": score_method(ground_truth, predictions, iou_threshold)}
    for split in ("train", "val", "test"):
        split_truth = {
            frame: objects
            for frame, objects in ground_truth.items()
            if frame_splits[frame] == split
        }
        scores[split] = score_method(split_truth, predictions, iou_threshold)
    return scores


def main() -> None:
    args = parse_args()
    shard_records = []
    for shard in args.shard:
        shard_records.extend(read_jsonl(shard))
    shard_records.sort(key=lambda record: int(record["camera_frame"]))
    frames = [int(record["camera_frame"]) for record in shard_records]
    if len(shard_records) != 300 or frames != list(range(119, 419)):
        raise ValueError(
            "experiment shards must contain camera frames 119 through 418 exactly once"
        )

    ground_truth_records = read_jsonl(args.ground_truth)
    ground_truth = {
        int(record["camera_frame"]): record["objects"] for record in ground_truth_records
    }
    frame_splits = {
        int(record["camera_frame"]): str(record["split"]) for record in ground_truth_records
    }
    vision = {int(record["camera_frame"]): record["vision_detections"] for record in shard_records}
    radar_gated_candidates = {
        int(record["camera_frame"]): record["radar_gated_detections"] for record in shard_records
    }
    radar_bounded_records = read_jsonl(args.radar_bounded)
    radar_bounded = {
        int(record["camera_frame"]): record.get("remapped_deduped", [])
        for record in radar_bounded_records
    }

    validation_truth = {
        frame: objects for frame, objects in ground_truth.items() if frame_splits[frame] == "val"
    }
    threshold_sweep = []
    for integer_threshold in range(10, 19):
        threshold = integer_threshold / 100
        filtered = filter_predictions(radar_gated_candidates, threshold)
        score = score_method(validation_truth, filtered, args.iou_threshold)
        threshold_sweep.append(
            {
                "threshold": threshold,
                "validation": score,
            }
        )
    selected = max(
        threshold_sweep,
        key=lambda item: (
            item["validation"]["overall"]["f1"],
            item["threshold"],
        ),
    )
    selected_threshold_value = selected["threshold"]
    if not isinstance(selected_threshold_value, (int, float)):
        raise TypeError("selected confidence threshold must be numeric")
    selected_threshold = float(selected_threshold_value)
    radar_gated = filter_predictions(radar_gated_candidates, selected_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        args.output_dir / "vision_only_tiled.jsonl",
        shard_records,
        "vision_detections",
    )
    radar_gated_records = [
        {
            "camera_frame": record["camera_frame"],
            "selected_detections": radar_gated[int(record["camera_frame"])],
        }
        for record in shard_records
    ]
    write_predictions(
        args.output_dir / "radar_confidence_gated.jsonl",
        radar_gated_records,
        "selected_detections",
    )
    metrics = {
        "iou_threshold": args.iou_threshold,
        "frame_count": len(shard_records),
        "configuration": {
            "vision_confidence_threshold": 0.18,
            "radar_candidate_threshold": 0.10,
            "selected_radar_gate_threshold": selected_threshold,
            "threshold_selection_split": "val",
        },
        "radar_gate_validation_sweep": threshold_sweep,
        "methods": {
            "vision_only_tiled": score_by_split(
                ground_truth,
                vision,
                frame_splits,
                args.iou_threshold,
            ),
            "radar_confidence_gated": score_by_split(
                ground_truth,
                radar_gated,
                frame_splits,
                args.iou_threshold,
            ),
            "radar_bounded_crops": score_by_split(
                ground_truth,
                radar_bounded,
                frame_splits,
                args.iou_threshold,
            ),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
