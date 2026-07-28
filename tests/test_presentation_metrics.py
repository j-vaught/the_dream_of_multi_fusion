from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from presentation.metrics.pipeline import (
    DEFAULT_FALLBACK_METRICS,
    DEFAULT_GROUND_TRUTH,
    DEFAULT_RADAR_BOUNDED,
    DEFAULT_RADAR_GATED,
    DEFAULT_VISION,
    EXPECTED_FRAMES,
    box_iou,
    build_metrics,
    load_radar_bounded_predictions,
    match_frame,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PresentationMetricsUnitTest(unittest.TestCase):
    def test_iou_threshold_boundary_is_inclusive(self) -> None:
        truth = [
            {
                "track_id": "boat",
                "class_name": "boat",
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
            }
        ]
        prediction = [
            {
                "label": "boat",
                "score": 0.8,
                "bbox_xyxy": [0.0, 0.0, 5.0, 10.0],
            }
        ]
        self.assertEqual(
            box_iou(truth[0]["bbox_xyxy"], prediction[0]["bbox_xyxy"]),
            0.5,
        )
        outcome = match_frame(truth, prediction, "boat", 0.5)
        self.assertEqual((outcome.tp, outcome.fp, outcome.fn), (1, 0, 0))

    def test_matching_is_deterministic_and_one_to_one(self) -> None:
        truth = [
            {
                "track_id": "b2",
                "class_name": "buoy",
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
            },
            {
                "track_id": "b1",
                "class_name": "buoy",
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
            },
        ]
        predictions = [
            {
                "label": "buoy",
                "score": 0.9,
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
            }
        ]
        first = match_frame(truth, predictions, "buoy", 0.5)
        second = match_frame(truth, predictions, "buoy", 0.5)
        self.assertEqual(first, second)
        self.assertEqual(first.matches[0].track_id, "b1")
        self.assertEqual((first.tp, first.fp, first.fn), (1, 0, 1))

    def test_radar_bounded_loader_accepts_remapped_deduped_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "radar-bounded_full.jsonl"
            records = []
            for frame in EXPECTED_FRAMES:
                detections = []
                if frame == EXPECTED_FRAMES[0]:
                    detections = [
                        {
                            "label": "navigation buoy",
                            "score": 0.75,
                            "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                            "source_tracks": [7],
                        }
                    ]
                records.append(
                    json.dumps(
                        {
                            "camera_frame": frame,
                            "remapped_deduped": detections,
                        },
                        separators=(",", ":"),
                    )
                )
            path.write_text("\n".join(records) + "\n", encoding="utf-8")
            loaded = load_radar_bounded_predictions(path)
        self.assertEqual(len(loaded), 300)
        self.assertEqual(loaded[119][0]["label"], "buoy")
        self.assertEqual(loaded[119][0]["source_tracks"], [7])


class PresentationMetricsIntegrationTest(unittest.TestCase):
    def test_committed_sources_recompute_exactly_and_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            output = temporary / "data"
            manifest_path = temporary / "metrics_manifest.json"
            manifest = build_metrics(
                ground_truth_path=DEFAULT_GROUND_TRUTH,
                vision_path=DEFAULT_VISION,
                radar_gated_path=DEFAULT_RADAR_GATED,
                radar_bounded_path=DEFAULT_RADAR_BOUNDED,
                fallback_metrics_path=DEFAULT_FALLBACK_METRICS,
                output_dir=output,
                manifest_path=manifest_path,
            )

            self.assertFalse(manifest["fallback_used"])
            self.assertTrue(
                all(source["detail_available"] for source in manifest["prediction_sources"])
            )

            method_rows = {row["method"]: row for row in read_csv(output / "method_metrics.csv")}
            committed = json.loads(DEFAULT_FALLBACK_METRICS.read_text(encoding="utf-8"))
            for method in (
                "vision_only_tiled",
                "radar_confidence_gated",
                "radar_bounded_crops",
            ):
                expected = committed["methods"][method]["all"]["overall"]
                actual = method_rows[method]
                for count in ("tp", "fp", "fn"):
                    self.assertEqual(int(actual[count]), expected[count])
                for metric in ("precision", "recall", "f1"):
                    self.assertAlmostEqual(float(actual[metric]), expected[metric])

            frame_rows = read_csv(output / "frame_metrics.csv")
            track_rows = read_csv(output / "track_metrics.csv")
            self.assertEqual(len(frame_rows), 900)
            self.assertEqual(len(track_rows), 27)
            self.assertEqual(
                {row["method"] for row in frame_rows},
                {
                    "vision_only_tiled",
                    "radar_confidence_gated",
                    "radar_bounded_crops",
                },
            )

            first_snapshot = {path.name: path.read_bytes() for path in sorted(output.glob("*.csv"))}
            first_manifest = manifest_path.read_bytes()
            build_metrics(
                ground_truth_path=DEFAULT_GROUND_TRUTH,
                vision_path=DEFAULT_VISION,
                radar_gated_path=DEFAULT_RADAR_GATED,
                radar_bounded_path=DEFAULT_RADAR_BOUNDED,
                fallback_metrics_path=DEFAULT_FALLBACK_METRICS,
                output_dir=output,
                manifest_path=manifest_path,
            )
            second_snapshot = {
                path.name: path.read_bytes() for path in sorted(output.glob("*.csv"))
            }
            self.assertEqual(first_snapshot, second_snapshot)
            self.assertEqual(first_manifest, manifest_path.read_bytes())

    def test_missing_bounded_source_uses_explicit_aggregate_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            output = temporary / "data"
            manifest = build_metrics(
                ground_truth_path=DEFAULT_GROUND_TRUTH,
                vision_path=DEFAULT_VISION,
                radar_gated_path=DEFAULT_RADAR_GATED,
                radar_bounded_path=temporary / "missing-radar-bounded.jsonl",
                fallback_metrics_path=DEFAULT_FALLBACK_METRICS,
                output_dir=output,
                manifest_path=temporary / "metrics_manifest.json",
            )
            self.assertTrue(manifest["fallback_used"])
            bounded_source = next(
                source
                for source in manifest["prediction_sources"]
                if source["method"] == "radar_bounded_crops"
            )
            self.assertFalse(bounded_source["detail_available"])
            self.assertEqual(bounded_source["mode"], "committed_metrics_json")
            frame_rows = read_csv(output / "frame_metrics.csv")
            track_rows = read_csv(output / "track_metrics.csv")
            self.assertEqual(len(frame_rows), 600)
            self.assertEqual(len(track_rows), 18)
            self.assertNotIn(
                "radar_bounded_crops",
                {row["method"] for row in track_rows},
            )


if __name__ == "__main__":
    unittest.main()
