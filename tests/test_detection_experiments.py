from __future__ import annotations

import unittest

from evaluate_detection_experiments import filter_predictions, match_frame
from render_detection_experiment_videos import categorize_detections
from run_detection_experiments import (
    axis_positions,
    class_aware_nms,
    normalize_label,
    radar_supported,
    tile_boxes,
)


class DetectionExperimentTest(unittest.TestCase):
    def test_tiles_cover_full_image(self) -> None:
        boxes = tile_boxes(5320, 3032, 2048, 1600, 300)
        self.assertEqual(len(boxes), 9)
        self.assertEqual(boxes[0][:2], [0, 0])
        self.assertEqual(boxes[-1][2:], [5320, 3032])
        self.assertEqual(axis_positions(100, 200, 10), [0])

    def test_label_normalization_matches_radar_pipeline(self) -> None:
        self.assertEqual(normalize_label("navigation buoy"), "buoy")
        self.assertEqual(normalize_label("green"), "buoy")
        self.assertEqual(normalize_label("vessel"), "boat")

    def test_class_aware_nms_keeps_best_overlapping_box(self) -> None:
        detections = [
            {"label": "boat", "score": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
            {"label": "boat", "score": 0.8, "bbox_xyxy": [1, 1, 11, 11]},
            {"label": "buoy", "score": 0.7, "bbox_xyxy": [1, 1, 11, 11]},
        ]
        kept = class_aware_nms(detections, 0.5)
        self.assertEqual(len(kept), 2)
        self.assertEqual([item["label"] for item in kept], ["boat", "buoy"])

    def test_radar_support_accepts_center_or_overlap(self) -> None:
        regions: list[list[float]] = [[100, 100, 200, 200]]
        self.assertTrue(radar_supported([120, 120, 130, 130], regions, 0.25))
        self.assertFalse(radar_supported([0, 0, 50, 50], regions, 0.25))

    def test_matching_is_one_to_one(self) -> None:
        truth = [
            {"class_name": "boat", "bbox_xyxy": [0, 0, 10, 10]},
        ]
        predictions = [
            {"label": "boat", "score": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
            {"label": "boat", "score": 0.8, "bbox_xyxy": [0, 0, 10, 10]},
        ]
        self.assertEqual(
            match_frame(truth, predictions, "boat", 0.5),
            {"tp": 1, "fp": 1, "fn": 0},
        )

    def test_confidence_filter_preserves_selected_predictions(self) -> None:
        predictions = {
            119: [
                {"score": 0.159, "label": "buoy"},
                {"score": 0.16, "label": "buoy"},
                {"score": 0.8, "label": "boat"},
            ]
        }
        filtered = filter_predictions(predictions, 0.16)
        self.assertEqual([item["score"] for item in filtered[119]], [0.16, 0.8])

    def test_video_categories_match_evaluator(self) -> None:
        ground_truth = [
            {"class_name": "boat", "bbox_xyxy": [0, 0, 10, 10]},
            {"class_name": "buoy", "bbox_xyxy": [20, 20, 25, 30]},
        ]
        predictions = [
            {"label": "vessel", "score": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
            {"label": "buoy", "score": 0.8, "bbox_xyxy": [50, 50, 60, 60]},
        ]
        true_positives, false_positives, false_negatives = categorize_detections(
            ground_truth,
            predictions,
            0.5,
        )
        self.assertEqual(len(true_positives), 1)
        self.assertEqual(len(false_positives), 1)
        self.assertEqual(len(false_negatives), 1)
        self.assertEqual(true_positives[0]["label"], "boat")
        self.assertEqual(false_negatives[0]["class_name"], "buoy")


if __name__ == "__main__":
    unittest.main()
