from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from presentation.errors.mine_errors import (
    CATEGORY_ORDER,
    FP_COLOR,
    add_temporally_verified_omission_candidates,
    box_iou,
    enclosing_crop,
    match_detections,
    render_candidate,
    select_candidates,
)


def truth(class_name: str, box: list[float], track_id: str) -> dict[str, object]:
    return {
        "class_name": class_name,
        "bbox_xyxy": box,
        "track_id": track_id,
    }


def prediction(
    label: str,
    score: float,
    box: list[float],
) -> dict[str, object]:
    return {
        "label": label,
        "score": score,
        "bbox_xyxy": box,
    }


class PresentationErrorMiningTest(unittest.TestCase):
    def test_iou_has_expected_boundary_value(self) -> None:
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 20, 10]), 0.5)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_matching_is_class_aware_one_to_one_and_deterministic(self) -> None:
        ground_truth = [
            truth("boat", [0, 0, 10, 10], "boat"),
            truth("buoy", [20, 0, 30, 10], "b1"),
        ]
        predictions = [
            prediction("vessel", 0.8, [0, 0, 10, 10]),
            prediction("boat", 0.9, [0, 0, 10, 10]),
            prediction("buoy", 0.7, [20, 0, 30, 10]),
        ]
        result = match_detections(ground_truth, predictions, 0.5)
        self.assertEqual(len(result["true_positives"]), 2)
        self.assertEqual(len(result["false_positives"]), 1)
        self.assertEqual(len(result["false_negatives"]), 0)
        self.assertEqual(
            result["false_positives"][0]["prediction"]["label"],
            "boat",
        )
        self.assertEqual(
            result["false_positives"][0]["nearest_same_class_iou"],
            1.0,
        )

    def test_subthreshold_overlap_is_not_a_match(self) -> None:
        result = match_detections(
            [truth("buoy", [0, 0, 10, 10], "b1")],
            [prediction("buoy", 0.9, [0, 0, 4.9, 10])],
            0.5,
        )
        self.assertEqual(len(result["true_positives"]), 0)
        self.assertEqual(len(result["false_positives"]), 1)
        self.assertEqual(len(result["false_negatives"]), 1)
        self.assertAlmostEqual(
            result["false_positives"][0]["nearest_same_class_iou"],
            0.49,
        )

    def test_selection_caps_categories_and_separates_frames(self) -> None:
        candidates = {category: [] for category in CATEGORY_ORDER}
        for frame, rank, identity in (
            (10, 1, "a"),
            (11, 2, "b"),
            (20, 3, "c"),
            (30, 4, "d"),
        ):
            candidates["tiny_misses"].append(
                {
                    "camera_frame": frame,
                    "_rank": (rank,),
                    "_identity": identity,
                }
            )
        selected = select_candidates(candidates, max_per_category=3, frame_separation=5)
        self.assertEqual(
            [item["camera_frame"] for item in selected["tiny_misses"]],
            [10, 20, 30],
        )
        self.assertEqual(
            [item["rank"] for item in selected["tiny_misses"]],
            [1, 2, 3],
        )
        self.assertTrue(all(len(selected[category]) <= 3 for category in CATEGORY_ORDER))

    def test_temporal_omission_requires_three_consecutive_false_positives(self) -> None:
        candidates = {category: [] for category in CATEGORY_ORDER}
        bounded_matches = {}
        for frame, x_offset in ((10, 0), (11, 1), (12, 2)):
            bounded_matches[frame] = {
                "false_positives": [
                    {
                        "prediction": prediction(
                            "boat",
                            0.3,
                            [100 + x_offset, 100, 140 + x_offset, 120],
                        ),
                        "nearest_same_class_iou": 0.0,
                    }
                ]
            }
        add_temporally_verified_omission_candidates(candidates, bounded_matches)
        self.assertEqual(len(candidates["ambiguous_omitted_objects"]), 1)
        self.assertEqual(
            candidates["ambiguous_omitted_objects"][0]["details"]["temporal_support_frames"],
            [10, 11, 12],
        )

    def test_crop_preserves_output_aspect_and_clamps_to_image(self) -> None:
        crop = enclosing_crop([[0, 0, 10, 10]], (5320, 3032))
        self.assertEqual(crop[:2], [0, 0])
        self.assertAlmostEqual((crop[2] - crop[0]) / (crop[3] - crop[1]), 1.6)
        self.assertLessEqual(crop[2], 5320)
        self.assertLessEqual(crop[3], 3032)

    def test_renderer_draws_boxes_without_text_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            destination = root / "evidence.png"
            Image.new("RGB", (800, 500), "white").save(source)
            item = {
                "focus_boxes": [[300, 200, 400, 300]],
                "annotations": [
                    {
                        "kind": "false_positive",
                        "bbox_xyxy": [300, 200, 400, 300],
                        "class_name": "boat",
                        "method": "vision_only_tiled",
                    }
                ],
            }
            render_candidate(source, item, destination)
            self.assertTrue(destination.is_file())
            self.assertEqual(item["output_size"], [1600, 1000])
            with Image.open(destination) as evidence:
                colors = evidence.getcolors(maxcolors=evidence.width * evidence.height)
            self.assertIsNotNone(colors)
            assert colors is not None
            self.assertIn(
                tuple(int(FP_COLOR[index : index + 2], 16) for index in (1, 3, 5)),
                {color for _, color in colors},
            )


if __name__ == "__main__":
    unittest.main()
