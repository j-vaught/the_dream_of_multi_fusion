from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "presentation/media/vision/build_vision_media.py"
SPEC = importlib.util.spec_from_file_location("build_vision_media", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT}")
vision_media = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vision_media
SPEC.loader.exec_module(vision_media)


class VisionMediaSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictions = vision_media.predictions_by_frame(vision_media.PREDICTIONS_JSONL)
        cls.ground_truth = vision_media.ground_truth_by_frame(vision_media.GROUND_TRUTH_JSONL)

    def test_verified_false_positive_frames_are_stable(self) -> None:
        false_positives = vision_media.find_false_positive_frames(
            self.ground_truth,
            self.predictions,
        )
        self.assertEqual(tuple(false_positives), (341, 416))
        self.assertTrue(
            all(
                detection["label"] == "boat"
                for detections in false_positives.values()
                for detection in detections
            )
        )

    def test_correct_frame_matches_boat_and_buoy_without_false_positives(self) -> None:
        matches, false_positives, _ = vision_media.match_detections(
            self.ground_truth[119],
            self.predictions[119],
        )
        self.assertEqual(false_positives, [])
        self.assertEqual(
            {match.prediction["label"] for match in matches},
            {"boat", "buoy"},
        )
        self.assertTrue(all(match.iou >= 0.5 for match in matches))

    def test_false_positive_timeline_uses_context_and_holds(self) -> None:
        timeline = vision_media.false_positive_timeline()
        self.assertEqual(len(timeline), 132)
        self.assertEqual(timeline.count(341), 45)
        self.assertEqual(timeline.count(416), 45)
        self.assertEqual(min(timeline), 337)
        self.assertEqual(max(timeline), 418)


class VisionMediaArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(vision_media.MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_records_required_style(self) -> None:
        style = self.manifest["style"]
        self.assertFalse(style["burned_in_text"])
        self.assertEqual(style["false_positive"]["color"], "#FF0000")
        self.assertEqual(style["predictions"]["boat"]["line_style"], "solid")
        self.assertEqual(style["predictions"]["buoy"]["line_style"], "solid")
        self.assertEqual(style["ground_truth"]["boat"]["line_style"], "dashed")
        self.assertEqual(style["ground_truth"]["buoy"]["line_style"], "dashed")
        self.assertEqual(style["ground_truth"]["boat"]["color"], "#FFFFFF")
        self.assertEqual(style["ground_truth"]["buoy"]["color"], "#FFFFFF")

    def test_assets_match_manifest_hashes_and_timing(self) -> None:
        vision_media.verify_manifest()
        correct = self.manifest["assets"]["correct_comparison"]["video"]["media"]
        self.assertEqual(correct["frame_count"], 300)
        self.assertEqual(correct["r_frame_rate"], "60/1")
        self.assertEqual(correct["duration_seconds"], 5.0)
        false_positive = self.manifest["assets"]["false_positive_episodes"]["video"]["media"]
        self.assertEqual(false_positive["frame_count"], 132)
        self.assertEqual(false_positive["r_frame_rate"], "60/1")
        self.assertEqual(false_positive["duration_seconds"], 2.2)

    def test_manifest_records_selected_frame_hashes(self) -> None:
        correct = self.manifest["assets"]["correct_comparison"]["selected_frame"]
        self.assertEqual(correct["camera_frame"], 119)
        self.assertEqual(len(correct["clean_source_rgb_sha256"]), 64)
        selected = self.manifest["assets"]["false_positive_episodes"]["selected_frames"]
        self.assertEqual(
            [record["camera_frame"] for record in selected],
            [337, 338, 339, 340, 341, 342, 343, 344, 345, 412, 413, 414, 415, 416, 417, 418],
        )
        self.assertTrue(all(len(record["zoom_decoded_rgb_sha256"]) == 64 for record in selected))


if __name__ == "__main__":
    unittest.main()
