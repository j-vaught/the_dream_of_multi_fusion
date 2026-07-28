from __future__ import annotations

import json
import unittest
from pathlib import Path

from presentation.media.gate.generate_gate_media import (
    ASSET_FILES,
    FINAL_PREDICTIONS,
    GLOBAL_THRESHOLD,
    GROUND_TRUTH,
    MINIMUM_IOA,
    OUTPUT_DIR,
    POSTER_SOURCE,
    RADAR_THRESHOLD,
    SHARDS,
    categorize_detections,
    choose_false_positive_episode,
    gate_decision,
    read_jsonl,
    select_examples,
    sha256_file,
    support_reason,
)
from run_detection_experiments import radar_supported


class GateLogicTests(unittest.TestCase):
    def test_threshold_boundaries_are_inclusive(self) -> None:
        self.assertEqual(
            gate_decision(GLOBAL_THRESHOLD, supported=False),
            "accepted_global",
        )
        self.assertEqual(
            gate_decision(RADAR_THRESHOLD, supported=True),
            "accepted_radar",
        )
        self.assertEqual(
            gate_decision(RADAR_THRESHOLD, supported=False),
            "rejected",
        )
        self.assertEqual(
            gate_decision(RADAR_THRESHOLD - 0.000001, supported=True),
            "rejected",
        )

    def test_center_ioa_and_touching_match_experiment_logic(self) -> None:
        regions = [[100.0, 100.0, 200.0, 200.0]]
        center_box = [120.0, 120.0, 160.0, 160.0]
        quarter_ioa_box = [180.0, 100.0, 260.0, 200.0]
        touching_box = [200.0, 100.0, 280.0, 200.0]

        self.assertEqual(
            support_reason(center_box, regions),
            ("center_inside", 0, 1.0),
        )
        reason, region_index, ioa = support_reason(quarter_ioa_box, regions)
        self.assertEqual(reason, "minimum_ioa")
        self.assertEqual(region_index, 0)
        self.assertAlmostEqual(ioa, MINIMUM_IOA)
        self.assertEqual(
            support_reason(touching_box, regions),
            (None, None, 0.0),
        )

        self.assertTrue(radar_supported(center_box, regions, MINIMUM_IOA))
        self.assertTrue(radar_supported(quarter_ioa_box, regions, MINIMUM_IOA))
        self.assertFalse(radar_supported(touching_box, regions, MINIMUM_IOA))

    def test_committed_examples_are_deterministic(self) -> None:
        records = []
        for path in SHARDS:
            records.extend(read_jsonl(path))
        examples = select_examples(records)

        self.assertEqual(examples["high"]["camera_frame"], 119)
        self.assertEqual(examples["center"]["camera_frame"], 138)
        self.assertEqual(examples["low"]["camera_frame"], 119)
        self.assertEqual(examples["center"]["support_reason"], "center_inside")
        self.assertGreaterEqual(
            examples["center"]["detection"]["score"],
            RADAR_THRESHOLD,
        )
        self.assertLess(
            examples["center"]["detection"]["score"],
            GLOBAL_THRESHOLD,
        )
        self.assertLess(examples["low"]["detection"]["score"], RADAR_THRESHOLD)

    def test_false_positive_episode_selection_is_deterministic(self) -> None:
        ground_truth = {
            int(record["camera_frame"]): record["objects"] for record in read_jsonl(GROUND_TRUTH)
        }
        predictions = {
            int(record["camera_frame"]): record["detections"]
            for record in read_jsonl(FINAL_PREDICTIONS)
        }
        self.assertEqual(
            choose_false_positive_episode(ground_truth, predictions),
            [393, 394, 395, 396],
        )

        for camera_frame in (394, 395):
            _, false_positives, _ = categorize_detections(
                ground_truth[camera_frame],
                predictions[camera_frame],
            )
            self.assertEqual(len(false_positives), 1)


class GateAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = OUTPUT_DIR / "asset_manifest.json"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_manifest_covers_required_assets_and_hashes(self) -> None:
        records = {record["path"]: record for record in self.manifest["assets"]}
        self.assertEqual(set(records), set(ASSET_FILES))
        for filename in ASSET_FILES:
            path = OUTPUT_DIR / filename
            self.assertTrue(path.is_file())
            self.assertEqual(records[filename]["sha256"], sha256_file(path))
            self.assertFalse(records[filename]["burned_in_text"])
            self.assertEqual(records[filename]["width"], 1920)
            self.assertEqual(records[filename]["height"], 1080)

    def test_videos_use_presentation_encoding(self) -> None:
        records = {record["path"]: record for record in self.manifest["assets"]}
        for filename in (
            "gate_acceptance_animation.mp4",
            "correct_predictions_over_dashed_ground_truth.mp4",
        ):
            record = records[filename]
            self.assertEqual(record["codec"], "h264")
            self.assertEqual(record["r_frame_rate"], "60/1")
            self.assertEqual(record["avg_frame_rate"], "60/1")
            self.assertEqual(record["frame_count"], 180)
            self.assertEqual(record["pixel_format"], "yuv420p")
            self.assertAlmostEqual(record["duration_seconds"], 3.0)

    def test_manifest_records_geometry_and_frame_provenance(self) -> None:
        selections = self.manifest["selections"]
        self.assertEqual(
            selections["center_inside"]["camera_frame"],
            138,
        )
        self.assertEqual(
            selections["center_inside"]["support_reason"],
            "center_inside",
        )
        self.assertEqual(
            selections["ioa_geometry"],
            {
                "accepted_candidate_area_fraction": 0.25,
                "accepted_center_inside": False,
                "provenance": "geometry_only",
                "touching_candidate_area_fraction": 0.0,
                "touching_center_inside": False,
            },
        )
        self.assertEqual(
            selections["false_positive_episode"],
            [393, 394, 395, 396],
        )
        self.assertEqual(
            selections["correct_prediction_sequence"],
            list(range(119, 179)),
        )

    def test_source_hashes_are_current(self) -> None:
        for source in self.manifest["sources"]:
            path = OUTPUT_DIR.parents[2] / source["path"]
            self.assertEqual(source["sha256"], sha256_file(path))

    def test_existing_prediction_only_video_is_reference_only(self) -> None:
        reference = self.manifest["existing_prediction_only_base_video"]
        self.assertFalse(reference["generated_or_duplicated"])
        asset_paths = {record["path"] for record in self.manifest["assets"]}
        self.assertNotIn(Path(reference["path"]).name, asset_paths)

    def test_cetz_source_contains_no_text_drawing(self) -> None:
        source = POSTER_SOURCE.read_text(encoding="utf-8")
        self.assertIn("@preview/cetz:0.4.2", source)
        self.assertNotIn("text(", source)


if __name__ == "__main__":
    unittest.main()
