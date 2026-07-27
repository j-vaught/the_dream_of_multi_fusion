from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from export_yolo_dataset import export_dataset, parse_absence_rules, yolo_box
from label_server import LabelStore


class YoloDatasetExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rgb_dir = self.root / "rgb"
        rgb_dir.mkdir()
        for frame in range(119, 123):
            Image.new("RGB", (200, 100)).save(rgb_dir / f"{frame}_rgb.png")

        seed_path = self.root / "initial_boxes.json"
        seed_path.write_text(
            json.dumps(
                {
                    "camera_frame": 119,
                    "image_size": {"width": 200, "height": 100},
                    "boxes": [
                        {"label": "boat", "bbox_xyxy": [20, 10, 60, 30]},
                        {"label": "b2", "bbox_xyxy": [100, 40, 120, 60]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        tracker_path = self.root / "tracker.jsonl"
        tracker_path.write_text(
            "".join(
                json.dumps(
                    {
                        "camera_frame": frame,
                        "tracks": [
                            {
                                "label": "boat",
                                "bbox_xyxy": [20, 10, 60, 30],
                                "confidence": 0.9,
                            },
                            {
                                "label": "b2",
                                "bbox_xyxy": [100, 40, 120, 60],
                                "confidence": 0.8,
                            },
                        ],
                    }
                )
                + "\n"
                for frame in range(119, 123)
            ),
            encoding="utf-8",
        )
        state_path = self.root / "state.json"
        state_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "image_size": {"width": 200, "height": 100},
                    "frame_range": {"first": 119, "last": 122},
                    "tracks": [
                        {
                            "track_id": "boat",
                            "class_id": 0,
                            "class_name": "boat",
                            "display_name": "boat",
                            "color": "#FF4FA3",
                        },
                        {
                            "track_id": "b2",
                            "class_id": 1,
                            "class_name": "buoy",
                            "display_name": "b2",
                            "color": "#00FFFF",
                        },
                    ],
                    "keyframes": {
                        "119": {
                            "reviewed": True,
                            "note": "",
                            "objects": [
                                {
                                    "track_id": "boat",
                                    "bbox_xyxy": [20, 10, 60, 30],
                                    "visibility": "visible",
                                },
                                {
                                    "track_id": "b2",
                                    "bbox_xyxy": [100, 40, 120, 60],
                                    "visibility": "visible",
                                },
                            ],
                        },
                        "121": {
                            "reviewed": True,
                            "note": "",
                            "objects": [
                                {
                                    "track_id": "boat",
                                    "bbox_xyxy": [20, 10, 60, 30],
                                    "visibility": "visible",
                                },
                                {
                                    "track_id": "b2",
                                    "bbox_xyxy": [100, 40, 120, 60],
                                    "visibility": "absent",
                                },
                            ],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        self.store = LabelStore(
            rgb_dir=rgb_dir,
            preview_dir=None,
            seed_path=seed_path,
            tracker_path=tracker_path,
            state_path=state_path,
            export_path=self.root / "ground_truth.jsonl",
            first_frame=119,
            last_frame=122,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_yolo_box_is_normalized(self) -> None:
        self.assertEqual(yolo_box([20, 10, 60, 30], 200, 100), [0.2, 0.2, 0.2, 0.2])

    def test_export_enforces_cutoff_and_chronological_splits(self) -> None:
        output = self.root / "dataset"
        summary = export_dataset(
            store=self.store,
            output=output,
            image_mode="hardlink",
            train_fraction=0.5,
            val_fraction=0.25,
            absence_rules={"b2": 121},
        )
        self.assertEqual(summary["split_counts"], {"test": 1, "train": 2, "val": 1})
        self.assertEqual(summary["last_labeled_frame"]["b2"], 120)
        self.assertEqual((output / "labels" / "val" / "cf_000121.txt").read_text().count("\n"), 1)
        records = [
            json.loads(line)
            for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        late_b2 = [
            obj
            for record in records
            if record["camera_frame"] >= 121
            for obj in record["objects"]
            if obj["track_id"] == "b2"
        ]
        self.assertEqual(late_b2, [])

    def test_absence_rule_parser(self) -> None:
        self.assertEqual(parse_absence_rules(["b2:211"]), {"b2": 211})


if __name__ == "__main__":
    unittest.main()
