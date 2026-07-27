from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from label_server import LabelStore, build_app


class LabelStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rgb_dir = self.root / "rgb"
        self.rgb_dir.mkdir()
        for frame in range(119, 122):
            (self.rgb_dir / f"{frame}_rgb.png").touch()

        self.seed_path = self.root / "initial_boxes.json"
        self.seed_path.write_text(
            json.dumps(
                {
                    "camera_frame": 119,
                    "image_size": {"width": 100, "height": 80},
                    "boxes": [
                        {
                            "label": "boat",
                            "color": "#65780B",
                            "bbox_xyxy": [10, 10, 30, 30],
                        },
                        {
                            "label": "b1",
                            "color": "#CC2E40",
                            "bbox_xyxy": [50, 20, 55, 30],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.tracker_path = self.root / "tracker.jsonl"
        tracker_records = []
        for index, frame in enumerate(range(119, 122)):
            tracker_records.append(
                {
                    "frame_idx": index,
                    "camera_frame": frame,
                    "tracks": [
                        {
                            "label": "boat",
                            "bbox_xyxy": [10 + index, 10, 30 + index, 30],
                            "confidence": 0.9 - index * 0.1,
                        },
                        {
                            "label": "b1",
                            "bbox_xyxy": [50, 20, 55, 30],
                            "confidence": 0.8,
                        },
                    ],
                }
            )
        self.tracker_path.write_text(
            "".join(json.dumps(record) + "\n" for record in tracker_records),
            encoding="utf-8",
        )
        self.state_path = self.root / "keyframes.json"
        self.export_path = self.root / "ground_truth.jsonl"
        self.store = LabelStore(
            rgb_dir=self.rgb_dir,
            seed_path=self.seed_path,
            tracker_path=self.tracker_path,
            state_path=self.state_path,
            export_path=self.export_path,
            first_frame=119,
            last_frame=121,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_tracker_motion_is_applied_after_seed_frame(self) -> None:
        payload = self.store.frame_payload(121)
        boat = next(obj for obj in payload["objects"] if obj["track_id"] == "boat")
        self.assertEqual(boat["bbox_xyxy"], [12.0, 10.0, 32.0, 30.0])
        self.assertEqual(boat["source_frame"], 119)
        self.assertEqual(boat["source"], "tracked")

    def test_manual_correction_offsets_later_tracker_boxes(self) -> None:
        frame_120 = self.store.frame_payload(120)
        objects = frame_120["objects"]
        boat = next(obj for obj in objects if obj["track_id"] == "boat")
        boat["bbox_xyxy"] = [20, 20, 40, 40]
        self.store.save_frame(
            120,
            {
                "reviewed": True,
                "objects": [
                    {
                        "track_id": obj["track_id"],
                        "bbox_xyxy": obj["bbox_xyxy"],
                        "visibility": obj["visibility"],
                    }
                    for obj in objects
                ],
            },
        )
        later = self.store.frame_payload(121)
        later_boat = next(obj for obj in later["objects"] if obj["track_id"] == "boat")
        self.assertEqual(later_boat["bbox_xyxy"], [21.0, 20.0, 41.0, 40.0])
        self.assertEqual(later_boat["source_frame"], 120)

    def test_save_exports_complete_jsonl(self) -> None:
        payload = self.store.frame_payload(119)
        self.store.save_frame(
            119,
            {
                "reviewed": True,
                "objects": [
                    {
                        "track_id": obj["track_id"],
                        "bbox_xyxy": obj["bbox_xyxy"],
                        "visibility": obj["visibility"],
                    }
                    for obj in payload["objects"]
                ],
            },
        )
        records = [
            json.loads(line) for line in self.export_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 3)
        self.assertTrue(records[0]["reviewed"])
        self.assertEqual(records[-1]["camera_frame"], 121)
        self.assertEqual({obj["class_name"] for obj in records[0]["objects"]}, {"boat", "buoy"})

    def test_api_serves_and_saves_frame_data(self) -> None:
        client = build_app(self.store).test_client()
        response = client.get("/api/frame/119")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        response = client.post(
            "/api/frame/119",
            json={
                "reviewed": True,
                "objects": [
                    {
                        "track_id": obj["track_id"],
                        "bbox_xyxy": obj["bbox_xyxy"],
                        "visibility": obj["visibility"],
                    }
                    for obj in payload["objects"]
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["reviewed"])


if __name__ == "__main__":
    unittest.main()
