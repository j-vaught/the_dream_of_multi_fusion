from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments/detection_comparison/experiment_manifest.json"


class _NmsIndices:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return self.values


def _load_recovered_pipeline():
    fake_numpy = types.ModuleType("numpy")
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = object()
    fake_torch.tensor = lambda values, dtype=None: values

    fake_ops = types.ModuleType("torchvision.ops")

    def fake_nms(boxes, scores, iou_threshold):
        del boxes, iou_threshold
        return _NmsIndices([max(range(len(scores)), key=scores.__getitem__)])

    fake_ops.nms = fake_nms
    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.ops = fake_ops

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForZeroShotObjectDetection = object
    fake_transformers.AutoProcessor = object

    fake_land = types.ModuleType("land_segmentation")
    fake_land.compute_water_mask = object()
    fake_land.load_segformer = object()
    fake_land.water_fraction = object()

    stubs = {
        "numpy": fake_numpy,
        "torch": fake_torch,
        "torchvision": fake_torchvision,
        "torchvision.ops": fake_ops,
        "transformers": fake_transformers,
        "land_segmentation": fake_land,
    }
    spec = importlib.util.spec_from_file_location(
        "_recovered_dream_multi_fusion",
        ROOT / "dream_multi_fusion.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load recovered pipeline")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class RadarBoundedRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = _load_recovered_pipeline()
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_recovered_artifact_hashes_and_row_count(self) -> None:
        for name, metadata in self.manifest["artifacts"].items():
            path = (
                ROOT / name
                if name.endswith(".py")
                else ROOT / "experiments/detection_comparison" / name
            )
            content = path.read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), metadata["sha256"])
            self.assertEqual(len(content), metadata["bytes"])

        detections_path = (
            ROOT / "experiments/detection_comparison/radar_bounded_full.jsonl"
        )
        rows = [
            json.loads(line)
            for line in detections_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 300)
        self.assertEqual([row["frame_idx"] for row in rows], list(range(300)))
        self.assertEqual(self.manifest["frame_range"]["row_count"], len(rows))

    def test_recovered_default_parameters_match_manifest(self) -> None:
        with patch.object(sys, "argv", ["dream_multi_fusion.py"]):
            args = self.pipeline.parse_args()
        parameters = self.manifest["parameters"]
        detector = parameters["detector"]
        self.assertEqual(args.model_id, detector["model_id"])
        self.assertEqual(args.dino_prompt, detector["prompt"])
        self.assertEqual(args.box_thresh, detector["box_threshold"])
        self.assertEqual(args.text_thresh, detector["text_threshold"])
        self.assertEqual(args.dino_short_edge, detector["resize"]["shortest_edge"])
        self.assertEqual(args.dino_long_edge, detector["resize"]["longest_edge"])
        self.assertEqual(
            args.buffer_px,
            parameters["radar_crop"]["padding_pixels_per_side"],
        )
        self.assertEqual(
            args.max_aspect_ratio,
            parameters["radar_crop"]["max_padded_width_to_height_ratio"],
        )
        self.assertEqual(
            args.seg_model,
            parameters["water_filter"]["model_id"],
        )
        self.assertEqual(
            args.water_thresh,
            parameters["water_filter"]["minimum_water_fraction"],
        )
        self.assertEqual(args.merge_iou, parameters["crop_merge"]["threshold"])
        self.assertEqual(args.nms_iou, parameters["nms"]["iou_threshold"])

    def test_water_threshold_is_inclusive_and_padding_is_clamped(self) -> None:
        detection = {"bbox_xyxy": [10, 20, 30, 40]}
        with patch.object(self.pipeline, "water_fraction", return_value=0.9):
            result = self.pipeline.classify_detection(
                detection,
                object(),
                100,
                100,
                15,
                0.9,
                2.1,
            )
        self.assertEqual(result["status"], "keep")
        self.assertEqual(result["padded_bbox"], [0, 5, 45, 55])

        with patch.object(self.pipeline, "water_fraction", return_value=0.899):
            result = self.pipeline.classify_detection(
                {"bbox_xyxy": [10, 20, 30, 40]},
                object(),
                100,
                100,
                15,
                0.9,
                2.1,
            )
        self.assertEqual(result["status"], "skip_land")

    def test_crop_merge_is_strict_and_transitive(self) -> None:
        detections = [
            {"padded_bbox": [0, 0, 10, 10]},
            {"padded_bbox": [5, 0, 15, 10]},
            {"padded_bbox": [10, 0, 20, 10]},
        ]
        self.assertEqual(len(self.pipeline.merge_keep_crops(detections, 0.3)), 1)
        self.assertEqual(len(self.pipeline.merge_keep_crops(detections, 0.5)), 3)

    def test_nms_is_class_aware_and_score_ordered(self) -> None:
        detections = [
            {"label": "boat", "score": 0.7, "bbox_xyxy": [0, 0, 10, 10]},
            {"label": "boat", "score": 0.9, "bbox_xyxy": [1, 1, 11, 11]},
            {"label": "buoy", "score": 0.8, "bbox_xyxy": [1, 1, 11, 11]},
        ]
        kept = self.pipeline.class_aware_nms(detections, 0.5)
        self.assertEqual([item["label"] for item in kept], ["boat", "buoy"])
        self.assertEqual([item["score"] for item in kept], [0.9, 0.8])


if __name__ == "__main__":
    unittest.main()
