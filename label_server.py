#!/usr/bin/env python3
"""Correction-first annotation server for the 300-frame Dream Fusion sequence."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parent
CLASS_IDS = {"boat": 0, "buoy": 1}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def class_for_track(label: str) -> str:
    return "boat" if label.lower().startswith("boat") else "buoy"


def normalize_box(values: list[float], width: int, height: int) -> list[float]:
    if len(values) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in values)
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise ValueError("bounding boxes must be at least one pixel wide and high")
    return [x1, y1, x2, y2]


@dataclass
class LabelStore:
    rgb_dir: Path
    preview_dir: Path | None
    seed_path: Path
    tracker_path: Path
    state_path: Path
    export_path: Path
    first_frame: int
    last_frame: int

    def __post_init__(self) -> None:
        self.lock = threading.RLock()
        self.frame_paths = {
            int(path.stem.split("_")[0]): path
            for path in self.rgb_dir.glob("*_rgb.png")
            if path.stem.split("_")[0].isdigit()
        }
        self.frames = [
            frame
            for frame in sorted(self.frame_paths)
            if self.first_frame <= frame <= self.last_frame
        ]
        if not self.frames:
            raise FileNotFoundError(f"No RGB frames found in {self.rgb_dir}")
        self.preview_paths = {}
        if self.preview_dir and self.preview_dir.exists():
            for index, frame in enumerate(self.frames):
                preview = self.preview_dir / f"{index:06d}.jpg"
                if preview.exists():
                    self.preview_paths[frame] = preview

        self.seed = json.loads(self.seed_path.read_text(encoding="utf-8"))
        image_size = self.seed["image_size"]
        self.width = int(image_size["width"])
        self.height = int(image_size["height"])
        self.tracker = self._load_tracker()
        self.state = self._load_or_initialize_state()

    def _load_tracker(self) -> dict[int, dict[str, dict[str, Any]]]:
        result: dict[int, dict[str, dict[str, Any]]] = {}
        if not self.tracker_path.exists():
            return result
        for line in self.tracker_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            result[int(record["camera_frame"])] = {
                str(track["label"]): track for track in record.get("tracks", [])
            }
        return result

    def reload_tracker(self) -> int:
        with self.lock:
            self.tracker = self._load_tracker()
            return len(self.tracker)

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") != "1.0":
                raise ValueError(f"Unsupported schema version in {self.state_path}")
            return state

        tracks = []
        seed_objects = []
        for box in self.seed["boxes"]:
            track_id = str(box["label"])
            class_name = class_for_track(track_id)
            tracks.append(
                {
                    "track_id": track_id,
                    "class_id": CLASS_IDS[class_name],
                    "class_name": class_name,
                    "display_name": track_id,
                    "color": box.get("color", "#CC2E40"),
                }
            )
            seed_objects.append(
                {
                    "track_id": track_id,
                    "bbox_xyxy": [float(value) for value in box["bbox_xyxy"]],
                    "visibility": "visible",
                }
            )

        state = {
            "schema_version": "1.0",
            "dataset": {
                "name": "dream_fusion_lake_murray_300",
                "image_size": {"width": self.width, "height": self.height},
                "first_camera_frame": self.frames[0],
                "last_camera_frame": self.frames[-1],
                "frame_count": len(self.frames),
            },
            "classes": [
                {"class_id": 0, "name": "boat"},
                {"class_id": 1, "name": "buoy"},
            ],
            "tracks": tracks,
            "keyframes": {
                str(self.frames[0]): {
                    "reviewed": False,
                    "note": "Imported seed boxes. Verify before accepting.",
                    "objects": seed_objects,
                }
            },
        }
        atomic_write_json(self.state_path, state)
        return state

    @property
    def track_definitions(self) -> dict[str, dict[str, Any]]:
        return {track["track_id"]: track for track in self.state["tracks"]}

    def baseline_box(self, frame: int, track_id: str) -> tuple[list[float] | None, float | None]:
        tracker_record = self.tracker.get(frame, {}).get(track_id)
        if tracker_record:
            return (
                [float(value) for value in tracker_record["bbox_xyxy"]],
                float(tracker_record.get("confidence", 1.0)),
            )
        seed_box = next(
            (box for box in self.seed["boxes"] if str(box["label"]) == track_id),
            None,
        )
        if seed_box:
            return [float(value) for value in seed_box["bbox_xyxy"]], None
        return None, None

    def latest_keyframe_object(
        self, frame: int, track_id: str
    ) -> tuple[int, dict[str, Any]] | None:
        candidates = []
        for frame_text, record in self.state["keyframes"].items():
            keyframe = int(frame_text)
            if keyframe > frame:
                continue
            for obj in record.get("objects", []):
                if obj["track_id"] == track_id:
                    candidates.append((keyframe, obj))
                    break
        return max(candidates, key=lambda item: item[0]) if candidates else None

    def resolve_object(self, frame: int, track_id: str) -> dict[str, Any] | None:
        definition = self.track_definitions.get(track_id)
        if not definition:
            return None
        keyframe_match = self.latest_keyframe_object(frame, track_id)
        baseline_now, confidence = self.baseline_box(frame, track_id)
        if keyframe_match is None and baseline_now is None:
            return None

        source_frame = None
        visibility = "visible"
        if keyframe_match:
            source_frame, manual = keyframe_match
            visibility = manual.get("visibility", "visible")
            manual_box = [float(value) for value in manual["bbox_xyxy"]]
            baseline_then, _ = self.baseline_box(source_frame, track_id)
            if baseline_now is not None and baseline_then is not None:
                box = [
                    manual_box[index] + baseline_now[index] - baseline_then[index]
                    for index in range(4)
                ]
            else:
                box = manual_box
        else:
            box = baseline_now

        if box is None:
            return None
        box = normalize_box(box, self.width, self.height)
        return {
            **definition,
            "bbox_xyxy": box,
            "visibility": visibility,
            "tracker_confidence": confidence,
            "source_frame": source_frame,
            "source": "manual" if source_frame == frame else "tracked",
        }

    def frame_payload(self, frame: int) -> dict[str, Any]:
        if frame not in self.frame_paths or frame not in self.frames:
            raise KeyError(frame)
        with self.lock:
            objects = [
                resolved
                for track_id in self.track_definitions
                if (resolved := self.resolve_object(frame, track_id)) is not None
            ]
            keyframe = self.state["keyframes"].get(str(frame), {})
            confidences = [
                obj["tracker_confidence"]
                for obj in objects
                if obj["tracker_confidence"] is not None
            ]
            return {
                "camera_frame": frame,
                "frame_index": self.frames.index(frame),
                "frame_count": len(self.frames),
                "previous_frame": self.frames[max(0, self.frames.index(frame) - 1)],
                "next_frame": self.frames[min(len(self.frames) - 1, self.frames.index(frame) + 1)],
                "image_size": {"width": self.width, "height": self.height},
                "reviewed": bool(keyframe.get("reviewed", False)),
                "note": keyframe.get("note", ""),
                "tracker_ready": frame in self.tracker,
                "preview_ready": frame in self.preview_paths,
                "minimum_tracker_confidence": min(confidences) if confidences else None,
                "objects": objects,
            }

    def save_frame(self, frame: int, payload: dict[str, Any]) -> dict[str, Any]:
        if frame not in self.frames:
            raise KeyError(frame)
        objects = payload.get("objects")
        if not isinstance(objects, list):
            raise ValueError("objects must be a list")

        known_tracks = self.track_definitions
        normalized = []
        seen = set()
        for obj in objects:
            track_id = str(obj.get("track_id", ""))
            if track_id not in known_tracks:
                raise ValueError(f"Unknown track_id {track_id!r}")
            if track_id in seen:
                raise ValueError(f"Duplicate track_id {track_id!r}")
            seen.add(track_id)
            normalized.append(
                {
                    "track_id": track_id,
                    "bbox_xyxy": normalize_box(obj["bbox_xyxy"], self.width, self.height),
                    "visibility": obj.get("visibility", "visible"),
                }
            )

        with self.lock:
            self.state["keyframes"][str(frame)] = {
                "reviewed": bool(payload.get("reviewed", True)),
                "note": str(payload.get("note", "")),
                "objects": normalized,
            }
            atomic_write_json(self.state_path, self.state)
            self.export()
        return self.frame_payload(frame)

    def export(self) -> dict[str, int]:
        reviewed_count = 0
        manual_count = 0
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.export_path.with_suffix(self.export_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for index, frame in enumerate(self.frames):
                payload = self.frame_payload(frame)
                reviewed_count += int(payload["reviewed"])
                objects = []
                for obj in payload["objects"]:
                    manual_count += int(obj["source"] == "manual")
                    objects.append(
                        {
                            "track_id": obj["track_id"],
                            "class_id": obj["class_id"],
                            "class_name": obj["class_name"],
                            "bbox_xyxy": obj["bbox_xyxy"],
                            "visibility": obj["visibility"],
                            "source": obj["source"],
                            "source_frame": obj["source_frame"],
                            "tracker_confidence": obj["tracker_confidence"],
                        }
                    )
                output.write(
                    json.dumps(
                        {
                            "frame_index": index,
                            "camera_frame": frame,
                            "reviewed": payload["reviewed"],
                            "objects": objects,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        temporary.replace(self.export_path)
        return {
            "frames": len(self.frames),
            "reviewed_frames": reviewed_count,
            "manual_objects": manual_count,
        }

    def progress(self) -> dict[str, Any]:
        reviewed = {
            int(frame)
            for frame, record in self.state["keyframes"].items()
            if record.get("reviewed")
        }
        tracker_frames = set(self.tracker)
        return {
            "frame_count": len(self.frames),
            "reviewed_count": len(reviewed),
            "tracker_frame_count": len(tracker_frames),
            "first_frame": self.frames[0],
            "last_frame": self.frames[-1],
        }

    def next_frame(self, frame: int, mode: str) -> int:
        start = self.frames.index(frame) + 1
        for candidate in self.frames[start:]:
            payload = self.frame_payload(candidate)
            if mode == "low_confidence":
                confidence = payload["minimum_tracker_confidence"]
                if confidence is not None and confidence < 0.5:
                    return candidate
            elif not payload["reviewed"]:
                return candidate
        return self.frames[-1]


def build_app(store: LabelStore) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return send_file(ROOT / "label.html")

    @app.get("/frame/<int:frame>")
    def frame_image(frame: int):
        if frame not in store.frame_paths or frame not in store.frames:
            return jsonify({"error": "frame not found"}), 404
        return send_from_directory(store.rgb_dir, store.frame_paths[frame].name, max_age=3600)

    @app.get("/preview/<int:frame>")
    def frame_preview(frame: int):
        preview = store.preview_paths.get(frame)
        if preview is None:
            return jsonify({"error": "preview not found"}), 404
        return send_from_directory(preview.parent, preview.name, max_age=86400)

    @app.get("/api/frame/<int:frame>")
    def frame_data(frame: int):
        try:
            return jsonify(store.frame_payload(frame))
        except KeyError:
            return jsonify({"error": "frame not found"}), 404

    @app.post("/api/frame/<int:frame>")
    def save_frame(frame: int):
        try:
            return jsonify(store.save_frame(frame, request.get_json(force=True) or {}))
        except (KeyError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

    @app.get("/api/progress")
    def progress():
        return jsonify(store.progress())

    @app.get("/api/next/<int:frame>")
    def next_frame(frame: int):
        mode = request.args.get("mode", "unreviewed")
        try:
            return jsonify({"camera_frame": store.next_frame(frame, mode)})
        except ValueError:
            return jsonify({"error": "frame not found"}), 404

    @app.post("/api/reload-tracker")
    def reload_tracker():
        return jsonify({"tracker_frame_count": store.reload_tracker()})

    @app.post("/api/export")
    def export():
        return jsonify(store.export())

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rgb-dir", type=Path, default=ROOT / "data" / "rgb_out")
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional sequential JPEG directory where 000000.jpg maps to --first-frame",
    )
    parser.add_argument("--seed", type=Path, default=ROOT / "annotations" / "initial_boxes.json")
    parser.add_argument(
        "--tracker",
        type=Path,
        default=ROOT / "out" / "experiments" / "tracker" / "lorat_baseline.jsonl",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "annotations" / "ground_truth_keyframes.json",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=ROOT / "annotations" / "ground_truth.jsonl",
    )
    parser.add_argument("--first-frame", type=int, default=119)
    parser.add_argument("--last-frame", type=int, default=418)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = LabelStore(
        rgb_dir=args.rgb_dir,
        preview_dir=args.preview_dir,
        seed_path=args.seed,
        tracker_path=args.tracker,
        state_path=args.state,
        export_path=args.export,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
    )
    app = build_app(store)
    print(
        f"Labeler ready at http://{args.host}:{args.port} "
        f"for CF {store.frames[0]} through {store.frames[-1]}"
    )
    print(f"Keyframes: {store.state_path}")
    print(f"Per-frame export: {store.export_path}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
