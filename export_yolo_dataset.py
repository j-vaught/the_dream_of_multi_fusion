#!/usr/bin/env python3
"""Export corrected Dream Fusion tracks as a YOLO object-detection dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from label_server import CLASS_IDS, ROOT, LabelStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, default=ROOT / "data" / "rgb_out")
    parser.add_argument(
        "--seed",
        type=Path,
        default=ROOT / "annotations" / "initial_boxes.json",
    )
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
        "--output",
        type=Path,
        default=ROOT / "out" / "datasets" / "dream_fusion_yolo",
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Hardlinks preserve full PNG quality without duplicating storage",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--assert-absent-after",
        action="append",
        default=[],
        metavar="TRACK:FRAME",
        help="Fail if TRACK is labeled at or after FRAME; may be repeated",
    )
    return parser.parse_args()


def parse_absence_rules(values: list[str]) -> dict[str, int]:
    rules = {}
    for value in values:
        try:
            track_id, frame_text = value.rsplit(":", 1)
            frame = int(frame_text)
        except ValueError as error:
            raise ValueError(f"Invalid absence rule {value!r}; expected TRACK:FRAME") from error
        if not track_id:
            raise ValueError(f"Invalid absence rule {value!r}; track ID is empty")
        rules[track_id] = frame
    return rules


def yolo_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def split_boundaries(
    frame_count: int,
    train_fraction: float,
    val_fraction: float,
) -> tuple[int, int]:
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("split fractions must satisfy train > 0, val >= 0, and train + val < 1")
    train_end = int(frame_count * train_fraction)
    val_end = train_end + int(frame_count * val_fraction)
    return train_end, val_end


def split_name(index: int, train_end: int, val_end: int) -> str:
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def place_image(source: Path, destination: Path, image_mode: str) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if image_mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_dataset_yaml(output: Path) -> None:
    class_names = [name for name, _ in sorted(CLASS_IDS.items(), key=lambda item: item[1])]
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    (output / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n",
        encoding="utf-8",
    )


def export_dataset(
    store: LabelStore,
    output: Path,
    image_mode: str,
    train_fraction: float,
    val_fraction: float,
    absence_rules: dict[str, int],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    train_end, val_end = split_boundaries(
        len(store.frames),
        train_fraction,
        val_fraction,
    )
    class_counts: Counter[str] = Counter()
    track_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    last_labeled_frame: dict[str, int] = {}
    manifest_records = []

    for index, camera_frame in enumerate(store.frames):
        split = split_name(index, train_end, val_end)
        stem = f"cf_{camera_frame:06d}"
        source_image = store.frame_paths[camera_frame]
        image_path = output / "images" / split / f"{stem}{source_image.suffix.lower()}"
        label_path = output / "labels" / split / f"{stem}.txt"
        place_image(source_image, image_path, image_mode)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        payload = store.frame_payload(camera_frame)
        label_lines = []
        manifest_objects = []
        for obj in payload["objects"]:
            if obj["visibility"] == "absent":
                continue
            track_id = str(obj["track_id"])
            cutoff = absence_rules.get(track_id)
            if cutoff is not None and camera_frame >= cutoff:
                raise ValueError(
                    f"{track_id} is visible at frame {camera_frame}, violating cutoff {cutoff}"
                )
            normalized = yolo_box(
                obj["bbox_xyxy"],
                payload["image_size"]["width"],
                payload["image_size"]["height"],
            )
            class_id = int(obj["class_id"])
            label_lines.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in normalized))
            class_counts[obj["class_name"]] += 1
            track_counts[track_id] += 1
            last_labeled_frame[track_id] = camera_frame
            manifest_objects.append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": obj["class_name"],
                    "bbox_xyxy": obj["bbox_xyxy"],
                    "bbox_yolo": normalized,
                    "source": obj["source"],
                    "source_frame": obj["source_frame"],
                    "tracker_confidence": obj["tracker_confidence"],
                }
            )
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        split_counts[split] += 1
        manifest_records.append(
            {
                "camera_frame": camera_frame,
                "split": split,
                "image": image_path.relative_to(output).as_posix(),
                "label": label_path.relative_to(output).as_posix(),
                "objects": manifest_objects,
            }
        )

    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in manifest_records),
        encoding="utf-8",
    )
    write_dataset_yaml(output)
    summary = {
        "schema_version": "1.0",
        "image_size": {"width": store.width, "height": store.height},
        "frame_count": len(store.frames),
        "split_method": "chronological",
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "track_counts": dict(sorted(track_counts.items())),
        "last_labeled_frame": dict(sorted(last_labeled_frame.items())),
        "absence_rules": absence_rules,
        "class_ids": CLASS_IDS,
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    absence_rules = parse_absence_rules(args.assert_absent_after)
    store = LabelStore(
        rgb_dir=args.rgb_dir,
        preview_dir=None,
        seed_path=args.seed,
        tracker_path=args.tracker,
        state_path=args.state,
        export_path=args.output / "ground_truth.jsonl",
        first_frame=119,
        last_frame=418,
    )
    summary = export_dataset(
        store=store,
        output=args.output,
        image_mode=args.image_mode,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        absence_rules=absence_rules,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
