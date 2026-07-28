#!/usr/bin/env python3
"""Generate deterministic frame-119 presentation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

FRAME_NUMBER = 119
EXPECTED_SIZE = (5320, 3032)
CROP_ORDER = ("boat", "b1", "b2", "b3", "b6", "b4", "b5", "b7", "b8")
BUOY_COLOR = "#00FFFF"
BOAT_COLOR = "#FF4FA3"
LINE_WIDTH = 6
BOAT_CROP_SIZE = (256, 160)
BUOY_CROP_SIZE = (128, 128)
REMOTE_SOURCE = "comech-2422:/home/j-vaught/dream_fusion/data/rgb_out/119_rgb.png"
VIDEO_SPECS = (
    ("ground_truth", "07_ground_truth_full_resolution.mp4"),
    ("experiment_1_vision_only", "08_exp1_vision_only_detections.mp4"),
    ("experiment_2_radar_confidence_gated", "09_exp2_radar_confidence_gated_detections.mp4"),
    ("experiment_3_radar_bounded", "10_exp3_radar_bounded_detections.mp4"),
)


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path, root: Path) -> str:
    """Return a stable POSIX path, relative to the repository when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def integer_box(box: Sequence[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Enclose a floating-point annotation in integer image coordinates."""
    if len(box) != 4:
        raise ValueError(f"Expected four box coordinates, received {len(box)}")
    width, height = image_size
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Box lies outside the image or has no area: {box}")
    return (
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(width - 1, math.ceil(x2)),
        min(height - 1, math.ceil(y2)),
    )


def centered_crop_box(
    box: Sequence[float],
    crop_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return a fixed-size native-pixel crop centered on a box."""
    crop_width, crop_height = crop_size
    image_width, image_height = image_size
    if crop_width > image_width or crop_height > image_height:
        raise ValueError("Crop dimensions exceed the source image dimensions")
    x1, y1, x2, y2 = (float(value) for value in box)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = round(center_x - crop_width / 2)
    top = round(center_y - crop_height / 2)
    left = min(max(0, left), image_width - crop_width)
    top = min(max(0, top), image_height - crop_height)
    return (left, top, left + crop_width, top + crop_height)


def save_png(image: Image.Image, path: Path) -> None:
    """Write a PNG with stable encoder settings and no ancillary metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG", optimize=False, compress_level=9)
    temporary.replace(path)


def png_record(path: Path, root: Path, role: str) -> dict[str, Any]:
    """Describe a generated PNG."""
    with Image.open(path) as image:
        size = list(image.size)
        mode = image.mode
    return {
        "path": relative_path(path, root),
        "role": role,
        "sha256": sha256_file(path),
        "width": size[0],
        "height": size[1],
        "mode": mode,
    }


def load_annotations(path: Path, image_size: tuple[int, int]) -> dict[str, dict[str, Any]]:
    """Load and validate the nine frame-119 annotations."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("camera_frame") != FRAME_NUMBER:
        raise ValueError(f"Expected camera frame {FRAME_NUMBER}")
    declared_size = payload.get("image_size", {})
    if (declared_size.get("width"), declared_size.get("height")) != image_size:
        raise ValueError("Annotation dimensions do not match the source image")
    boxes = payload.get("boxes", [])
    by_label = {item["label"]: item for item in boxes}
    if len(by_label) != len(boxes):
        raise ValueError("Annotation labels must be unique")
    if set(by_label) != set(CROP_ORDER):
        raise ValueError(f"Expected labels {CROP_ORDER}, received {tuple(by_label)}")
    for item in boxes:
        integer_box(item["bbox_xyxy"], image_size)
    return by_label


def extract_poster(video: Path, output: Path) -> None:
    """Decode the first video frame to a deterministic lossless PNG."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.png")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "png",
        "-compression_level",
        "9",
        "-pred",
        "mixed",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(output)


def generate(
    *,
    source: Path,
    annotations: Path,
    output_dir: Path,
    repository_root: Path,
    poster_videos: Mapping[str, Path],
    source_reference: str = REMOTE_SOURCE,
) -> dict[str, Any]:
    """Generate the annotated frame, native-pixel crops, posters, and manifest."""
    with Image.open(source) as opened:
        source_image = opened.convert("RGB")
    image_size = source_image.size
    annotation_by_label = load_annotations(annotations, image_size)

    annotated = source_image.copy()
    drawing = ImageDraw.Draw(annotated)
    rendered_boxes: list[dict[str, Any]] = []
    for label in CROP_ORDER:
        item = annotation_by_label[label]
        rendered = integer_box(item["bbox_xyxy"], image_size)
        color = BOAT_COLOR if label == "boat" else BUOY_COLOR
        drawing.rectangle(rendered, outline=color, width=LINE_WIDTH)
        rendered_boxes.append(
            {
                "label": label,
                "class": "boat" if label == "boat" else "buoy",
                "annotation_bbox_xyxy": item["bbox_xyxy"],
                "rendered_bbox_xyxy": list(rendered),
                "color": color,
            }
        )

    full_frame_path = output_dir / "frame_119_boxes.png"
    save_png(annotated, full_frame_path)
    outputs = [png_record(full_frame_path, repository_root, "annotated_full_frame")]

    crops: list[dict[str, Any]] = []
    for index, label in enumerate(CROP_ORDER, start=1):
        item = annotation_by_label[label]
        crop_size = BOAT_CROP_SIZE if label == "boat" else BUOY_CROP_SIZE
        coordinates = centered_crop_box(item["bbox_xyxy"], crop_size, image_size)
        crop_path = output_dir / "crops" / f"{index:02d}_{label}.png"
        save_png(source_image.crop(coordinates), crop_path)
        record = png_record(crop_path, repository_root, "native_resolution_object_crop")
        outputs.append(record)
        crops.append(
            {
                "order": index,
                "label": label,
                "class": "boat" if label == "boat" else "buoy",
                "source_bbox_xyxy": item["bbox_xyxy"],
                "crop_xyxy": list(coordinates),
                "width": crop_size[0],
                "height": crop_size[1],
                "output": record["path"],
                "sha256": record["sha256"],
            }
        )

    posters: list[dict[str, Any]] = []
    for key, video in poster_videos.items():
        if not video.is_file():
            raise FileNotFoundError(video)
        poster_path = output_dir / "posters" / f"{key}_frame_119.png"
        extract_poster(video, poster_path)
        record = png_record(poster_path, repository_root, "video_poster_frame")
        if (record["width"], record["height"]) != image_size:
            raise ValueError(f"Poster dimensions do not match the source for {video}")
        outputs.append(record)
        posters.append(
            {
                "key": key,
                "camera_frame": FRAME_NUMBER,
                "video_frame_index": 0,
                "video": relative_path(video, repository_root),
                "video_sha256": sha256_file(video),
                "output": record["path"],
                "sha256": record["sha256"],
            }
        )

    manifest = {
        "schema_version": 1,
        "camera_frame": FRAME_NUMBER,
        "source": {
            "path": relative_path(source, repository_root),
            "reference": source_reference,
            "sha256": sha256_file(source),
            "width": image_size[0],
            "height": image_size[1],
            "mode": "RGB",
        },
        "annotations": {
            "path": relative_path(annotations, repository_root),
            "sha256": sha256_file(annotations),
        },
        "rendering": {
            "boat_color": BOAT_COLOR,
            "buoy_color": BUOY_COLOR,
            "line_width": LINE_WIDTH,
            "corner_style": "square",
            "labels_or_text": False,
            "boxes": rendered_boxes,
        },
        "crop_order": list(CROP_ORDER),
        "crop_policy": {
            "pixels_are_resampled": False,
            "boat_size": list(BOAT_CROP_SIZE),
            "buoy_size": list(BUOY_CROP_SIZE),
            "boundary_behavior": "shift_window_to_remain_inside_source",
        },
        "crops": crops,
        "posters": posters,
        "outputs": outputs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repository_root / "presentation" / "stills" / "source" / "119_rgb.png",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=repository_root / "annotations" / "initial_boxes.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "presentation" / "stills",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the repository's frame-119 evidence assets."""
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    videos = {key: repository_root / filename for key, filename in VIDEO_SPECS}
    manifest = generate(
        source=args.source,
        annotations=args.annotations,
        output_dir=args.output_dir,
        repository_root=repository_root,
        poster_videos=videos,
    )
    print(
        f"Generated {len(manifest['outputs'])} PNG assets and {args.output_dir / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
