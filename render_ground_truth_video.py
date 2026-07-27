#!/usr/bin/env python3
"""Render corrected ground-truth boxes over full-resolution camera frames."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from label_server import ROOT, LabelStore

BUOY_COLOR = "#00FFFF"
BOAT_COLOR = "#FF4FA3"


def overlay_color(class_name: str) -> str:
    return BOAT_COLOR if class_name == "boat" else BUOY_COLOR


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
        default=ROOT / "out" / "videos" / "07_ground_truth_full_resolution.mp4",
    )
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--line-width", type=int, default=6)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Reuse an existing --frames-dir and run only the encoding step",
    )
    parser.add_argument(
        "--encoder",
        choices=("auto", "h264_nvenc", "libx264"),
        default="auto",
    )
    return parser.parse_args()


def choose_encoder(requested: str, width: int) -> str:
    if requested != "auto":
        return requested
    if width > 4096:
        return "libx264"
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    return "h264_nvenc" if "h264_nvenc" in result.stdout else "libx264"


def render_frames(store: LabelStore, frames_dir: Path, line_width: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = len(store.frames)
    for index, camera_frame in enumerate(store.frames):
        payload = store.frame_payload(camera_frame)
        with Image.open(store.frame_paths[camera_frame]) as source:
            image = source.convert("RGB")
        drawing = ImageDraw.Draw(image)
        for obj in payload["objects"]:
            if obj["visibility"] == "absent":
                continue
            box = tuple(round(value) for value in obj["bbox_xyxy"])
            drawing.rectangle(
                box,
                outline=overlay_color(obj["class_name"]),
                width=line_width,
            )
        image.save(
            frames_dir / f"{index:06d}.jpg",
            quality=95,
            subsampling=0,
        )
        if (index + 1) % 25 == 0 or index + 1 == total:
            print(f"Rendered {index + 1}/{total} frames", flush=True)


def encode_video(
    frames_dir: Path,
    output: Path,
    frame_count: int,
    fps: int,
    encoder: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%06d.jpg"),
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        encoder,
    ]
    if encoder == "h264_nvenc":
        command.extend(["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0"])
    else:
        command.extend(["-preset", "fast", "-crf", "18"])
    command.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)])
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir or args.output.parent / f"{args.output.stem}_frames"
    store = LabelStore(
        rgb_dir=args.rgb_dir,
        preview_dir=None,
        seed_path=args.seed,
        tracker_path=args.tracker,
        state_path=args.state,
        export_path=args.output.with_suffix(".jsonl"),
        first_frame=119,
        last_frame=418,
    )
    if not args.skip_render:
        render_frames(store, frames_dir, args.line_width)
    encoder = choose_encoder(args.encoder, store.width)
    print(f"Encoding {len(store.frames)} frames with {encoder}", flush=True)
    encode_video(frames_dir, args.output, len(store.frames), args.fps, encoder)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
