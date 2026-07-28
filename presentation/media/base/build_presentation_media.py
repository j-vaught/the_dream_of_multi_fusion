#!/usr/bin/env python3
"""Build deterministic, presentation-ready copies of the experiment videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

FRAME_COUNT = 300
SOURCE_FRAME_RATE = Fraction(20, 1)
OUTPUT_FRAME_RATE = Fraction(60, 1)
OUTPUT_DURATION_SECONDS = Fraction(5, 1)
MAX_WIDTH = 1920
TARGET_BIT_RATE = 5_500_000
MAX_BIT_RATE = 6_000_000
BUFFER_SIZE = 12_000_000

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
MANIFEST_PATH = SCRIPT_DIR / "manifest.json"


@dataclass(frozen=True)
class Asset:
    asset_id: str
    source_name: str
    output_name: str

    @property
    def source_path(self) -> Path:
        return REPOSITORY_ROOT / self.source_name

    @property
    def output_path(self) -> Path:
        return SCRIPT_DIR / self.output_name

    @property
    def poster_path(self) -> Path:
        return SCRIPT_DIR / f"{Path(self.output_name).stem}.poster.png"


ASSETS = (
    Asset(
        "ground_truth",
        "07_ground_truth_full_resolution.mp4",
        "ground_truth.mp4",
    ),
    Asset(
        "vision_only_predictions",
        "08_exp1_vision_only_detections.mp4",
        "vision_only_predictions.mp4",
    ),
    Asset(
        "radar_gated_predictions",
        "09_exp2_radar_confidence_gated_detections.mp4",
        "radar_gated_predictions.mp4",
    ),
    Asset(
        "radar_bounded_predictions",
        "10_exp3_radar_bounded_detections.mp4",
        "radar_bounded_predictions.mp4",
    ),
    Asset(
        "zoom_vision_only",
        "11_zoom_exp1_vision_only.mp4",
        "zoom_vision_only.mp4",
    ),
    Asset(
        "zoom_radar_gated",
        "12_zoom_exp2_radar_gated.mp4",
        "zoom_radar_gated.mp4",
    ),
    Asset(
        "zoom_radar_bounded",
        "13_zoom_exp3_radar_bounded.mp4",
        "zoom_radar_bounded.mp4",
    ),
)

VIDEO_PROBE_ENTRIES = (
    "format=format_name,start_time,duration,size,bit_rate:"
    "stream=index,codec_name,codec_long_name,profile,codec_type,width,height,"
    "coded_width,coded_height,pix_fmt,level,r_frame_rate,avg_frame_rate,time_base,"
    "start_time,duration,bit_rate,nb_frames"
)


def executable(name: str) -> str:
    """Resolve a required executable or raise a useful error."""
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"Required executable is unavailable: {name}")
    return resolved


def run(command: Sequence[str]) -> None:
    """Run a command and fail with its stderr attached."""
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}\n"
            f"{rendered}\n{completed.stderr.strip()}"
        )


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    """Read stable stream and container metadata as JSON."""
    command = [
        executable("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        VIDEO_PROBE_ENTRIES,
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def tool_version(name: str) -> str:
    """Return the first version line for a media executable."""
    completed = subprocess.run(
        [executable(name), "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()[0]


def video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    """Return the sole video stream from probe output."""
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream, found {len(streams)}")
    return streams[0]


def validate_source(asset: Asset, probe: dict[str, Any]) -> None:
    """Ensure an input has the expected frame count, rate, and duration."""
    stream = video_stream(probe)
    frame_rate = Fraction(stream["avg_frame_rate"])
    frame_count = int(stream["nb_frames"])
    duration = Fraction(probe["format"]["duration"])
    if frame_rate != SOURCE_FRAME_RATE:
        raise ValueError(
            f"{asset.source_name} has frame rate {frame_rate}, expected {SOURCE_FRAME_RATE}"
        )
    if frame_count != FRAME_COUNT:
        raise ValueError(f"{asset.source_name} has {frame_count} frames, expected {FRAME_COUNT}")
    if duration != Fraction(15, 1):
        raise ValueError(f"{asset.source_name} has duration {duration}, expected 15 seconds")


def video_command(asset: Asset, destination: Path) -> list[str]:
    """Construct the deterministic FFmpeg command for one presentation clip."""
    scale_and_retime = f"scale=w='min({MAX_WIDTH},iw)':h=-2:flags=lanczos,settb=expr=1/60,setpts=N"
    return [
        executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(asset.source_path),
        "-map",
        "0:v:0",
        "-vf",
        scale_and_retime,
        "-frames:v",
        str(FRAME_COUNT),
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "60/1",
        "-fps_mode",
        "cfr",
        "-b:v",
        str(TARGET_BIT_RATE),
        "-maxrate",
        str(MAX_BIT_RATE),
        "-bufsize",
        str(BUFFER_SIZE),
        "-g",
        "120",
        "-keyint_min",
        "120",
        "-sc_threshold",
        "0",
        "-threads",
        "1",
        "-x264-params",
        "nal-hrd=vbr:force-cfr=1:threads=1",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-metadata",
        "encoder=",
        "-metadata:s:v:0",
        "encoder=",
        "-video_track_timescale",
        "60000",
        str(destination),
    ]


def poster_command(video_path: Path, destination: Path) -> list[str]:
    """Construct the deterministic FFmpeg command for a first-frame poster."""
    return [
        executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "png",
        "-compression_level",
        "9",
        "-pred",
        "mixed",
        "-threads",
        "1",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        str(destination),
    ]


def mp4_atom_offsets(path: Path) -> dict[str, int]:
    """Return byte offsets for top-level MP4 atoms."""
    offsets: dict[str, int] = {}
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        offset = 0
        while offset + 8 <= file_size:
            handle.seek(offset)
            header = handle.read(8)
            size = int.from_bytes(header[:4], "big")
            atom_type = header[4:8].decode("ascii", errors="replace")
            header_size = 8
            if size == 1:
                extended_size = handle.read(8)
                if len(extended_size) != 8:
                    break
                size = int.from_bytes(extended_size, "big")
                header_size = 16
            elif size == 0:
                size = file_size - offset
            if size < header_size or offset + size > file_size:
                break
            offsets.setdefault(atom_type, offset)
            offset += size
    return offsets


def validate_output(path: Path, probe: dict[str, Any]) -> dict[str, Any]:
    """Validate presentation media requirements and report fast-start status."""
    stream = video_stream(probe)
    format_metadata = probe["format"]
    atom_offsets = mp4_atom_offsets(path)
    faststart = (
        "moov" in atom_offsets
        and "mdat" in atom_offsets
        and atom_offsets["moov"] < atom_offsets["mdat"]
    )
    checks = {
        "codec": stream["codec_name"] == "h264",
        "pixel_format": stream["pix_fmt"] == "yuv420p",
        "frame_count": int(stream["nb_frames"]) == FRAME_COUNT,
        "frame_rate": Fraction(stream["avg_frame_rate"]) == OUTPUT_FRAME_RATE,
        "duration": Fraction(format_metadata["duration"]) == OUTPUT_DURATION_SECONDS,
        "max_width": int(stream["width"]) <= MAX_WIDTH,
        "bit_rate": int(format_metadata["bit_rate"]) <= 6_500_000,
        "faststart": faststart,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{path.name} failed validation checks: {', '.join(failed)}")
    return {
        "checks": checks,
        "mp4_atom_offsets": {
            "moov": atom_offsets["moov"],
            "mdat": atom_offsets["mdat"],
        },
    }


def temporary_path(destination: Path) -> Path:
    """Return a same-directory temporary path with the destination suffix."""
    return destination.with_name(f".{destination.stem}.tmp{destination.suffix}")


def build_asset(asset: Asset) -> dict[str, Any]:
    """Build and describe one video and poster pair."""
    if not asset.source_path.is_file():
        raise FileNotFoundError(f"Missing source video: {asset.source_path}")

    source_probe = ffprobe(asset.source_path)
    validate_source(asset, source_probe)

    video_tmp = temporary_path(asset.output_path)
    poster_tmp = temporary_path(asset.poster_path)
    video_tmp.unlink(missing_ok=True)
    poster_tmp.unlink(missing_ok=True)

    print(f"Encoding {asset.asset_id} from {asset.source_name}", flush=True)
    run(video_command(asset, video_tmp))
    run(poster_command(video_tmp, poster_tmp))

    video_probe = ffprobe(video_tmp)
    validation = validate_output(video_tmp, video_probe)
    poster_probe = ffprobe(poster_tmp)
    poster_stream = video_stream(poster_probe)
    output_stream = video_stream(video_probe)
    if poster_stream["codec_name"] != "png":
        raise ValueError(f"{poster_tmp.name} is not a PNG")
    if (poster_stream["width"], poster_stream["height"]) != (
        output_stream["width"],
        output_stream["height"],
    ):
        raise ValueError(f"{poster_tmp.name} dimensions do not match its video")

    video_tmp.replace(asset.output_path)
    poster_tmp.replace(asset.poster_path)

    return {
        "id": asset.asset_id,
        "source": {
            "path": str(asset.source_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256(asset.source_path),
            "ffprobe": source_probe,
        },
        "video": {
            "path": asset.output_path.name,
            "sha256": sha256(asset.output_path),
            "ffprobe": video_probe,
            **validation,
        },
        "poster": {
            "path": asset.poster_path.name,
            "sha256": sha256(asset.poster_path),
            "ffprobe": poster_probe,
        },
    }


def build_manifest(asset_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a stable manifest without wall-clock-dependent fields."""
    return {
        "schema_version": 1,
        "toolchain": {
            "ffmpeg": tool_version("ffmpeg"),
            "ffprobe": tool_version("ffprobe"),
        },
        "encoding": {
            "frame_count": FRAME_COUNT,
            "frame_rate": f"{OUTPUT_FRAME_RATE.numerator}/{OUTPUT_FRAME_RATE.denominator}",
            "duration_seconds": float(OUTPUT_DURATION_SECONDS),
            "video_codec": "libx264",
            "pixel_format": "yuv420p",
            "audio": False,
            "max_width": MAX_WIDTH,
            "scale_filter": "lanczos",
            "target_bit_rate": TARGET_BIT_RATE,
            "max_bit_rate": MAX_BIT_RATE,
            "buffer_size": BUFFER_SIZE,
            "faststart": True,
            "poster_frame": 0,
        },
        "assets": asset_entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        action="append",
        choices=[asset.asset_id for asset in ASSETS],
        help="Build only the selected asset. May be supplied more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_ids = set(args.asset or ())
    selected_assets = [
        asset for asset in ASSETS if not selected_ids or asset.asset_id in selected_ids
    ]

    entries_by_id: dict[str, dict[str, Any]] = {}
    if selected_ids and MANIFEST_PATH.is_file():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entries_by_id = {entry["id"]: entry for entry in existing.get("assets", [])}

    for asset in selected_assets:
        entries_by_id[asset.asset_id] = build_asset(asset)

    missing_ids = [asset.asset_id for asset in ASSETS if asset.asset_id not in entries_by_id]
    if missing_ids:
        raise RuntimeError(
            "A complete manifest requires all assets. Missing: " + ", ".join(missing_ids)
        )

    ordered_entries = [entries_by_id[asset.asset_id] for asset in ASSETS]
    manifest = build_manifest(ordered_entries)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    MANIFEST_PATH.write_text(manifest_text, encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
