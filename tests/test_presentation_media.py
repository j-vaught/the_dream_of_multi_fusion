from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import unittest
from fractions import Fraction
from pathlib import Path
from typing import Any

from presentation.media.base.build_presentation_media import (
    ASSETS,
    BUFFER_SIZE,
    FRAME_COUNT,
    MANIFEST_PATH,
    MAX_BIT_RATE,
    MAX_WIDTH,
    OUTPUT_DURATION_SECONDS,
    OUTPUT_FRAME_RATE,
    TARGET_BIT_RATE,
    mp4_atom_offsets,
    video_command,
)

BASE_DIR = MANIFEST_PATH.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration,bit_rate:"
                "stream=codec_name,codec_type,width,height,pix_fmt,"
                "r_frame_rate,avg_frame_rate,nb_frames"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


class PresentationMediaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("ffprobe") is None:
            raise unittest.SkipTest("ffprobe is required")
        if not MANIFEST_PATH.is_file():
            raise unittest.SkipTest("presentation media has not been generated")
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_asset_inventory_and_encoding_contract(self) -> None:
        self.assertEqual(
            [asset.asset_id for asset in ASSETS],
            [
                "ground_truth",
                "vision_only_predictions",
                "radar_gated_predictions",
                "radar_bounded_predictions",
                "zoom_vision_only",
                "zoom_radar_gated",
                "zoom_radar_bounded",
            ],
        )
        self.assertEqual(len(self.manifest["assets"]), 7)
        encoding = self.manifest["encoding"]
        self.assertEqual(encoding["frame_count"], FRAME_COUNT)
        self.assertEqual(encoding["frame_rate"], "60/1")
        self.assertEqual(encoding["duration_seconds"], 5.0)
        self.assertEqual(encoding["max_width"], MAX_WIDTH)
        self.assertEqual(encoding["target_bit_rate"], TARGET_BIT_RATE)
        self.assertEqual(encoding["max_bit_rate"], MAX_BIT_RATE)
        self.assertEqual(encoding["buffer_size"], BUFFER_SIZE)
        self.assertFalse(encoding["audio"])
        self.assertTrue(encoding["faststart"])

    def test_ffmpeg_command_has_no_overlay_or_audio(self) -> None:
        command = video_command(ASSETS[0], Path("/tmp/presentation-media-test.mp4"))
        rendered = " ".join(command)
        self.assertIn("settb=expr=1/60,setpts=N", rendered)
        self.assertIn("-frames:v 300", rendered)
        self.assertIn("-pix_fmt yuv420p", rendered)
        self.assertIn("-maxrate 6000000", rendered)
        self.assertIn("-movflags +faststart", rendered)
        self.assertIn("-an -sn -dn", rendered)
        self.assertNotIn("overlay", rendered)
        self.assertNotIn("drawtext", rendered)

    def test_generated_videos_match_presentation_contract(self) -> None:
        for entry in self.manifest["assets"]:
            with self.subTest(asset=entry["id"]):
                video_path = BASE_DIR / entry["video"]["path"]
                metadata = probe(video_path)
                video_streams = [
                    stream for stream in metadata["streams"] if stream["codec_type"] == "video"
                ]
                audio_streams = [
                    stream for stream in metadata["streams"] if stream["codec_type"] == "audio"
                ]
                self.assertEqual(len(video_streams), 1)
                self.assertEqual(audio_streams, [])

                stream = video_streams[0]
                self.assertEqual(stream["codec_name"], "h264")
                self.assertEqual(stream["pix_fmt"], "yuv420p")
                self.assertEqual(Fraction(stream["avg_frame_rate"]), OUTPUT_FRAME_RATE)
                self.assertEqual(int(stream["nb_frames"]), FRAME_COUNT)
                self.assertLessEqual(int(stream["width"]), MAX_WIDTH)
                self.assertEqual(
                    Fraction(metadata["format"]["duration"]),
                    OUTPUT_DURATION_SECONDS,
                )
                self.assertLessEqual(int(metadata["format"]["bit_rate"]), 6_500_000)

                atom_offsets = mp4_atom_offsets(video_path)
                self.assertLess(atom_offsets["moov"], atom_offsets["mdat"])

    def test_posters_match_video_dimensions(self) -> None:
        for entry in self.manifest["assets"]:
            with self.subTest(asset=entry["id"]):
                video = probe(BASE_DIR / entry["video"]["path"])
                poster = probe(BASE_DIR / entry["poster"]["path"])
                video_stream = video["streams"][0]
                poster_stream = poster["streams"][0]
                self.assertEqual(poster_stream["codec_name"], "png")
                self.assertEqual(
                    (poster_stream["width"], poster_stream["height"]),
                    (video_stream["width"], video_stream["height"]),
                )

    def test_manifest_hashes_match_files(self) -> None:
        for entry in self.manifest["assets"]:
            with self.subTest(asset=entry["id"]):
                for media_type in ("video", "poster"):
                    media = entry[media_type]
                    self.assertEqual(sha256(BASE_DIR / media["path"]), media["sha256"])


if __name__ == "__main__":
    unittest.main()
