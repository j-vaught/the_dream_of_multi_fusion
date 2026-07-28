from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops, ImageColor

from presentation.stills.generate import (
    BOAT_COLOR,
    BUOY_COLOR,
    CROP_ORDER,
    centered_crop_box,
    generate,
    sha256_file,
)


class PresentationStillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository_root = Path(__file__).resolve().parents[1]

    def test_centered_crop_shifts_at_image_edges_without_resampling(self) -> None:
        crop = centered_crop_box((395.0, 145.0, 399.0, 155.0), (128, 128), (400, 300))
        self.assertEqual(crop, (272, 86, 400, 214))

    def test_fixture_generation_is_deterministic_and_uses_native_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            annotations = root / "annotations.json"
            output = root / "stills"
            image = Image.new("RGB", (400, 300))
            pixels = image.load()
            assert pixels is not None
            for y in range(image.height):
                for x in range(image.width):
                    pixels[x, y] = (x % 251, y % 241, (x + y) % 239)
            image.save(source)
            boxes = []
            for index, label in enumerate(CROP_ORDER):
                left = 20.25 + index * 35
                top = 90.5 + index * 4
                boxes.append(
                    {
                        "label": label,
                        "bbox_xyxy": [left, top, left + 8.5, top + 15.25],
                    }
                )
            annotations.write_text(
                json.dumps(
                    {
                        "camera_frame": 119,
                        "image_size": {"width": 400, "height": 300},
                        "boxes": boxes,
                    }
                ),
                encoding="utf-8",
            )

            first = generate(
                source=source,
                annotations=annotations,
                output_dir=output,
                repository_root=root,
                poster_videos={},
                source_reference="test-source",
            )
            first_hashes = {item["path"]: item["sha256"] for item in first["outputs"]}
            second = generate(
                source=source,
                annotations=annotations,
                output_dir=output,
                repository_root=root,
                poster_videos={},
                source_reference="test-source",
            )
            second_hashes = {item["path"]: item["sha256"] for item in second["outputs"]}

            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual([item["label"] for item in first["crops"]], list(CROP_ORDER))
            self.assertFalse(first["crop_policy"]["pixels_are_resampled"])
            with Image.open(source) as source_image:
                for crop in first["crops"]:
                    expected = source_image.crop(tuple(crop["crop_xyxy"]))
                    with Image.open(root / crop["output"]) as actual:
                        self.assertIsNone(ImageChops.difference(expected, actual).getbbox())

            with Image.open(source) as clean, Image.open(output / "frame_119_boxes.png") as boxed:
                changed_colors = {
                    boxed.getpixel((x, y))
                    for y in range(boxed.height)
                    for x in range(boxed.width)
                    if boxed.getpixel((x, y)) != clean.getpixel((x, y))
                }
            self.assertEqual(
                changed_colors,
                {ImageColor.getrgb(BOAT_COLOR), ImageColor.getrgb(BUOY_COLOR)},
            )

    def test_committed_manifest_matches_every_generated_asset(self) -> None:
        stills = self.repository_root / "presentation" / "stills"
        manifest_path = stills / "manifest.json"
        self.assertTrue(manifest_path.is_file(), "Run presentation/stills/generate.py")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["crop_order"], list(CROP_ORDER))
        self.assertEqual(len(manifest["crops"]), 9)
        self.assertEqual(len(manifest["posters"]), 4)
        self.assertEqual(len(manifest["outputs"]), 14)
        self.assertEqual(manifest["source"]["width"], 5320)
        self.assertEqual(manifest["source"]["height"], 3032)
        for output in manifest["outputs"]:
            path = self.repository_root / output["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(sha256_file(path), output["sha256"])
        source = self.repository_root / manifest["source"]["path"]
        self.assertEqual(sha256_file(source), manifest["source"]["sha256"])


if __name__ == "__main__":
    unittest.main()
