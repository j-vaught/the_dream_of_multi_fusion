from __future__ import annotations

import unittest

from PIL import Image

from presentation.media.radar_bounded.build_media import (
    FRAME_COUNT,
    Episode,
    canonical_sha256,
    clamp_crop,
    episode_output_selection,
    map_box_from_crop,
    match_predictions,
    primary_crop,
    render_disagreement_montage,
    select_episodes,
    select_separated_frames,
)


class PresentationRadarMediaTest(unittest.TestCase):
    def test_matching_is_score_ordered_class_aware_and_one_to_one(self) -> None:
        truth = [
            {"class_name": "boat", "bbox_xyxy": [0, 0, 10, 10]},
            {"class_name": "buoy", "bbox_xyxy": [20, 20, 30, 40]},
        ]
        predictions = [
            {"label": "vessel", "score": 0.8, "bbox_xyxy": [0, 0, 10, 10]},
            {"label": "boat", "score": 0.7, "bbox_xyxy": [0, 0, 10, 10]},
            {"label": "green buoy", "score": 0.6, "bbox_xyxy": [20, 20, 30, 40]},
        ]
        result = match_predictions(truth, predictions)
        self.assertEqual(len(result.matched_truth_indices), 2)
        self.assertEqual(
            [prediction["_is_true_positive"] for prediction in result.predictions],
            [True, False, True],
        )

    def test_primary_crop_uses_detection_count_then_area(self) -> None:
        record = {
            "crops": [
                {
                    "merged_crop_xyxy": [0, 0, 200, 200],
                    "dino_detections_crop": [{}],
                },
                {
                    "merged_crop_xyxy": [10, 10, 50, 50],
                    "dino_detections_crop": [{}, {}],
                },
            ]
        }
        self.assertEqual(primary_crop(record), record["crops"][1])
        self.assertIsNone(primary_crop({"crops": []}))

    def test_episode_selection_is_deterministic_non_overlapping_and_exact_length(self) -> None:
        scores = {frame: 0 for frame in range(119, 419)}
        focus = {}
        for frame, score in ((130, 5), (220, 5), (400, 4), (135, 99)):
            scores[frame] = score
            focus[frame] = [[100, 100, 120, 140]]
        episodes = select_episodes(
            scores,
            focus,
            first_frame=119,
            last_frame=418,
        )
        self.assertEqual([episode.center_frame for episode in episodes], [135, 220, 400])
        self.assertTrue(
            all(
                episodes[index].end_frame < episodes[index + 1].start_frame
                for index in range(len(episodes) - 1)
            )
        )
        selection = episode_output_selection(episodes)
        self.assertEqual(len(selection), FRAME_COUNT)
        self.assertEqual(selection.count(episodes[0].start_frame), 5)

    def test_crop_clamps_to_original_frame(self) -> None:
        self.assertEqual(clamp_crop([0, 0, 10, 10]), (0, 0, 1150, 650))
        self.assertEqual(
            clamp_crop([5310, 3022, 5320, 3032]),
            (4170, 2382, 5320, 3032),
        )

    def test_crop_box_mapping_clips_and_scales_to_display_coordinates(self) -> None:
        self.assertEqual(
            map_box_from_crop([100, 100, 300, 300], [100, 100, 500, 500], (800, 600)),
            (0, 0, 400, 300),
        )
        self.assertEqual(
            map_box_from_crop([0, 0, 200, 200], [100, 100, 500, 500], (800, 600)),
            (0, 0, 200, 150),
        )
        self.assertIsNone(map_box_from_crop([0, 0, 50, 50], [100, 100, 500, 500], (800, 600)))

    def test_separated_frame_selection_uses_score_then_frame_tie_break(self) -> None:
        frames = select_separated_frames(
            {119: 4, 120: 9, 140: 9, 180: 8},
            count=3,
            minimum_separation=15,
        )
        self.assertEqual(frames, [120, 140, 180])

    def test_canonical_hash_ignores_mapping_insertion_order(self) -> None:
        self.assertEqual(
            canonical_sha256({"a": 1, "b": [2, 3]}),
            canonical_sha256({"b": [2, 3], "a": 1}),
        )

    def test_montage_has_fixed_full_hd_geometry(self) -> None:
        tiles = [Image.new("RGB", (960, 540), color) for color in ("red", "blue", "green", "white")]
        montage = render_disagreement_montage(tiles)
        self.assertEqual(montage.size, (1920, 1080))
        self.assertEqual(montage.getpixel((10, 10)), (255, 0, 0))
        self.assertEqual(montage.getpixel((970, 550)), (255, 255, 255))

    def test_episode_expansion_contract(self) -> None:
        episodes = [
            Episode(128, 119, 138, 5, (0, 0, 900, 650)),
            Episode(228, 219, 238, 4, (0, 0, 900, 650)),
            Episode(408, 399, 418, 3, (0, 0, 900, 650)),
        ]
        selection = episode_output_selection(episodes)
        self.assertEqual(selection[0], 119)
        self.assertEqual(selection[-1], 418)
        self.assertEqual(len(selection), 300)


if __name__ == "__main__":
    unittest.main()
