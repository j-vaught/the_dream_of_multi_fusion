# Dream Fusion YOLO Dataset

This directory stores the versioned YOLO labels and audit metadata for camera
frames 119 through 418. The complete dataset, including 300 full-resolution
5320 by 3032 PNG images, is generated at
`out/datasets/dream_fusion_yolo` by `export_yolo_dataset.py`.

The dataset uses class 0 for `boat` and class 1 for `buoy`. Each YOLO label row
contains the class ID followed by normalized center $x$, center $y$, width, and
height. `manifest.jsonl` retains camera-frame numbers, stable track IDs,
pixel-coordinate boxes, normalized boxes, tracker confidence, and provenance
for later auditing.

The split is chronological. Training contains frames 119 through 328,
validation contains frames 329 through 373, and testing contains frames 374
through 418. This avoids leakage from nearly identical adjacent video frames.

Track `b2` is labeled only on frames 119 through 210. It is absent from frame
211 through frame 418. The exporter command uses
`--assert-absent-after b2:211` and fails if that constraint is violated.
