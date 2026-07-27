# Radar Input for the 300-Frame Video

This directory contains the exact processed radar input used for camera frames
119 through 418 in `01_input_raw.mp4` and `02_radar_bounding.mp4`.

`frames.jsonl` contains one record for each of the 300 camera frames. Each
record identifies the synchronized radar frame, the corresponding file under
`points/`, and the detection point ranges, bounding boxes, centroids, and track
IDs used by the rendering pipeline.

Each `.npz` file contains the arrays `u`, `v`, and `intensity`. The `u` and `v`
values are radar returns that were already projected into the original
5320-by-3032 camera image plane. Files associated with the same radar frame
have identical payloads. The 300 camera records use 28 distinct radar sweeps
and contain 207,781 distinct projected radar returns.

These files are the complete radar input to `dream_multi_fusion.py`. They are
not the upstream raw Simrad recording. The original projection pipeline loaded
marine radar returns from MATLAB `.mat` files, but those source files were not
present in the server checkout when this subset was assembled.

`SHA256SUMS` records the hashes of `frames.jsonl` and all 300 point files.
