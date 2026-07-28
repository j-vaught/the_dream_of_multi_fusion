# Radar confidence gate media

The generated media explains and demonstrates the confidence gate without
burned-in labels or confidence text. The static figures are authored in Typst
with CeTZ. The two videos are encoded at 60 frames per second from those
figure frames.

Run the deterministic generator from the repository root.

```sh
uv run python presentation/media/gate/generate_gate_media.py
```

The generator reads the committed detection shards, final gated predictions,
YOLO ground truth, and the raw source video. It does not render or duplicate
the existing prediction-only base video. `asset_manifest.json` records source
hashes, selected camera frames, geometry provenance, media properties, and
output hashes.
