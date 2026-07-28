# Presentation metrics

This directory contains the deterministic evaluation and figure pipeline for the
300 camera frames from 119 through 418. J.C. Vaught authored the pipeline.

The evaluator reads the committed YOLO manifest and scores each experiment with
class-aware, confidence-ordered, one-to-one matching. A match is accepted when
intersection over union is at least 0.50. Ties are resolved from stable fields,
so repeated runs produce byte-identical CSV files.

Run the metric build from the repository root.

```sh
uv run python -m presentation.metrics.pipeline
```

The build writes aggregate method metrics, class metrics, track recall and match
quality, frame metrics, frame-by-class metrics, track-frame outcomes, match
events, rolling frame metrics, and six figure-specific CSV files under `data`.
The file `metrics_manifest.json` records input hashes, the matching contract,
the selected prediction source for each method, and whether detailed records
were available.

The radar-bounded prediction file is
`experiments/detection_comparison/radar_bounded_full.jsonl`. Each JSONL record
must contain `camera_frame` and a `remapped_deduped` list. When that file is
absent, the build uses aggregate radar-bounded values from the committed
`metrics.json`. It does not invent track-level or frame-level bounded-radar
results. The manifest marks that temporary limitation.

Compile the six figures with the build script. The figures use Calibri from the
local Microsoft Office installation and are drawn with CeTZ.

```sh
presentation/metrics/build_figures.sh
```

The PDF and PNG outputs are written under `figures/build`. Figures intentionally
contain no internal plot title because the surrounding presentation supplies
the headline.
