# Detection Error Evidence Mining

The error miner class-matches scored predictions to You Only Look Once (YOLO)
ground truth with deterministic greedy one-to-one assignment at an
intersection-over-union threshold of $0.5$. It reads the final vision-only and
radar-confidence-gated JSON Lines files. It uses the recovered local
radar-bounded file when available and otherwise copies that single file from
`comech-2422` into the owned evidence cache.

The six candidate categories use explicit ranking rules. Tiny misses are
ground-truth objects missed by every final method and are ranked by pixel area.
Localization cases are ranked by the highest same-class overlap below $0.5$.
Duplicate hypotheses come from pre-non-maximum-suppression radar-crop
predictions that overlap one ground-truth object. Radar clutter is ranked by
the elongation of the supporting radar region after excluding likely
localization errors. Ambiguous omissions require an unmatched radar-bounded
hypothesis to persist across at least three consecutive frames. Method
disagreements require one final method to match an object while another misses
it.

Run the deterministic build from the repository root.

```sh
uv run python -m presentation.errors.mine_errors
```

Selection occurs before image transfer. The build requests only the unique
clean source PNGs needed by selected evidence and retains no unselected source
frames in the evidence directory. Every rendered crop is $1600 \times 1000$
pixels with no text overlay. False positives use solid red, false negatives use
dashed cyan, correct boats use solid magenta, and correct buoys use solid
yellow.

The manifest records input and output hashes, source and rendered coordinates,
scores, intersection-over-union values, rank keys, and category-specific
selection evidence. Repeated builds against unchanged inputs produce identical
manifest and crop hashes.

Compile the CeTZ-authored quality-assurance montage and inspect the resulting
PNG after changing selection logic or source data.

```sh
typst compile --format png --ppi 90 \
  presentation/errors/qa_montage.typ \
  presentation/errors/qa_montage.png
```
