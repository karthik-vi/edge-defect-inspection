# Decisions (ADR: Context -> Options -> Decision -> Evidence)
Phase entries are added as each component is measured.

## ADR-001: Classical baseline alignment (Phase 1)
- **Context:** Phase 1 baseline = golden-image rules (pixel z-diff vs mean/std golden image, per-cell edge density, per-cell Laplacian variance; max of val-normalised rule scores). Alignment was meant to reduce pose variation.
- **Options:** (a) none, (b) centroid translation, (c) centroid translation + principal-axis rotation (only if eccentricity > 1.3).
- **Decision:** `align_mode: none` as the fixed baseline config.
- **Evidence** (image AUROC, 20 golden + 20 val-good, MVTec test split; `results/classical_align_ablation.json`):

| align | bottle | metal_nut | screw |
|---|---|---|---|
| none | 0.985 | 0.637 | 0.526 |
| translate | 0.960 | 0.613 | 0.569 |
| translate+rotate | 0.960 | 0.613 | **0.103** |

  Moment-based rotation is unstable (180° ambiguity, near-symmetric nut) and inverts the screw result; centroid translation did not help on 2/3 categories. Alignment is therefore *not* earning its place in the classical pipeline; it is re-tested in Phase 2 on learned features.
  Caveat: the mode was compared on the test split with 3 configs; "none" is the best average (0.716), not tuned beyond that.
