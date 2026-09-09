# Reproducible experiment reports

This directory stores small, reviewable artifacts needed to reproduce claims:

- frozen manifests and split assignments;
- exact experiment configuration;
- per-sample JSON or CSV metrics;
- training curves and checkpoint metadata (not weight tensors);
- environment and command summaries.

Reports produced after the RMSD correction must declare
`metric_schema: ospedit.structure_metrics.v2`. Older coordinate RMSD values used
per-coordinate RMSE and are smaller than standard atom RMSD by `sqrt(3)`.

Large checkpoints, downloaded structures, embeddings, and prediction caches stay
outside Git. A tracked report should contain their checksum and provenance when
they contribute to a result.
