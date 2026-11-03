# Frozen aggregate evidence

These small JSON summaries copy existing pilot and held-out **aggregate**
results; generating them does not fit a model or recompute a performance metric.
They contain no cell/patient records, identifiers, embeddings, tile features,
raw matrices, images, checkpoints or pretrained weights.

- `pbmc_multiome/aggregate_results.json`: every pilot, frozen classification /
  clustering / alignment results, RNA-to-ATAC metrics and baseline-identity notes.
- `tcga_brca/aggregate_results.json`: every pilot, frozen probe and native-head
  results, and receptor-removal / site-held-out aggregate sensitivities.

Validation and test results remain separate. An unadvanced method has no test
entry. Original machine verdicts are retained; the stricter audited interpretation
is explained in the project report. Source SHA256 values identify the saved
experimental artifacts from which these excerpts were copied.
