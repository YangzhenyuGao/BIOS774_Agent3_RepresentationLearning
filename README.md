# BIOS774 Agent 3: Representation Learning

[GitHub repository](https://github.com/YangzhenyuGao/BIOS774_Agent3_RepresentationLearning) · [Project report](https://github.com/YangzhenyuGao/BIOS774_Agent3_RepresentationLearning/blob/main/reports/report.pdf) · [Study 1: PBMC RNA + ATAC](https://github.com/YangzhenyuGao/BIOS774_Agent3_RepresentationLearning/blob/main/reports/generated_report_1.pdf) · [Study 2: TCGA-BRCA](https://github.com/YangzhenyuGao/BIOS774_Agent3_RepresentationLearning/blob/main/reports/generated_report_2.pdf)

For submission, `reports/report.pdf` links directly to the code repository and
both complete study reports. It has four body pages, plus references/appendices.
The public source snapshot includes all loaders and model adapters, tests,
configurations, objectives, templates, environment locks and the three PDFs.
`examples/` contains aggregate frozen results for inspection without local data.
Raw datasets, per-sample records, learned embeddings, tile feature stores,
checkpoints and pretrained weights are excluded from GitHub.

Working directory: `/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning`.

Execution status (2026-11-03): both real-data experiments, held-out evaluation,
downstream analyses, scientific validators and final PDF visual review are complete. The user authorized
`prompt/prompt.md` and supplied a November 10, 2026 deadline (Eastern date; exact
hour unspecified). Release acceptance is recorded in
`reports/visual_review_validation.json`, `reports/project_report_validation.json`
and each dataset's `outputs/<dataset>/validation_results.json`.

The reusable core is a deterministic objective/profile planner, a real method
registry, budgeted empirical pilots, validation-only selection/tuning, frozen
held-out testing and evidence-based reporting. No external LLM was used. Both
applications run through the same CLI and stage DAG. Full smoke **5003437** passed
28 tests, two complete synthetic pipelines and checksum-validated resume; four
additional budget tests, two WSI artifact checks and three serialization checks
also passed. All four local pathology models passed H100 runtime acceptance.

| Dataset | Cohort and validation-selected representation | Held-out result | Integration verdict |
|---|---|---|---|
| PBMC RNA + ATAC | 10,243 QC-retained cells; CLIP-style paired encoders, 32 dimensions | Macro-F1 0.925; symmetric paired recall@10 0.497 | `integrate_with_caution` |
| TCGA-BRCA histology + clinical | 1,132 primary slides / 1,061 patients; 802 four-class PAM50 labels; late fusion, 64 dimensions | Macro-F1 0.723; balanced accuracy 0.714 | `integrate_with_caution` |

These are the predeclared representative-seed results under the same frozen
logistic-probe protocol. TCGA's late-fusion image branch is gated attention MIL
on frozen UNI2-h tile features, paired with a clinical MLP. Training averages the
branches' class logits; the exported representation concatenates their latent
vectors. End-to-end head scores are separately labeled in the dataset report.

PBMC improves paired retrieval, but its macro-F1 gain over RNA alone is 0.016
(95% paired CI −0.009 to 0.060), with residual QC/depth predictability. Expert
labels are WNN-derived and the sample is one donor. TCGA's gain over histology
MIL is 0.078 (95% CI −0.009 to 0.170), with source-site predictability. Removing
receptors gives F1 0.583; the separate site-held-out sensitivity gives 0.679.
Exploratory survival C-index is 0.665 (95% CI 0.558–0.770; 30 test deaths).
These results do not establish donor/site generalization or clinical benefit.

All baselines and eligible learned pilots succeeded: PBMC four baselines plus
12 learned runs; TCGA eight baseline runs (including three MIL seeds) plus 21
learned pilots, including actual CONCH-Z tower fine-tuning. PBMC production
**5003451** completed in 30m29s. TCGA training **5003644** completed all model
fits, then exposed two Arrow export limitations. The scoped CLI IO adapter
preserves float64 values and canonical nested receipt metadata. Frozen-export
QA and production audit verify exact values/IDs/splits for 7,427 patient-model
rows, all 1,132 slide vectors, both frozen plans and 11 TCGA checkpoints.
Recovery **5003726** completed export, evaluation and downstream analyses in
5m08s without model refitting. Initial combined finalizer **5003732** passed all
scientific/source checks (377 PBMC and 353 TCGA numeric claims).

Three frozen encoders (UNI2-h, CONCH-Z and ImageNet ResNet-18) cover every primary
slide, with a training-throughput/queue-derived cap of 128 tiles per slide.
One uniform red slide-edge artifact required a separately logged segmentation-
thumbnail repair; original pixels, failed coordinates and receipts were kept,
region-QC thresholds were unchanged and no patient was excluded. Full recovery
history is in `docs/PROGRESS.md` and the generated evidence bundle.

The project GPU ledger reconciles to **2.770 GPU-hours**, zero outstanding GPU
reservations, below the authorized 48-hour / two-concurrent-GPU limits. Dataset
validators also confirm shared datasets, the model store, Agent1/2 and their
environments match the initial protected snapshot. All Agent3 derivatives stay
in this directory. Analysis code remains frozen at
`bb6b6aba223fbb128a4ad0f75eb78d3edd9d8e12b31e57fec7b5fcf98debce71`.

## Critical main-report revision (2026-11-03)

`reports/report.pdf` has been rewritten as a compact academic paper: **four body
pages, one reference page and two appendix pages**. It displays all eight PBMC
and thirteen TCGA pilot strategies, with held-out results only for models that
were actually advanced to final evaluation. Internal model identifiers are
replaced by descriptive names in the body. The prior main report is preserved
locally under `reports/archive/before_critical_revision_20261103/`.

The revised interpretation is stricter than the unchanged machine verdicts:
PBMC improves paired retrieval but fails the requested structural noninferiority
on ARI; NMI is mixed, and incremental RNA-to-ATAC prediction benefit is not
established. TCGA's receptor-free fusion F1 **0.583** is below image-only MIL
**0.645**. The fusion native head's F1 **0.482** and Her2 recall **0/10** are
explicitly distinguished from the latent probe's F1 **0.723** and recall **0.700**.
The report does not claim general multimodal synergy.

It also documents short/unauditable MultiVI convergence, the noncanonical WNN
extension, duplicate ridge/decoder references, limited encoder coverage and
conservative tile/tuning budgets. Two factual qualifications matter: MultiVI
has a custom validation callback despite disabling built-in early stopping;
the logistic probe is also unweighted, so weighting cannot explain its head gap.
These are interpretations of saved evidence. No model, preprocessing, test
metric, original dataset report or machine verdict was rerun or changed.
The revised main report passes **124** numeric-source checks and full visual QA.
`logs/report_revision_release.json` records this report-only release;
`logs/workflow_state.json` retains the original full computational run history.

## Assignment and deliverables

Build an AI agent for **multimodal representation learning**. Given a dataset
and a user objective, it should inspect modalities, choose and perform suitable
preprocessing, select an integration strategy, encoders and training objectives,
train or fine-tune models, evaluate representations, perform downstream analysis,
and automatically report decisions and findings. Choices must respond to the data
and objective rather than replay two manually fixed pipelines.

Relevant method families include contrastive, self-supervised, supervised,
CLIP-style learning and pretrained multimodal model fine-tuning. Not every method
must run on every dataset; the agent must explain its choices. The two planned
demonstrations are RNA + ATAC PBMC and histopathology + clinical TCGA-BRCA.

Canvas deliverables:

- Source code with a single-command complete pipeline and minimal intervention.
- For each dataset: learned representations, quantitative evaluation, downstream
  results, preprocessing/method decisions, and an automatically generated report
  named `generated_report_1` (PBMC Multiome) or `generated_report_2` (TCGA-BRCA).
- `report.pdf`: at most four main-body pages, excluding references/appendices,
  describing architecture, decision logic, tools/models, both datasets' results,
  strengths and limitations. Emphasize agent behavior and reproducibility rather
  than an exhaustive model benchmark.

## Shared input data

**Storage boundary:** Treat both shared datasets as read-only analysis inputs.
All new Agent3 outputs, intermediate files, image tiles, embeddings, caches,
checkpoints, logs, reports and dataset splits must remain under this project's
working directory. Do not write new analysis artifacts beside the shared inputs.
Do not alter Agent1/Agent2 projects or their environments.

### 1. Single-cell RNA + ATAC: 10x Multiome PBMC

The 10x "PBMC from a Healthy Donor, Granulocytes Removed Through Cell Sorting (10k)"
Multiome files (Cell Ranger ARC 1.0.0) are stored in:

`/vast/som/yunligrp/users/ygao/BIOS774Agent/data/pbmc_granulocyte_sorted_10k`

This replaced the originally downloaded Chromium X sample, which had no cell-type
annotation, on 2026-11-02. It is the sample used by the Seurat WNN, Signac and
SeuratData `pbmcMultiome` tutorials.

Use the filtered HDF5 matrix for the called-cell RNA and ATAC matrices. Both
modalities share barcode columns; split rows using `features/feature_type`.
The raw HDF5 matrix additionally contains background/non-called barcodes.
Downloaded reference analyses are vendor outputs, not results produced by Agent3.

- Filtered matrix: **11,909 cells, 36,601 genes, 108,377 ATAC peaks**, GRCh38.
- Raw matrix: 736,320 barcodes, including background/non-called barcodes.
- Ten 10x files plus the SeuratData annotation package, **3,181,633,244 bytes**:
  filtered/raw HDF5 matrices, fragments and Tabix index, peaks, peak annotation,
  barcode/sample QC, vendor analysis archive and web report, and
  `annotations/pbmcMultiome.SeuratData_0.1.4.tar.gz`. FASTQ/BAM are not included.
- **Cell-type labels:** use `annotations/pbmc_10x_annotation_FixFlip.csv` (column
  `celltype`; 19 types for 10,412 cells, plus 1,497 `filtered`). These are the Satija
  lab's curated annotations of WNN clusters of these cells, not measured ground truth.
  The released `seurat_annotations` field has Naive B and Memory B swapped, as
  verified with markers; see `annotations/LABEL_PROVENANCE.json`.
- Barcode pairing, sparse structure and peak-count checks passed, and the label
  barcodes match the filtered matrix exactly. Acquisition records are in the dataset's
  `README.md`, `download_metadata/`, `SHA256SUMS` and `annotations/SHA256SUMS`.
- Single-donor data (healthy female, age 25); cell-level evaluation does not
  establish donor generalization.

Source: [10x dataset](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-1-0-0);
labels from [SeuratData `pbmcMultiome`](https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis).

`scripts/download_multiome.sh` and `scripts/inspect_multiome.py` were used for
acquisition. They write download/inspection metadata into the shared data
directory: do not rerun them as read-only inspection commands. Any future adapted
inspection should write its results inside Agent3 instead. No redownload is needed.

### 2. Histopathology + clinical variables: TCGA-BRCA

Shared dataset root:
`/vast/som/yunligrp/users/ygao/imaging_fi/data/TCGA-BRCA`

**This entire directory belongs to another active project and must not be modified,
moved, cleaned up, or overwritten, including `processed/`, `annotations/` and
`provenance/`.** Read data directly from there; keep every Agent3 derivative here.

- `raw/wsi/<file UUID>/<filename>.svs`: **1,133 diagnostic slides, 1,062 patients,
  approximately 1.16 TB**. All manifest-listed files existed and matched their
  recorded sizes during inspection; full slide checksums/image QC were not rerun.
- `raw/clinical/TCGA-BRCA.with_followups.json`: 1,098 clinical cases. All 1,062
  slide patients match a clinical record via the TCGA patient identifier. Relevant
  fields include age, stage, demographic information and follow-up; some are missing.
- `processed/clinical/`: existing `phenotype.parquet`, clinical lineage and revision
  audit. The audit documents primary-diagnosis selection and missingness. The
  Parquet table itself has not yet been loaded in Agent3.
- `annotations/pam50/BRCA_clinicalMatrix.tsv`: UCSC Xena annotation table, containing
  primary tumor, normal tissue and metastatic samples. Restricting to primary
  tumor sample code `01` gave an initial PAM50 overlap of 826 slide patients.
  Agent3's primary-slide cohort has **1,132 slides / 1,061 patients**, including
  **802 four-class PAM50 patients** (LumA 410, LumB 190, Basal 137, Her2 65) and
  23 Normal-like patients excluded from primary classification. There are 872
  patients with complete clinical inputs, including 665 with four-class labels.
- `provenance/manifests/wsi/TCGA-BRCA.diagnostic.all.tsv`: slide IDs, filenames,
  sizes and MD5 values. Existing acquisition QC reports PASS. The manifest includes
  one metastatic slide, excluded from Agent3's primary-slide cohort.
- `raw/omics/`, `processed/omics/`, and `processed/cohorts/` contain work from the
  other project. Existing cohort definitions and draft splits are not Agent3's
  approved protocol and must not be adopted automatically.

Agent3's manifested split keeps every patient's slides/tiles together and includes
eligible unlabeled patients before any pretraining. Clinical inputs are restricted
to the approved allowlist, separately from PAM50 and survival targets. Missing
covariates are encoded explicitly; labels are not required for every patient.

Assignment reference: [TCGA-BRCA collection](https://www.cancerimagingarchive.net/collection/tcga-brca/).
The local WSI/clinical manifests are the inventory used for this handoff.

## Environment and execution

Use the independent environment:
`/vast/som/yunligrp/users/ygao/conda/envs/agent3`.

`environment.yml` and the exported locks describe the installed analysis environment.
Additional Python analysis/modeling dependencies belong in this environment. Do
not install them into Agent1, Agent2, base Conda or a home Python user-site library.
R steps may use `module load r/4.6.0` and the separately authorized personal R
library `~/R/x86_64-pc-linux-gnu-library/4.6`; its migration is complete and
recorded under `scripts/r_home_setup/` and `environment.R.lock.csv`.

Heavy computation runs in Slurm allocations. Sycamore H100 submissions
must explicitly select account `h100_harlin` and an appropriate partition:
`h100_short` (up to two hours), `h100_priority` (up to five days), or
`h100_preempt` (up to five days, preemptible). Recheck availability before use.
CUDA PyTorch 2.8.0 (cu128) and real analysis/model dependencies are installed in
agent3, with refreshed environment locks and passing `pip check`. The PDF
toolchain passed rendering/text extraction. All four local pathology FMs passed
strict CPU loading/random forward checks and H100 bf16 acceptance (job 5003345).

## Pipeline interface

Every shell entrypoint explicitly exports `PYTHONNOUSERSITE=1` and redirects
caches into Agent3. Activate the project settings before direct Python commands:

```bash
source scripts/slurm/common.sh
python scripts/agent3_cli.py run --config config/pbmc_multiome.yaml --objective objectives/pbmc_multiome_joint.yaml --resume
python scripts/agent3_cli.py run --config config/tcga_brca.yaml --objective objectives/tcga_brca_pam50.yaml --resume
```

Heavy stages require a Slurm allocation. Real-data baseline/pilot stages refuse
to start without a passing smoke gate tied to current analysis code. The final
plan/checkpoint hashes must be frozen before held-out test access. The smoke
entrypoint is `sbatch scripts/slurm/smoke.sbatch`.

The production wrapper forwards the package CLI's commands and arguments and
adapts Parquet vector serialization for heterogeneous floating dtypes. It keeps
float64 precision and checks an exact round trip, recording its own source hash
separately from the unchanged analysis hash. Encoder receipts are canonical JSON
strings preserving empty nested dictionaries. It changes no model, preprocessing,
selection or frozen-plan file. Package stage commands such as report/validate
continue to use `python -m agent3_replearn` directly.

From this project directory, `bash scripts/submit_workflow.sh` submits the complete
Slurm workflow. It refuses to launch a duplicate active controller. Its phase/job
record is `logs/workflow_state.json`; do not submit an independent copy while the
controller is active. Failed stages preserve their evidence and block dependents.

The GPU ledger is `logs/gpu_ledger.json`. Submit GPU work through
`scripts/gpu_jobs.py` to enforce at most two reserved/running GPUs and a cumulative
48 GPU-hour cap. Dataset/model provenance and caveats are in
`docs/DATA_PROVENANCE.md`; rules, budgets and comparison protocols were committed
before real pilots in `docs/METHOD_NOTES.md`. Requirement coverage is tracked in
`docs/REQUIREMENTS_TRACEABILITY.md`.

## Deliverables and handoff

- `reports/generated_report_1.pdf`: PBMC methods, decisions, representations,
  evaluation, downstream analyses and limitations.
- `reports/generated_report_2.pdf`: TCGA methods, decisions, MIL/clinical
  contributions, sensitivity analyses, attention examples and exploratory survival.
- `reports/report.pdf`: four main-body pages, followed by references and a
  plan-only objective-swap appendix.
- `submission/source_code.zip`: source, configurations, objectives, environment
  locks, documentation, manual visual-review record and all three PDFs.
- `submission/generated_outputs.zip`: identifier-labeled representations,
  metrics, downstream analyses, decision logs, figures, verdicts and audit proofs.
- `submission/manifest.json` and `submission/SHA256SUMS`: exact content, sizes and
  hashes. Shared raw data, tile feature stores, checkpoints and model weights
  are excluded from both archives.

Actual page inspections are recorded with PDF/page hashes in
`reports/visual_review.json`; `reports/visual_review_validation.json` checks that
those explicit records match the current files. Preview creation alone never
counts as visual approval. Finalization uses `scripts/build_project_report.py`
to preserve the reviewed main PDF bytes only while all report inputs, code and
output checksums match; its cache receipt is included in generated outputs. Controller/stage records are in
`logs/workflow_state.json`; final model plans remain frozen. See
`docs/PROGRESS.md` for job IDs, failures, recoveries and final acceptance.
The alternative survival-primary objective is a **plan-only** demonstration;
PAM50 is primary and survival secondary in the executed TCGA experiment.
GitHub publication is authorized by the latest user request. Canvas submission remains manual.
Local terminal SSH setup is documented in docs/SYCAMORE_GITHUB_ACCESS.md; it is
separate from the connected GitHub app used for this publication.
