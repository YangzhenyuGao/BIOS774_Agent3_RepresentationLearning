# Progress

## 2026-11-02 — shared Multiome inputs and isolated environment

- Established Agent3 as the working project; no modeling pipeline or assignment
  assumptions were added.
- Downloaded 10 original Cell Ranger ARC 2.0.0 output files from the requested 10x
  Chromium X Multiome PBMC dataset into the shared `BIOS774Agent/data/` directory,
  under `10k_PBMC_Multiome_nextgem_Chromium_X/`.
- Total original file size: **4,425,504,979 bytes**. All transfers completed; no
  `.part` files remain. Remote Content-Length values matched where provided.
  Local SHA-256 hashes, original URLs, HTTP headers, host and completion timestamp
  are saved alongside the data. Provider SHA-256 values were not available.
- `rc-dm.its.unc.edu` was reachable, but noninteractive SSH authentication was
  rejected. Transfers ran on `sycamore-login1`; no Slurm jobs were submitted.
- Created independent environment
  `/vast/som/yunligrp/users/ygao/conda/envs/agent3` using the shared system Conda
  executable. Added Python 3.11, pip, NumPy, SciPy, pandas and h5py. Import checks
  and `pip check` passed. Exact Conda artifacts and Python versions are recorded
  in `environment.conda-explicit.txt` and `environment.lock.txt`.
- Configured `PYTHONNOUSERSITE=1`. Agent1 and Agent2 project files and environments
  were not modified. The shared Conda package cache was used for the new install.
- Structural inspection passed: filtered matrix has **10,970 barcodes**,
  **36,601 Gene Expression features** and **111,743 Peaks**, reference **GRCh38**.
  Raw matrix has 733,612 barcodes (including background/non-called barcodes), the
  same feature order, and contains every filtered barcode. BED peak count agrees.
  Both modality matrices share the same barcode columns. Original sparse matrix
  dimensions and pointer arrays are consistent.
- Tabix index decompresses and has its expected signature; the fragment stream
  opens with the expected five columns. This is not a full fragment-level QC run
  or a full decompression/integrity scan of the entire fragment archive.
- Acquisition log: `logs/download_multiome.log`; environment log:
  `logs/environment_setup.log`; matrix check: `logs/matrix_inspection.log` and
  the dataset's `download_metadata/matrix_inspection.json`.
- Data scope excludes FASTQ/BAM, Loupe files and redundant Matrix Market copies.
  No preprocessing, training, biological inference or GPU runtime test occurred.
  Future modeling dependencies should be installed only into `agent3` once the
  assignment and methods are established.

## 2026-11-02 (evening) — PBMC sample replaced by the annotated Seurat/Signac sample; pathology model store

- At the user's request, replaced the PBMC Multiome sample because it had no cell-type
  annotation. Downloaded 10x "PBMC from a Healthy Donor, Granulocytes Removed Through Cell
  Sorting (10k)" (Cell Ranger ARC 1.0.0, CC BY 4.0) into
  `BIOS774Agent/data/pbmc_granulocyte_sorted_10k/`: the same ten file types as before plus the
  Satija lab's `pbmcMultiome.SeuratData_0.1.4.tar.gz`, 3,181,633,244 bytes in total. Remote sizes
  matched, and SHA256SUMS were written and re-verified. Downloaded on `longleaf-login4`, with
  `scripts/download_multiome.sh` re-pointed at this sample. Log:
  `logs/download_multiome_granulocyte_sorted_10k.log`.
- Structural inspection passed (`scripts/inspect_multiome.py`, now taking `--sample`): filtered
  matrix **11,909 cells, 36,601 genes, 108,377 peaks**, GRCh38. The raw matrix has 736,320 barcodes
  and contains every filtered barcode, and the BED peak count agrees. Log:
  `logs/matrix_inspection_granulocyte_sorted_10k.log`.
- Labels: `annotations/pbmc_10x_annotation_FixFlip.csv` was extracted verbatim from the SeuratData
  package (renamed from `.tsv` because it is comma-separated). Its barcodes equal the filtered matrix
  barcodes, in the same order. There are 19 types for 10,412 cells; the 1,497 `filtered` cells are
  exactly those failing the Seurat WNN vignette QC.
  - The package's released `seurat_annotations` (Rds and `pbmc.rna` metadata, identical to each
    other) differs from FixFlip only by swapping Naive B (142 released) and Memory B (371 released).
  - Canonical markers show FixFlip is correct: released "Memory B" has TCL1A/IGHD/FCER2/IL4R high
    and AIM2/SSPN/IGHA1/IGHG1/CD27 near zero; released "Naive B" shows the reverse.
  - Evidence: `annotations/LABEL_PROVENANCE.json`. R 4.5.0 (`module load r/4.5.0`, base R only) was
    used to read the Rds/rda; nothing was installed into R.
- The labels are curated WNN-cluster annotations, not measured ground truth. The prompt now flags
  WNN's structural advantage and requires label-free metrics alongside them.
- Deleted the previous sample `data/10k_PBMC_Multiome_nextgem_Chromium_X/` (4.2 GB, ARC 2.0.0) at
  the user's request, without backup, only after the replacement and its labels were verified.
  Earlier logs (`logs/download_multiome.log`, `logs/matrix_inspection.log`) describe that deleted
  sample and are kept as history.
- Pathology foundation models: the user's shared store `/vast/som/yunligrp/users/ygao/models/` now
  holds UNI2-h, Virchow2, CONCH-Z and MUSK, the four models in scope. Each was downloaded at a pinned
  Hugging Face revision with every file hash-verified and a `DOWNLOAD_MANIFEST.json`. CONCH v1.5 and
  CONCH v1 copies were removed at the user's request. The store is read-only for Agent3.
- `prompt/prompt.md` (execution prompt, awaiting user approval) updated accordingly. No
  preprocessing, training or GPU job has been run.


## 2026-11-02 (late evening) — authorized execution, Phase 0 and environment preparation

- User explicitly authorized execution of `prompt/prompt.md`; supplied deadline
  November 10, 2026. Current client timezone America/Indianapolis; exact hour
  unspecified. All sections were read before implementation.
- Confirmed host sycamore-login1; inspected Slurm partitions and associations.
  H100 partitions provide four nodes, four H100 GPUs/node; account h100_harlin.
  CPU stages use rc_yunmli_pi/small. GPU cap remains 48 aggregate hours and two
  concurrent GPUs. No real-data modeling has run.
- Initialized local Git main; initial preservation commit `1ac9fd0`. Added official
  assignment requirements and approved defaults in docs/ASSIGNMENT_REQUIREMENTS.md.
  Inspected Agent1/Agent2 configuration, worker isolation, smoke gate and Slurm/report
  infrastructure read-only; did not copy stale queue or dataset-specific logic.
- Input protection/integrity: job 5003326 failed before reading inputs because
  Slurm copies the submitted shell script, breaking dirname-relative sourcing.
  Fixed sourcing to the project path. Retry 5003331 completed exit 0 (1m42s):
  all PBMC data/annotation SHA256 entries and every file of all four approved
  pathology FMs match their supplied manifests. Initial metadata snapshots cover
  both datasets, four model folders, Agent1/Agent2 and their Conda environments:
  logs/preflight/protected_initial.json. Receipts: logs/preflight/integrity.json.
- Read CONCH-Z's five bundled remote-code Python files before any execution;
  loader records their hashes and uses only the pinned local files.
- Extended existing agent3 environment in place; all caches redirected under
  Agent3/.cache and explicit PYTHONNOUSERSITE=1 enforced by common.sh. Installed
  CUDA PyTorch 2.8.0/torchvision 0.23.0 (cu128), real scvi-tools, muon/mudata,
  mofapy2, scanpy, MIL/image/PDF/test dependencies. MUSK official source is pinned
  to 714b666969c1911e5efe70d991140a21030f4ef3; full package versions are in the
  refreshed environment locks. No Agent1/Agent2/base/user-site packages changed.
- pip check passed after the completed install; PDF render/text extraction passed
  (.cache/pdf_toolchain.pdf). Intermediate environment export during installation
  was incomplete; the final completed install re-exported the locks and passed.
- R migration is already complete: main 139/139 and Appendix B 169/169 load,
  job 5003271 completed exit 0. Full installed package/version/source/revision/path
  inventory is exported to environment.R.lock.csv; verification and functional
  tests are in logs/r_home_setup/. R used module r/4.6.0 and the home 4.6 library.
- Implemented initial typed config/objective/data/run/verdict schemas, input
  boundary checks, profile-based planner, exact multiome pairing loader, Agent3
  clinical/cohort builder, deterministic patient/cell splits, and train-fitted
  count/clinical preprocessing. These are implementation-in-progress, not yet
  smoke-tested or scientifically validated.
- Scientific decisions encoded: FixFlip curated labels evaluation-only; include
  unlabeled patients in TCGA split assignment; PAM50 four-class primary task;
  explicit clinical allowlist/missingness; training-only QC/HVG/IDF/PCA/LSI;
  Scrublet diagnostic-only with residual-doublet caveat; TSS fragment fraction
  is a proxy, not conventional TSS enrichment.
- Next: H100 runtime acceptance, synthetic adapter/CLI smoke tests, real-data
  inspection/splits, then commit a protocol before any real pilot.


## 2026-11-02 — Phase 2 preprocessing and adapter validation in progress

- Protocol pre-registered before any real pilot: commit 4b6f48f. Deadline
  November 10; no representation performance informed the protocol.
- CPU preprocessing job 5003346 completed PBMC inspection/split/preprocessing,
  then failed TCGA profile serialization because multiplicity keys were integers.
  Fixed keys to JSON strings. Job 5003352 completed TCGA preprocessing and ran
  synthetic adapter tests: 14 passed; MultiVI/MOFA API failures retained in logs.
- PBMC QC retained 10,243 cells (7,171 train/1,532 validation/1,540 test):
  10,078 curated labels, 165 vignette-filtered cells retained, 334 curated-labeled
  cells rejected by Agent3 QC. LSI first component depth correlation 0.93236;
  dropped by the pre-registered |r|>0.5 rule. Scrublet training diagnostic found
  9.9% predicted doublets at auto threshold ~0.18; no automatic doublet exclusion.
- TCGA: 1,132 eligible primary slides/1,061 patients; 802 four-class labeled
  patients (LumA 410/LumB 190/Basal 137/Her2 65), 23 Normal-like excluded from
  primary classification; 872 complete-clinical and 665 labeled+complete patients.
  All 1,061 eligible patients, including unlabeled, receive new Agent3 splits.
- Clinical cross-check: full-case PAM50 counts match the reference audit. Agent3
  endpoint policy yields 1,081 OS-eligible cases versus reference 1,075, with
  the same 149 deaths. Six additional censored cases have reference times missing
  and Agent3 follow-up durations 394–677 days. Differences are saved locally in
  preprocess/os_discrepancies.json and documented in DATA_PROVENANCE.md. No
  reference phenotype inputs/cohorts/splits were adopted.
- MultiVI 1.4 setup_anndata is a deprecated no-op; switched to genuine MuData
  rna/atac registration with explicit training/validation indices. Its train API
  internally supplies early_stopping_patience=50, conflicting with a caller
  override; implemented best-validation state capture/restore and stopping via
  a real Lightning callback. Job 5003353: MOFA pass, MultiVI callback conflict.
  Job 5003354: MultiVI passed training/encode/save/load. MOFA training statistics
  use the actual finite ELBO trace rather than a nonexistent number_iterations key.
- Four local FMs passed strict CPU weight matching and image forward tests in
  job 5003358 (exit 0, 1m04s), including matching projected image/text spaces for
  CONCH-Z and MUSK. Receipts: logs/preflight/cpu_model_diagnostic.json.
- GPU runtime job 5003345 remains pending. It was moved from h100_short to
  h100_priority without changing its ID, one-GPU allocation or 20-minute cap.
  The GPU ledger records partition changes and sacct elapsed time; used time
  is zero while pending. CPU checks do not replace H100 bf16 acceptance.
- Official torchvision ImageNet ResNet18 weights fetched to Agent3's .cache/torch,
  hash-prefix verified and full SHA256 recorded. No pathology weights redownloaded.
- Full synthetic smoke job 5003359 is running: real adapter training including
  actual pretrained last-block/image-text updates, tiled TIFF WSI tests, two
  complete synthetic pipelines, validated resume, unseen-modality and objective
  regression checks, PDF claim checks. No smoke acceptance gate is asserted yet.
- Implemented modular stage orchestration, failure-isolated workers, validation
  selection/tuning/final freeze, representation export, distinct evaluation
  categories, sensitivities, evidence-only reports and four-page project report.
  These remain implementation-in-progress until the full smoke/validators pass.
- Next: fix any full-smoke failures, complete H100 acceptance, add/verify the
  dependency-chained production launcher, then run all eligible real pilots and
  downstream analyses under the declared budget. Real-data models have not run.


## 2026-11-02 — H100 accepted; complete synthetic workflows passed; pre-production audit

- H100 job 5003345 completed exit 0 in 64 seconds on h100_priority: NVIDIA H100
  80GB, CUDA 12.8, native sm_90, bf16 matrix forward/backward, strict weight loading
  and random forward for all four pathology FMs; projected image/text dimensions
  match for CONCH-Z and MUSK. Actual cost ~0.01778 GPU-hours, reconciled in ledger.
- Full smoke 5003359: all 20 unit/adapter tests passed, including actual pretrained
  tower updates and save/load. The synthetic count pipeline reached report
  generation, then its validator flagged its own legacy-prefix search string.
  Fixed that false positive without weakening the path check.
- Full smoke 5003361 completed exit 0 in 8m34s: 20 tests passed; paired-count and
  image-bag/tabular workflows each completed inspection through validated PDFs,
  with all eligible synthetic candidates successful. A second run validated every
  stage checksum and skipped completed work. The automatic gate/evidence receipts
  were generated by the passing job, not written manually.
- CPU/FMs and H100 acceptance are complete; real-data model pilots remain unrun.
  Synthetic PDFs use smoke_* filenames and are excluded from production submission.
- Pre-production review clarified multi-primary selection: structure-equivalent
  candidates must also satisfy the next primary metric before complexity/time
  ties; whole-second runtime granularity prevents subsecond noise. Added a
  regression test. The protocol clarification was written before real pilots.
- Added typed RunResult validation, real process peak RSS, explicit save-failure
  status, repeated final seeds for stochastic ABMIL, replicated LR tuning with
  post-tuning validation reranking, primary-metric-specific bootstrap intervals
  and verdict checks, and stronger feature-store content resume.
- Actual tower models now refit for receptor removal/site sensitivity. Selected
  and reference adapter stability use matched training-subsample perturbations
  conditional on fixed train-only preprocessing; this limitation is explicit.
- Added canonical R Seurat/Signac train-only WNN cross-check, local objective-swap
  plan demos, bounded two-shard WSI extraction, dependency-chained controller,
  GPU reservation accounting for array tasks, protected-directory comparison and
  allowlisted submission packaging. These additions are undergoing final smoke
  validation before real pilots; the earlier gate is stale for the new code.
- Reports expose pilot attempts/failures, per-class metrics, primary comparison
  intervals, confounding, identifiers/hashes and actual tower capacity caveats.
  Numeric sources remain saved artifacts. Source/report code hashes are separate.
- Next: commit this pre-production implementation, run the current full smoke,
  then start the one-command workflow; no GitHub push or Canvas submission.

## 2026-11-03, 00:15 Eastern — current-code smoke passed; production submitted

- Full smoke **5003412** completed exit 0 in 11m00s: 24 tests passed, including
  actual pretrained tower updates, marker-file delimiter handling, multi-primary
  selection and uncertainty checks. Both synthetic count and image-bag/tabular
  pipelines completed through validated PDFs; every eligible synthetic candidate
  succeeded. A second pass validated checksums and resumed all completed stages.
  The job generated the current-code acceptance gate and its evidence hashes.
- Pre-production source commit: `2ae5c85` (following protocol commit `4b6f48f`).
  This smoke validates the implementation; it is not real-data scientific evidence.
- `bash scripts/submit_workflow.sh` initially failed with QOSMinCpuNotSatisfied.
  `sacctmgr show qos small` confirmed a minimum of two CPUs. The controller now
  requests two CPUs and the launcher exposes Slurm stderr. Retry submitted
  controller **5003413** successfully; no duplicate job was created by the failure.
- H100 acceptance remains passed (5003345). Reconciled GPU use before production:
  **0.01778 GPU-hours**, with a 48-hour aggregate cap and two-GPU concurrency cap.
- Generated-output packaging now includes the saved objective-swap plan demos.
  Production phases will build training-only R WNN references and objective swaps,
  run PBMC pilots/evaluation, then WSI tiling, measured throughput, feature shards,
  TCGA pilots/evaluation and report/validation/archive finalization.
- All shared inputs/model folders and Agent1/Agent2 remain protected read-only.
  Final scientific results, final PDFs and submission archives are not ready yet.

## 2026-11-03, 00:26 Eastern — preprocessing complete; H100 queue busy

- CPU phase **5003414** completed exit 0 in 3m20s. Both datasets were re-inspected,
  split and preprocessed with the accepted analysis code. R Seurat/Signac WNN
  succeeded on 7,171 training cells, without labels; |first LSI/depth r|=0.9913,
  so its first component was excluded by the pre-registered rule. Input/output
  hashes, R command and sessionInfo are under baseline/r_wnn_crosscheck/.
- Two saved plan-only objective swaps passed: PBMC label permission changes
  supervised fusion from reference-only to eligible; TCGA survival-primary changes
  supervised losses to Cox partial likelihood and primary metrics to C-index.
  These are planning demonstrations, not claims of survival-primary execution.
- PBMC H100 job **5003415** is queued. At inspection all 16 H100 GPUs were allocated
  to other jobs; Slurm forecast November 5, 00:00 Eastern. Both H100 priority and
  preempt partitions currently have PriorityTier=5000, and the cluster uses
  partition-priority preemption; the priority partition name alone does not
  confer a higher tier. No QoS outside this account's allowed normal QoS was used.
- To use the wait productively, reordered the independent CPU WSI index phase
  ahead of waiting for PBMC. Stopped only controller 5003413, preserving pending
  GPU job 5003415 and state, then resumed controller **5003417**. It reused passed
  smoke/preprocessing and submitted WSI CPU indexing **5003418** without duplicate
  GPU work. The two-GPU/48-hour limits and scientific budgets are unchanged.
- Presentation-only improvements expose saved alignment, modality masking,
  seed/subsample stability, sensitivity metrics and survival confidence intervals
  in readable tables. LumA/LumB descriptive F1 is an arithmetic summary of frozen
  confusion matrices, with no refit or selection. Numeric source maps cover these
  tables. Separate CPU job **5003416** passed on copies of both accepted fixtures
  (8/6-page PDFs, 113/98 verified claims). Original smoke evidence was preserved;
  the analysis hash still exactly matches the accepted smoke gate. Ruff passed.
- Next: finish WSI CPU indexing; H100-dependent pilots and feature extraction
  remain queued. No real representation performance or final report is asserted.

## 2026-11-03, 00:29 Eastern — coordinate-resume integrity repair before training

- Review found two concrete WSI resume gaps: coordinate NPZ files were absent from
  the tile-stage artifact map, and the per-slide cache did not check the saved NPZ
  checksum or coordinate-tool source. Fixed both. Tool fingerprints cover seeding,
  slide metadata and tissue-coordinate generation; artifact receipts include NPZs.
- Added a regression test that preserves an unchanged cache but regenerates a
  corrupted or deleted coordinate file. Both WSI tests passed; ruff passed.
- Stopped this project's controller 5003417 and partial CPU index 5003418, and
  canceled pending GPU job 5003415 before it received any allocation. No shared
  input or scientific result was removed. GPU use is unchanged at 0.01778 hours;
  the canceled pending job used zero GPU-hours. All old receipts are retained.
- This analysis-code fix correctly invalidates the earlier smoke gate. A fresh
  full smoke is required before the workflow is resumed. No real-data model
  fitting, tuning or test evaluation had started, and scientific rules are unchanged.

- Fresh full smoke **5003419** is running; all **25** unit/adapter tests passed.
  End-to-end fixture execution and validated resume must finish before acceptance.
- The Slurm controller now records content receipts for completed phase outputs,
  rechecks them on resume, fingerprints the actual production configs/objectives,
  and refuses to duplicate active jobs after a code/config change. A direct check
  validated 76 preprocessing artifacts and rejected interrupted and corrupted
  output fixtures; ruff and Python compilation passed for the controller scripts.
- Added a bounded WSI coordinate-QC stage: up to three training slides chosen by
  objective-power/rounded-MPP strata, without labels. It will save tissue-mask and
  coordinate overlays plus small region-QC panels under Agent3/features only.
  Synthetic QA PDFs are ignored and excluded from production archives.

- Added a final-stage PDF preview renderer, exercised successfully in CPU job
  **5003421**. It renders every page and explicitly leaves manual visual approval
  pending. Inspected sample pages of the synthetic QA reports: readable tables
  and figures; found isolated table headers near page breaks. Added semantic
  repeating table headers and kept headings with following content. These are
  report-only changes; the running smoke will validate current report hashes on
  its resume pass. Final real PDFs/figures still require visual review.
- The controller now queues the independent one-GPU PBMC and WSI-throughput work
  after their respective CPU prerequisites, retaining the two-reserved-GPU cap.
  Both must finish before the two-shard feature array is submitted. This improves
  queue overlap without changing pilot access, scientific budgets or thresholds.

## 2026-11-03, 00:42 Eastern — integrity-fixed version accepted and resumed

- Full smoke **5003419** completed exit 0 in **10m44s**. All 25 tests passed,
  every eligible synthetic candidate succeeded, and both pipelines completed
  through validated PDFs. The resume pass correctly re-rendered the count report
  after the report-only table-header change and revalidated it. All unchanged
  analysis stages were checksum-validated and skipped.
- Current accepted analysis SHA256:
  `d2674d7f7af7d147bcb29cfd281e6100cae00c37ecc8575e8c9689175bc14909`.
  The automatic gate references both final fixture validation hashes and job ID.
- `bash scripts/submit_workflow.sh` submitted controller **5003422**. It will
  preserve historical state, refresh preprocessing for this accepted code,
  independently queue PBMC and training-only WSI throughput after CPU prerequisites,
  wait for both, then submit the feature array, TCGA modeling and finalization.
- Current source commits include coordinate integrity fix `392beb3`, phase
  receipts/coordinate QC `6efc3f2`, and preview/queue-overlap changes `1fd68ea`.
  No real-data training, tuning or held-out evaluation has occurred yet. H100
  queue occupancy is an external scheduling constraint; no extra quota was requested.

## 2026-11-03 — refreshed CPU phase and report/artifact review

- Current-code preprocessing **5003425** completed exit 0 in 3m20s, including the
  R WNN reference and plan swaps. The controller verified 76 phase artifacts.
  PBMC GPU **5003427** is pending; WSI CPU indexing **5003428** is running.
- Independent CPU artifact QA **5003426** completed exit 0 in 1m22s: all protected
  datasets, four model folders, Agent1/Agent2 and their environments match the
  initial metadata snapshots. Current synthetic PDFs were rendered and all
  15 pages (9 count + 6 image/tabular) visually inspected. Tables repeat across
  pages, text/figures remain readable; the count fixture has a short final source
  paragraph on its last page. Actual reports still await real-data computation.
- Report-only clarification now states exactly which frozen runs use a
  training-pair Ridge alignment probe and which use native shared latent spaces.
  CPU QA **5003430** passed on fixture copies: 113/98 claims verified. Original
  accepted evidence and analysis hash were preserved.
- Submission allowlists now retain stage receipts, pilot budget, post-tuning
  decision, preprocessing/QC summaries, R sessionInfo/reference receipts and WSI
  throughput/tiling JSON summaries. Raw matrices, tile/feature arrays, model
  weights and checkpoints remain excluded. Actual archives have not been built;
  finalization will verify every included file, checksums and archive sizes.

## 2026-11-03 — WSI indexing complete; degenerate-diagnostic correction

- WSI CPU job **5003428** completed exit 0 in **17m20s**: all **1,132** primary
  slides produced validated coordinate stores, with zero tiling failures. Three
  bounded training-slide tissue/coordinate/region-QC panels were generated.
- Synthetic visual QA exposed a constant mitochondrial-fraction target reported
  by sklearn's finite-score convention as R²=1. This is not evidence of confounding.
  Changed constant/insufficient/non-finite numeric diagnostics and unidentifiable
  site diagnostics to explicit not-applicable records, with no fabricated score.
  Site chance is now explicitly uniform over training-site classes (1/K).
  Pre-registered primary metrics, thresholds and fit budgets are unchanged.
- Three targeted regression tests passed: constant QC/single-site cases,
  preserved variable-QC predictability and training-class chance, and unseen sites.
  Ruff passed. The clarification was recorded before real model fitting.
- Stopped only controller 5003422 and pending GPU job 5003427 (zero GPU use), while
  allowing WSI indexing to finish with its original code signature. A fresh full
  smoke is required for the diagnostic correction. Existing coordinate-tool code
  and per-slide signatures are unchanged, so the completed coordinate stores can
  be content-validated and reused after this unrelated evaluation change.

## 2026-11-03, 01:30 Eastern — final accepted implementation; GPU work queued

- Full smoke **5003437** completed exit 0 in **9m52s**: all **28** tests passed,
  both genuine adapter/complete fixture workflows reached validated reports,
  and validated resume skipped every unchanged stage. Constant QC is now reported
  as not applicable rather than a misleading perfect R². Current accepted analysis
  hash: `bb6b6aba223fbb128a4ad0f75eb78d3edd9d8e12b31e57fec7b5fcf98debce71`.
- Resumed production controller 5003446, then restarted only that CPU controller
  as **5003455** to load the final phase-receipt logic. Existing GPU jobs were
  preserved: PBMC **5003451**, WSI throughput **5003454**. No duplicate GPU job.
  Stable H5 feature hashes/selection/tiling metadata, rather than a later-rewritten
  aggregate progress receipt, now establish array-phase resume validity.
- Current-code preprocessing/R WNN **5003447** completed exit 0 in **3m22s**.
  WSI content validation **5003452** completed exit 0 in **17 seconds**, reusing all
  1,132 coordinate stores from 5003428. Three training-slide QC panels were
  inspected; title margins were corrected and the first regenerated panel checked.
  Controller receipts cover 76 preprocessing and 2,299 WSI-stage artifacts.
- Report-only fields now expose saved cell/QC flow, explicit threshold scales,
  LSI-depth decision, doublet diagnostics, WSI physical parameters and measured
  throughput/tile-cap calculations. CPU QA **5003448** passed: 132/108 numeric
  claims verified on fixture copies. The WSI rendering branch uses explicitly
  labeled synthetic metadata only in its QA fixture; it is not real throughput
  evidence. Preview job **5003456** succeeded and the new sections were inspected.
- At the latest check, both GPUs are pending. Slurm forecasts moved earlier to
  **November 3, 13:11:09 / 13:21:29 Eastern**, subject to scheduler changes.
  Reconciled actual use **0.01778 GPU-hours**; committed **7.01778**, cap 48, with
  two reserved GPUs. Canceled pending attempts consumed zero GPU-hours.
- Scientific state: implementation/smoke/preprocessing/coordinate indexing complete;
  real-data pilots, final training, held-out results, final reports and archives
  remain pending. The controller will continue the dependency chain. After actual
  report generation, every real page/figure still needs visual QA and the final
  archive/source handoff review. README records the current queue and handoff.

## 2026-11-03 — real PBMC complete; measured WSI scheduling and production

- Throughput **5003454** obtained H100 early and completed exit 0 in **1m16s**,
  before the earlier scheduler forecast. Twenty training slides; pipeline rate
  20.428 tiles/s, encoder-only rates UNI2-h 337.142, CONCH-Z 443.869 and ImageNet
  1587.785 tiles/s. Initial cap256 projected 7.881 aggregate hours with safety2.
- PBMC **5003451** also started early and completed exit 0 in **30m29s**. All
  four baselines and 12 eligible learned-candidate pilot runs succeeded. Frozen
  selected CLIP representative: macro-F1 **0.925014**, balanced accuracy **0.917368**,
  paired recall@10 **0.497078**. RNA-only macro-F1 **0.909449**. Selected-minus-RNA
  F1 **0.015565**, paired CI **[-0.008915, 0.060450]**; recall@10 difference
  **0.069156**, CI **[0.050154, 0.089627]**. Verdict **integrate_with_caution**:
  F1 gain remains uncertain and selected embedding predicts QC/depth with R²>0.5.
  No test result was used to retune or change the frozen PBMC analysis.
- Current H100 association inspection found per-user priority GrpTRES gpu=1.
  Kept PBMC running, canceled only unstarted priority feature array **5003478**
  (zero GPU-hours), and selected the largest measured cap fitting short-queue
  limits: **128**, 1.970 h/shard plus 90 s startup <=2 h. This happened before any
  TCGA model fitting. Original measurement SHA256
  `2e52e55c954c9d76f9b497a42c15812d40e17d11d4044bed52ef932ff311eed9`
  is preserved byte-identically; the controller migrates the existing receipt
  only after proving this hash equality. Raw measurement remains immutable.
- Feature array **5003507** uses h100_short, 120m/task, two shards. Controller
  **5003508** reuses it and the completed PBMC run. Initial throttle1 permits
  overlap with PBMC; after PBMC completion a locked check raised throttle2 without
  resubmission or changing tile cap/data. Both tasks always count toward hours.
  Expanded sacct queries use parent ID + --array so unstarted canceled child
  tasks release their zero-time reservations correctly. Four operational tests
  passed, including rejection of an unsafe throttle increase while PBMC is active.
- Actual feature files were inspected: common coordinates with UNI2-h 1536,
  CONCH-Z 768 and ImageNet 512 columns, complete H5 flags and <=128 retained tiles.
  At one progress check, 264 slides had all three stores. No failed slides were
  silently removed. Subsequent progress is in the two array logs.
- First real PBMC report job **5003520** passed rendering, scientific/source
  validation and protected-directory comparison. A direct afterok submission was
  rejected because the completed job had already been purged from scontrol;
  verified sacct COMPLETED/0 and submitted the CPU report without that stale
  dependency. Added display-only embedding dimensions and explicit count-MLP names;
  **5003539** re-rendered and validated successfully, with **265 claims / 12 pages**,
  then rendered every page for visual review. Frozen models/results were untouched.
- QA **5003512** passed copied-fixture report checks, including operational queue
  fields (132/112 claims before adding dimension columns). Core analysis gate
  remains exactly bb6b6aba...debce71. Source/environment/model/input boundaries
  remain protected. Remaining: PBMC visual review, both feature tasks, all TCGA
  pilots/final/sensitivities, report2/project report, final validators and archives.

### 2026-11-03 — frozen-output report review and production extraction progress

- Feature task **5003507_0** completed successfully in **46m21s**; task 1
  continues. At a later check, **1,016/1,132** slides had all three production
  feature stores. No feature failures have been silently excluded. Controller
  **5003508** remains responsible for the TCGA model phase after extraction.
- The PBMC report's first 13-page revision was reviewed page by page, including
  every embedded scientific figure. Cleaned encoder display names and baseline
  modality definitions, expanded heatmap/UMAP captions, and explained degenerate
  rare-class empirical recall intervals. Reporting now also derives QC-colored
  joint and RNA/ATAC tower UMAP displays from **frozen** checkpoints/embeddings;
  each map fits training embeddings and transforms held-out embeddings. Their
  source/output hashes and coordinates are stored in the evidence bundle.
  Modality separation remains visible; this is not presented as full mixing.
- Report job **5003595** was interrupted by the stage integrity guard because
  project-report source was edited while validation was running; no model or
  test evidence changed. Strict-shell retry **5003597** completed report/validate,
  protected-directory comparison and preview rendering: **265 claims, 13 pages**.
  The core analysis hash and frozen PBMC plan remain unchanged.
- Extended source checks to the displayed cohort counts, pilot attempt counts,
  final runtime/steps, structure/confounding scores, graph connectivity and
  RNA-to-ATAC results. Job **5003613** rendered **377 claims / 13 pages**; its
  full validation and refreshed visual previews are still pending at this entry.
  The final limitations block is kept together and the ledger's reconciliation
  timestamp is explicit. No representation fitting or metric retuning occurred.
- The four-page project-report template now includes both objective primary
  metric differences against both references, with source-linked paired CIs.
  Synthetic layout/source QA **5003614** is pending; outputs are isolated and
  explicitly marked synthetic, never substituted for the real project report.
- Reconciliation at **08:21:18 UTC** recorded **1.778611 GPU-hours used** and
  **3.319444 committed**, with one GPU slot active, within the 48-hour/two-GPU
  limits. Later totals must be read from the reconciled ledger. Source lint
  passed before the final report-fixture extension; final lint will be repeated.
- Remaining: finish production features, all real TCGA model/evaluation phases,
  final reporting/visual review, protected checks, archives and Git handoff.

### 2026-11-03 — explicit WSI artifact repair; frozen PBMC retained

- CPU report **5003613** passed report/validate, protected-directory comparison,
  **377 numeric claims / 13 pages**, and rendered every page. All 13 actual pages
  and embedded figures were visually reviewed. Receipt `reports/visual_review.json`
  binds the review to PDF SHA256
  `afab9a07741716102fb8f06a99b00cf0a6dea9b107aa0a360d805e793af4bd07`.
- Copied-fixture CPU QA **5003614** passed **204/174** numeric checks and the
  synthetic project report's **4-page** cap. Full source/script/test lint passed.
- Feature task **5003507_1** completed 565 slides but failed on one explicit
  zero-pass-region-QC slide. Controller **5003508** stopped and did not submit
  TCGA models. First-array results and failure receipt are preserved. Together
  with task 0, **1,131** slides succeeded; none was silently dropped.
- Diagnostic **5003624** showed all 256 original candidates were in a large,
  nearly constant red edge block (maximum blur variance 0.077). The raw thumbnail
  also contains real tissue above it. No target labels or model scores were
  inspected to make the repair decision.
- CPU repair **5003629** masked only qualifying uniform saturated edge components
  for thumbnail segmentation, regenerated this one derived coordinate store,
  diagnosed it, then ran the normal tile CLI successfully. Raw slide unchanged;
  old JSON/NPZ archived. Explicit coordinate_repair metadata retains base input
  identity plus postprocessor source/hash and old/new coordinate hashes.
  New candidates: **120/128** production / **244/256** diagnostic pass the same
  region-QC thresholds. The corrected tissue gallery was visually inspected.
  Cohort, split, tile cap and encoder/model rules are unchanged.
- Two new synthetic tests passed in **0.85 s**, checking artifact removal with
  preservation of textured content, and refusal when no qualifying artifact
  exists. Added a single bounded repair path to the Slurm feature wrapper with
  first-attempt archival; other failure types still stop. No core source file
  changed, and the existing bb6b6aba...debce71 smoke gate/PBMC freeze remain valid.
- Resumed controller **5003630** validated tiling via **5003631**, then queued
  recovery array **5003632**. The normal current-code resume path validates the
  original 1,131 feature sets rather than re-encoding them. Pending two-hour
  requests initially forecast 04:44 Eastern. They were safely shortened to
  **15 minutes/task**, verified by scontrol and synchronized under the same ledger
  lock with history and an operational receipt. Task 0 then started promptly.
- Ledger at **08:55:42 UTC**: **2.129722 GPU-hours used**, **2.624444 committed**,
  at most two slots. Remaining: recovery-array completion, TCGA experiments,
  final reports/validators/visual review, archives and final Git handoff.

- Recovery task **5003632_0** subsequently completed cache validation in **43 s**.
  Task 1 remained queued: all **16/16 H100 GPUs** were allocated. Since PBMC has
  released the priority association, moved only that existing pending task to
  `h100_priority` (same account, 15-minute limit). Slurm read-back and locked
  ledger reconciliation verified the change; original submission and partition
  histories remain recorded. Forecast was 04:54 Eastern, subject to scheduling.

### 2026-11-03 — feature recovery complete; TCGA submitted

- **5003632_0** completed in **43 s** and **5003632_1** completed in **1m15s**,
  both exit 0. Each shard has 566 results and zero unresolved failures; the
  intersection of UNI2-h/CONCH-Z/ImageNet stores is **1,132/1,132** slides.
- Controller **5003630** validated feature receipts and submitted the real TCGA
  baseline/pilot/final/sensitivity job **5003644**. It is resource-queued in
  h100_priority. No TCGA held-out performance has yet been read.
- Reconciled ledger at **09:06:53 UTC**: **2.157222 GPU-hours used** and
  **14.157222 committed**, one slot reserved. The 12-hour model allocation fits
  the approved 48-hour envelope; actual use will be reconciled after completion.
- Added primary method/data citations and pinned model-card references.
  Synthetic CPU QA **5003645** passed **204/174** dataset claims and a project
  report with **4 body pages + 1 separate reference page**. The validator now
  counts the explicitly separated reference section independently, as allowed by
  the assignment, while rejecting a fifth main-body page. No experiment changed.
- An optional proposal to hold the pending GPU job and add a separate CPU baseline
  dependency was rejected by automatic approval review, citing concurrent writes
  under the active controller and workflow/freeze risk. The command did not run.
  No hold, CPU baseline job, or dependency change was applied. The existing single
  workflow continues unchanged; no user approval is needed for that safer path.
- Remaining: actual TCGA model/evaluation phases, final reports and visual review,
  final protected checks, submission archives and local Git handoff.

- Report fixture QA **5003656** subsequently passed with **4 main-body pages +
  1 reference page + 1 planning appendix**. The appendix reads the existing
  objective-swap records and explicitly claims no training or held-out outcomes
  for those alternative objectives. References and appendix boundaries are
  independently detected by the PDF validator.
- The anomalous raw slide was independently checked against its acquisition
  manifest: **41,071,550 bytes**, MD5 **43dca16b547f4b734d20c93e7de61523**,
  both exact matches. The small read-only check and its timestamp are recorded
  in its QC diagnosis directory and copied into the report evidence summary.
  No redownload or raw-data mutation occurred.

### 2026-11-03 — TCGA fits complete; lossless export compatibility recovery

- Real H100 job **5003644** completed all **8 baseline runs**, **21 learned
  candidate runs** (15 neural/contrastive plus 6 genuine CONCH-Z tower fits),
  validation selection, replicated narrow tuning, and **11 final runs**.
  All fits succeeded. Validation selected **fusion_late**; no test metric was
  consulted for selection or tuning. Its image branch is gated attention MIL on
  frozen UNI2-h tile features; its clinical branch is an MLP, with decision-level
  fusion and a concatenated representation for the common logistic probe.
- The final plan was frozen with SHA256
  `16645145e8aca32f30b62bd158bd59528707b4db26850a0eb32cd012cb75b1ae`.
  Embed produced seven NPZ files, then Arrow rejected mixed NumPy float64/float32
  arrays in the object-valued embedding column. **5003644** failed after **30m34s**;
  controller **5003630** stopped. No TCGA held-out metric had been evaluated.
- Added a scoped production CLI IO adapter (`scripts/agent3_cli.py`) that forwards
  the original CLI and normalizes only floating embedding vector cells to Python
  lists for list<double> inference, retaining float64 precision. It checks exact
  Parquet round trips and records source/version/output hashes. The unchanged
  original stage executor determines completion; no success flag or frozen plan
  was hand-edited. The Slurm run entrypoint uses the adapter. Model/preprocessing/
  selection source and the bb6b6aba...debce71 automatic smoke gate are unchanged.
- Two regression checks passed in **0.24 s**, including float64 values that a
  float32 cast would lose and ordinary Parquet columns retaining their types.
  Initial CPU QA **5003695** stopped at a new-script lint import-order error;
  corrected it. Isolated real-vector QA **5003696** passed: **7 models / 7,427
  rows**, exact numeric values, IDs and splits; both frozen plans and all **11**
  TCGA checkpoint hashes unchanged. No package or model installation changed.
- Before recovery, recorded exact per-array dtype/shape/content fingerprints and
  frozen checkpoint/plan hashes in `logs/frozen_before_vector_export_fix.json`.
  The finalizer now independently verifies these against the actual production
  Parquet file, with the resulting proof included in the generated archive.
- Resumed only the existing single workflow: controller **5003697**, model-stage
  continuation **5003701**. Content-valid completed stages/checkpoints are reused;
  no model retuning or refitting is needed for the IO repair. Remaining: verify
  export, held-out evaluation/downstream/verdict, reports/visual review, final
  protected checks and archives.

### 2026-11-03 — complete patient/slide export QA; same frozen continuation

- **5003701** reused every stage through final training and passed the actual
  patient Parquet conversion (**7,427 rows**, exact round trip). It then stopped
  at Arrow's empty-struct restriction in the slide table's encoder receipt
  (`remote_code_sha256` is legitimately empty for the standard timm encoder).
  GPU use was **1m05s**; controller **5003697** stopped. No TCGA metric was read.
- Extended the same IO component to encode only encoder-receipt dictionaries as
  canonical JSON strings and verify parsed equality. Empty dictionaries remain
  empty; no dummy fields are added. Numeric data and model decisions do not change.
  Three serialization regression tests passed in **0.28 s**.
- Strengthened the isolated real-data check to mirror all patient-export scalar
  fields, then read frozen feature stores to construct/check **all 1,132 slide
  vectors and complete encoder receipts**. CPU **5003719** passed in **15 s**:
  exact patient/slide values, IDs, splits, metadata, both frozen plans and 11
  checkpoints, plus unchanged original NPZ array fingerprints. Its proof is in
  outputs/export_qa/validation.json. No production stage files were written by QA.
- The original 12-hour recovery reservation had been safely reduced to one hour
  while pending, verified in Slurm and synchronized under the ledger lock; no
  dependency change or parallel analysis was submitted. The controller now makes
  this distinction automatically: it validates every stage/artifact/signature
  through final training and the freeze receipt before using a 60-minute
  continuation; otherwise the full 720-minute training reservation remains.
- Resumed the same workflow with controller **5003724**. Source model/preprocessing/
  selection code and its automatic smoke hash remain bb6b6aba...debce71. Frozen
  models were never retuned. Remaining: complete normal export/evaluate/downstream/
  verdict stages, final reports/protected checks/visual QA and archives.

### 2026-11-03 — both real analyses validated; final presentation review

- Recovery controller **5003724** completed normally. GPU **5003726** completed
  export/evaluate/downstream/verdict in **5m08s**; CPU finalizer **5003732** passed
  in **3m51s**. No frozen model was refit or retuned after testing.
- Production frozen-export audit passed exact patient/slide vectors, metadata,
  identifiers and splits; both frozen plans, all 11 TCGA checkpoints and all
  original NPZ array fingerprints remain unchanged. Analysis SHA remains
  bb6b6aba223fbb128a4ad0f75eb78d3edd9d8e12b31e57fec7b5fcf98debce71.
- Both dataset validators passed, including protected input/model/Agent1/2/env
  snapshots, train-only fits, disjoint splits, validation-only selection, frozen
  access, provenance and verdict recomputation. Reports verified **377 PBMC**
  and **353 TCGA** numeric claims. Initial actual PDFs: **13 / 17 / 6** pages;
  project report has **4 body + 1 references + 1 appendix** pages.
- TCGA representative late-fusion macro-F1 **0.722860**, balanced accuracy
  **0.713978**; difference versus histology MIL **0.078179**, paired 95% CI
  **[-0.008742, 0.169677]**. Source-site prediction remains above the caution
  margin. Receptor-removal F1 **0.582671**, separate site-held-out F1 **0.678648**;
  survival C-index **0.664661** (CI **0.557732–0.770457**, 30 test deaths).
  Both datasets return **integrate_with_caution**; no generalization/clinical
  benefit claim is supported.
- Every actual project/TCGA PDF page and embedded figure was visually inspected.
  PBMC final changed pages 5/12/13 were inspected; the other ten page PNGs match
  the previously fully inspected PDF byte-for-byte. Hash-specific records are
  retained in reports/visual_review.json. Three presentation fixes are being
  applied: unclipped architecture borders, wider decision-table labels and
  readable three-reference comparison charts; TCGA receives a source-site legend
  chosen from training frequencies and an explicit exploratory survival caption.
  These report-only changes leave all frozen analysis evidence unchanged.
- GPU accounting reconciled to **2.770278 aggregate hours**, zero GPU reservations,
  within 48 hours / two concurrent GPUs. Initial archives passed ZIP integrity
  and <1 GB size checks. Final docs/visual proof will be included in refreshed
  archives. Manual review is accepted only when explicit PDF/page hashes match;
  preview generation cannot grant approval. Remaining: rerender/check affected
  pages, commit final deliverables, refresh/verify final archives and handoff.

### 2026-11-03 — report-ready release and handoff

- Report refresh controller **5003745** / CPU finalizer **5003747** completed.
  Both scientific validators passed again, including protected snapshots. No
  analysis source, model, selection rule or test metric was changed. Ruff passed
  across source/scripts/tests; the automatic smoke remains valid for the exact
  unchanged analysis SHA.
- Actual final PDFs and embedded figures were inspected, with byte-identical
  pages carried forward from the already inspected render only on exact PNG
  SHA256. Newly inspected PBMC pages **5/7**, TCGA **5/7/17**, project **1–4**.
  Architecture borders, decision-table labels, reference charts, site legend and
  survival caption now render correctly. All three PDFs have completed explicit
  hash-specific review records. A verifier checks records against actual PDF
  and page bytes, rather than treating previews as approval.
- Final reports retain **13 / 17 / 6** pages; main report **4 body + 1 references
  + 1 appendix**. PDF SHA256: PBMC
  dfc2a4559a9d (full hash in reports/qa_preview_manifest.json); TCGA 808c9753a74a;
  project 136fff9a7013. All 730 dataset numeric-source checks passed (377 + 353),
  as did the main-report source/page-limit checks. Both conclusions remain
  integrate_with_caution with the limitations above.
- The release commit includes the three reviewed PDFs, explicit inspection
  record, updated README/traceability and source. Repeating the same normal
  controller refreshes archives after the inspection-record hash changes;
  completed model/evaluation stages are content-validated and reused. Its final
  job IDs, completion state and visual-proof validation are recorded in
  logs/workflow_state.json. Archive integrity, sizes, per-file SHA256 and exact
  source/generated membership are in submission/manifest.json and SHA256SUMS.
- Deliverables: submission/source_code.zip, submission/generated_outputs.zip,
  reports/generated_report_1.pdf, generated_report_2.pdf and report.pdf. Raw
  shared data, tile features, checkpoints and model weights are excluded.
  No scientific work remains; remaining release operation is this final archive
  refresh/integrity check. Next external step is user review and Canvas upload;
  no GitHub push or external submission is authorized or performed.

### 2026-11-03 — preserve reviewed main-PDF bytes across finalization

- Archive refresh **5003750 / 5003752** passed scientific/frozen checks but the
  explicit visual verifier correctly rejected the newly rendered main PDF hash:
  all six page PNGs were **byte-identical**, while PDF container bytes changed.
  Both dataset PDFs remained byte-identical and accepted. No scientific result
  changed and this was not treated as automatic visual approval.
- Added scripts/build_project_report.py, a presentation-only resume wrapper.
  It fingerprints all source evidence/validation/freeze/reference/objective-swap
  files, report code/templates and every produced report artifact. Only exact
  input/output matches retain existing PDF bytes; otherwise the normal renderer
  and page-limit/source validator run. The Slurm finalizer uses this wrapper.
- Isolated CPU report/cache job **5003759** completed in **10 seconds** after the
  prior controller ended. Every main-report page PNG still matches the inspected
  version exactly. Its current PDF SHA256 is
  **870807248013f1580044adde9e747a9a43d4f5a84302fed2614722adf71eedb1**;
  an explicit hash-specific carry-forward inspection record was added. Cache
  receipts accompany generated outputs. All analysis code/checkpoints remain
  frozen. Final source/PDF/proof commit precedes the normal controller's final
  archive refresh; its completion/checksums remain authoritative in the workflow
  state and submission manifest. No scientific or presentation revision remains.

### 2026-11-03 — user-requested critical main-report rewrite; no new experiments

- Audited all eight external-review issues against saved metrics, source and
  worker logs. Rewrote only the main paper's presentation/interpretation, with
  compact serif academic layout and descriptive method names. Complete pilot
  tables distinguish validation from actual frozen test results; unadvanced
  methods remain explicitly unevaluated on test.
- Main-body corrections: receptor-free fusion **0.583** versus image-only MIL
  **0.645**; native-head **0.482** / Her2 **0 of 10** versus latent probe **0.723**
  / recall **0.700**; PBMC ARI regression with mixed NMI; no convincing incremental
  accessibility-prediction benefit; duplicate decoder/ridge identities; narrow
  FM coverage, sparse tuning data and operational rather than hard tile limits.
- Qualified two causal/factual claims: MultiVI has a custom validation callback;
  all three worker logs reached the 50-epoch cap with 25-epoch KL warmup, but
  saved curves are empty. The installed/API default is 500 epochs, not a claim
  that 500 is optimal here. The logistic probe is also unweighted; imbalance
  and branch-logit averaging are plausible contributors, not proven sole causes.
- First report-only CPU **5004060** stopped on a template dictionary/method name
  collision; corrected it without writing experimental outputs. Report-only CPU
  **5004061** succeeded. **118** numeric-source claims passed; the PDF has **4
  body + 1 references + 2 appendix** pages. All seven pages and embedded figures
  were actually inspected and recorded against their current hashes.
- Original main-report files are archived locally. Hash comparison against
  logs/report_revision_frozen_snapshot.json confirms both generated dataset
  reports and all captured frozen experimental artifacts are unchanged. Analysis
  SHA remains bb6b6aba223fbb128a4ad0f75eb78d3edd9d8e12b31e57fec7b5fcf98debce71.
  No training, fine-tuning, preprocessing, evaluation, sensitivity analysis or
  new UMAP fit was run. Only report generation, visual QA and archive packaging
  are in scope. Submission packaging is refreshed to contain this reviewed paper.

- Final report-only CPU **5004064** added sourced cohort/split counts and clarified
  that FixFlip labels fit only downstream probes. **124** numeric claims passed;
  the four-body-page limit still holds. Final page 1 was directly inspected;
  the other six page PNGs are byte-identical to their inspected versions.
  Frozen-input comparison passed for all 18 captured experimental files/PDFs.
  The report-only release record is logs/report_revision_release.json; the old
  full workflow state remains computational history, not a new experimental run.

### 2026-11-03 — user-authorized GitHub publication and linked submission PDF

- User explicitly authorized publishing core Agent3 material to
  YangzhenyuGao/BIOS774_Agent3_RepresentationLearning, excluding data, and adding
  repository / both study-report links at the very top of the main PDF.
- Sycamore has no usable HTTPS credential cache; SSH strict host verification
  is not configured. The connected GitHub app is authenticated as YangzhenyuGao
  and has repository push/admin permission. Publication uses that authorized
  connection, without copying or exposing a credential. A terminal SSH setup
  guide is in docs/SYCAMORE_GITHUB_ACCESS.md.
- Report-only CPU **5004096** passed: four body pages, 124 numeric-source checks,
  three actual first-page URI annotations. Page 1 was directly inspected; all
  six other page PNGs match their inspected versions exactly. Experimental
  artifacts and both study PDFs remain unchanged.
- Publication audit found unanchored data/ and models/ ignore rules also hid the
  actual source subpackages. Root-anchoring those rules includes all existing
  data loaders and genuine model adapters without changing their contents or
  the frozen analysis hash. Public source coverage is checked against every
  Python file under src/. No raw data or weights are included.
- Added compact aggregate-only examples for inspecting the frozen comparisons
  without data. Per-cell/patient records, embeddings, tile features and
  checkpoints stay local. Publication is a clean source snapshot based on the
  repository's existing initial commit, retaining remote history rather than
  force-pushing the local experimental history. Remote verification and exact
  public-file hashes are recorded in logs/github_publication_manifest.json.

- After fixing ignore scope, lint also sees the previously ignored source files:
  six pre-existing unused-import/import-order findings remain in the frozen
  loader/adapter code. They are not functional failures and are deliberately
  left unchanged to preserve the accepted analysis SHA. Changed report and
  publication helpers pass lint; every source module parses and is included.
