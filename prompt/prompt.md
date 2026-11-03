# Execution Prompt: BIOS 774 Project 3 — Objective-Aware Multimodal Representation Learning Agent

You are an autonomous coding agent working directly on **BIOS 774 Project 3: Building an AI Agent for Multimodal Representation Learning**. Read this entire prompt before changing files or launching jobs. Use it as the implementation contract unless the official assignment or a later explicit user instruction overrides it.

Build a rigorous, lightweight, reproducible system. The assignment emphasizes the design and behavior of the agent, not leaderboard performance. Do not turn this into an exhaustive multimodal benchmark or an elaborate product.

## 0. User-reviewable defaults

The user reviewed this prompt before execution. Unless the user edited a row below, treat each default as approved. If a row is marked `ASK`, stop at the point where it matters and ask; do not invent an answer.

| Item | Default |
|---|---|
| Due date | Not stated in the pasted assignment. `ASK` before final compute planning; do not invent a deadline. |
| Report numbering | `generated_report_1` = PBMC Multiome (RNA + ATAC); `generated_report_2` = TCGA-BRCA (WSI + clinical). |
| PBMC evaluation labels | The Satija lab's curated cell-type annotations for this sample (`celltype` in `annotations/pbmc_10x_annotation_FixFlip.csv`): 19 types for 10,412 cells. They are used only for evaluation, never to fit the primary representation. They are expert labels, not measured ground truth (Section 5.1). |
| TCGA primary target | PAM50 intrinsic subtype (`PAM50Call_RNAseq`, primary tumor sample code `01`), 4 classes (LumA, LumB, Basal, Her2). Normal-like is excluded from the primary task because it is very rare and often reflects normal-tissue content; its count is reported. Overall survival is secondary and exploratory. |
| TCGA clinical inputs | Primary input set: age at diagnosis, AJCC pathologic stage group, histological type, and ER/PR/HER2 clinical receptor status, each with explicit missingness indicators. Sensitivity set: the same without receptor status. |
| Pathology foundation models | Use the user's local model store (Section 1.3): exactly four models: UNI2-h, Virchow2, CONCH-Z and MUSK. Verify each against its manifest, and never download a second copy. Do not use any other pathology FM, even if more weights appear in the store later. A clearly labeled ImageNet-pretrained backbone is the non-pathology baseline encoder. |
| GPU budget | Sycamore H100. At most 2 concurrent GPUs; an aggregate cap of 48 GPU-hours for the whole project, tracked in a ledger. Ask before exceeding it. |
| LLM use | Same policy as Agents 1 and 2: deterministic, rule-based planner and templated reports as the reproducible core. An LLM layer is optional and only runs with an approved API key. Never claim an LLM was used when it was not. |
| Cluster access | The user runs this prompt from an agent session on Sycamore. Confirm with `hostname` and `sinfo` in Phase 0. |

## 1. Authoritative paths, existing work, and storage boundaries

Project root (working directory):

```text
/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning
```

Semester root:

```text
/vast/som/yunligrp/users/ygao/BIOS774Agent
```

Python environment (the only place to install Python packages):

```text
/vast/som/yunligrp/users/ygao/conda/envs/agent3
```

R on Sycamore (optional; see Section 2.1): `module load r/4.6.0` (Sycamore has no `r/4.4.0`), with the user's R library under `~/R/x86_64-pc-linux-gnu-library/4.6`.

Previous projects (inspect read-only, never modify, never install into their environments):

```text
/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent1_DR
/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent2_Generative
```

### 1.1 What already exists in Agent3

Inspect it first and preserve all of it:

- `README.md`: the handoff record, with dataset facts, storage boundaries and environment notes. Keep it, and update it once the pipeline exists.
- `docs/PROGRESS.md`: the 2026-11-02 acquisition and environment entries, including the replacement of the originally downloaded Chromium X Multiome sample by the annotated granulocyte-sorted sample. Append to it; do not rewrite history.
- `environment.yml`, `environment.lock.txt`, `environment.conda-explicit.txt`: the initial inspection environment (Python 3.11, NumPy, SciPy, pandas, h5py).
- `scripts/download_multiome.sh`, `scripts/inspect_multiome.py`: acquisition tools for the current PBMC sample that **write into the shared data directory**. Do not rerun them. Any new inspection writes inside Agent3.
- `logs/`: acquisition, environment and matrix-inspection logs.

No modeling pipeline, objective, preprocessing, split, or model choice exists yet. Agent3 is **not** a Git repository; initialize a local repository in the project root (Section 22).

### 1.2 Shared input data (read-only)

**Storage boundary:** both datasets are read-only inputs. Every Agent3 output, intermediate file, tile index, feature file, embedding, cache, checkpoint, log, split, and report stays under the Agent3 project root. Never write beside the shared inputs.

**Dataset A: 10x Multiome PBMC (RNA + ATAC).** Location:

```text
/vast/som/yunligrp/users/ygao/BIOS774Agent/data/pbmc_granulocyte_sorted_10k
```

- Sample: 10x "PBMC from a Healthy Donor, Granulocytes Removed Through Cell Sorting (10k)", Cell Ranger ARC 1.0.0, CC BY 4.0. This is the sample used by the Seurat WNN, Signac and SeuratData `pbmcMultiome` tutorials. The dataset's `README.md` documents provenance.
- Use `pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5` for called cells: **11,909 cells, 36,601 Gene Expression features, 108,377 Peaks**, GRCh38. Split rows by `features/feature_type`. Both modalities share barcode columns, so pairing is exact at the cell level. Structural checks passed (`download_metadata/matrix_inspection.json`).
- **Cell-type labels:** `annotations/pbmc_10x_annotation_FixFlip.csv` (column `celltype`; the first column holds barcodes, identical to and in the same order as the filtered matrix). It has 19 types for 10,412 cells, plus 1,497 cells labeled `filtered`. Read `annotations/LABEL_PROVENANCE.json` before using it. **Do not use the `seurat_annotations` field inside the SeuratData objects**: it has Naive B and Memory B swapped, as verified with canonical B-cell markers.
- Also available: raw HDF5 (736,320 barcodes including background), `atac_fragments.tsv.gz` + `.tbi`, `atac_peaks.bed`, `atac_peak_annotation.tsv` (peak-to-gene links), `per_barcode_metrics.csv` (GEX/ATAC QC per barcode, including `atac_fragments`, `atac_TSS_fragments`, `atac_peak_region_fragments`, `atac_mitochondrial_reads`), `summary.csv`, `web_summary.html`, and the vendor `analysis.tar.gz` (Cell Ranger ARC PCA/LSA/UMAP, k-means and graph clustering).
- Vendor analyses are 10x outputs, not Agent3 results. They may appear only as an external reference and never as ground truth.
- The data come from a single donor (healthy female, age 25). Cell-level evaluation does not establish donor generalization; say so.

**Dataset B: TCGA-BRCA (diagnostic H&E WSI + clinical variables).** Location:

```text
/vast/som/yunligrp/users/ygao/imaging_fi/data/TCGA-BRCA
```

**This entire directory belongs to another active project. Never modify, move, clean up, overwrite, or create files inside it, including `processed/`, `annotations/`, and `provenance/`.** Read from it directly.

- `raw/wsi/<file UUID>/<filename>.svs`: **1,133 diagnostic slides, 1,062 patients, about 1.16 TB**. Manifest: `provenance/manifests/wsi/TCGA-BRCA.diagnostic.all.tsv` (columns `id, filename, md5, size, state`). The manifest includes one metastatic slide. Patients can have several slides.
- `raw/clinical/TCGA-BRCA.with_followups.json` (1,098 cases) and `raw/clinical/TCGA-BRCA.json`: GDC clinical records (demographics, diagnoses, stage, follow-up). Link to slides by the TCGA patient barcode (`TCGA-XX-YYYY`).
- `annotations/pam50/BRCA_clinicalMatrix.tsv` (UCSC Xena, with `provenance.json`): sample-level table including `PAM50Call_RNAseq`, `PAM50_mRNA_nature2012`, `breast_carcinoma_estrogen_receptor_status`, `breast_carcinoma_progesterone_receptor_status`, `lab_proc_her2_neu_immunohistochemistry_receptor_status`, `HER2_Final_Status_nature2012`, `histological_type`, `age_at_initial_pathologic_diagnosis`, AJCC stage fields, and `_PATIENT`. It mixes primary tumor, normal and metastatic samples. Restricting to sample code `01` gives an unambiguous PAM50 match for **826 slide patients**. This is a candidate overlap, not a finalized cohort.
- `processed/clinical/phenotype.parquet`, `clinical_lineage.json`, `clinical_revision_audit.json`: the other project's derived clinical table and audit. Its audit reports, for 1,098 clinical cases: PAM50 LumA 421 / LumB 192 / Basal 140 / Her2 67 / Normal 23; 149 deaths among 1,075 OS-eligible cases; primary-diagnosis selection rules; and missingness. You may read these to cross-check, but Agent3 builds its own clinical table from raw sources, with its own lineage.
- `raw/omics/`, `processed/omics/`, `processed/cohorts/` (including `draft_splits.json` and `draft_seed_*.tsv`) are the other project's work. **Do not adopt their cohorts or splits.**

### 1.3 Pretrained model store (read-only)

```text
/vast/som/yunligrp/users/ygao/models/
├── UNI2-h/       # MahmoodLab/UNI2-h    rev d517a8dd…  vision-only, ViT-H/14 (timm kwargs in README)   CC-BY-NC-ND-4.0
├── Virchow2/     # paige-ai/Virchow2    rev 3158645…   vision-only, ViT-H/14 (timm; safetensors + bin) CC-BY-NC-ND-4.0
├── CONCH-Z/      # MahmoodLab/CONCH-Z   rev 60211fc…   CONCH v1.5 vision encoder + projection
│                 #                                      + text tower aligned on ~335k pathology reports;
│                 #                                      transformers remote code; 448 px input; 768-d    CC-BY-NC-ND-4.0
└── MUSK/         # xiangjx/musk         rev de1ffed…   image–text (model.safetensors, tokenizer.spm)  CC-BY-NC-ND-4.0
```

These four are the only pathology foundation models in scope for Agent3. Ignore any other folder in the store (for example `H-Optimus-1/`).

The user maintains this shared store for this and later projects. **Never write, move, or delete anything in it.** Each populated folder holds a full mirror of the Hugging Face repo at a pinned revision, plus `DOWNLOAD_MANIFEST.json` (repo ID, revision, license, per-file SHA256 verified against the HF LFS hashes at download). Before use, verify the SHA256 values against that manifest and record the revision in every run record. Load weights from local paths with `HF_HUB_OFFLINE=1` (for timm, a `pretrained_cfg_overlay` pointing to the local file, plus the architecture kwargs from the model's README). Do not download the same weights into the project cache. Prefer `model.safetensors` where both formats exist. All of these licenses are non-commercial (CC-BY-NC-ND-4.0), and their terms forbid redistributing the weights. Record the licenses in the reports, and never place FM weights or fine-tuned derivatives (including LoRA adapters) in Git or in the submission archives.

Full 40-character revisions are in each folder's `DOWNLOAD_MANIFEST.json`. On 2026-11-02, every file in all four populated folders was verified: LFS files against their SHA256 hashes, and small non-LFS files (including CONCH-Z's remote-code `.py` files) against their Git blob hashes. The separate CONCH v1.5 (vision-only) and original CONCH v1 downloads were deliberately removed by the user; CONCH-Z is the only CONCH model to use.

**Image–text capability:** the image–text models are CONCH-Z and MUSK. Use CONCH-Z for CONCH v1.5 patch features. Its unprojected features (`encode_image_features`) are CONCH v1.5 features. Image–text comparisons must use the projected space (`encode_image` / `project_image_features`). CONCH-Z's text tower was aligned to pathology reports, which makes it the most natural candidate for encoding templated clinical text in the `pretrained_multimodal_finetune` arm.

**Remote code:** CONCH-Z loads through `transformers` with `trust_remote_code=True`, which runs the Python files shipped in `CONCH-Z/` (`modeling_conch_z.py`, `configuration_conch_z.py`, `vision_tower.py`, `text_tower.py`, `conch_tokenizer.py`). Load it only from the local folder at the recorded revision. Read those files before the first execution and record their SHA256 hashes in the run records.

### 1.4 Reuse policy

Inspect Agent1 and Agent2 read-only and reuse proven architecture: typed configuration and schemas, a CLI with stage commands, stage manifests and validated resume, registries, worker isolation, Slurm logging and dependency chains, smoke gates tied to the code hash, evidence bundles, decision logs, Jinja2 → HTML → WeasyPrint report rendering, PDF verification, validators, and the `docs/PROGRESS.md` style. Do not copy dataset-specific code, stale paths, or the `/proj/` legacy prefix. Reuse architecture, not accidental technical debt.

## 2. Compute environment and cluster policy

### 2.1 Software environments

**Python (the orchestration language):**

- Install every new Python dependency into `agent3` only. Do not install Python packages into Agent1's `agent` or Agent2's `agent2` environments, the base Conda, or `~/.local` (no `pip install --user`). Never delete or recreate `agent3` without explicit approval.
- **User-site leakage was observed.** `~/.local/lib/python3.11/site-packages` contains TensorFlow 2.19 and h5py 3.14, which appear in `pip list` whenever `agent3`'s Python runs without activation. `PYTHONNOUSERSITE=1` is set in `conda-meta/state`, but it only takes effect on activation. Every script, Makefile target and Slurm job must export `PYTHONNOUSERSITE=1` explicitly. The environment check must fail if any `~/.local` path is on `sys.path`. Do not delete the user's `~/.local` packages.
- Redirect every cache into the project: `HF_HOME`, `TORCH_HOME`, `MPLCONFIGDIR`, `NUMBA_CACHE_DIR`, pip's cache, and `XDG_CACHE_HOME` (for example under `.cache/`). Downloading weights must not depend on the home quota.
- Expected real dependencies, installed when needed: CUDA PyTorch built for H100 (sm_90), torchvision, timm, huggingface_hub, safetensors, scanpy, anndata, muon/mudata, scvi-tools, mofapy2 (optional), leidenalg, python-igraph, umap-learn, scikit-learn, scrublet or an equivalent doublet tool, pyarrow, openslide-python with openslide-bin (or tiffslide), tifffile, opencv-python-headless or scikit-image, lifelines or scikit-survival, pydantic ≥ 2, pyyaml, matplotlib, jinja2, weasyprint, pypdf, psutil, pytest, ruff. Install the model code that the local foundation models need from their official sources: the `musk` package (lilab-stanford/MUSK) for MUSK; `transformers` for CONCH-Z's bundled remote code (see Section 1.3); a timm version that supports the UNI2-h and Virchow2 architectures (`SwiGLUPacked`, `SiLU`, register tokens, and the kwargs given in each README). Every FM must load with `strict=True` state-dict matching, and its output dimension must be checked. Record the exact source commits. If a required package is missing, install the real package. Do not substitute a weaker approximation and keep the original name.
- WeasyPrint needs fonts and Fontconfig. Agent1 needed explicit `FONTCONFIG_FILE`/`FONTCONFIG_PATH` and `LD_LIBRARY_PATH` settings (see `transfer_readme.md` in the semester root). Apply the same pattern to `agent3` only, and verify by rendering and extracting text from a one-page PDF.
- After every install, export `environment.yml`, `environment.lock.txt` and `environment.conda-explicit.txt`, and run `pip check`.

**R and command-line tools (allowed when they are the right tool):**

- R is available on Sycamore through `module load r/4.6.0`. The user is reinstalling their Longleaf R library there by following `prompt/install.md`, into the R 4.6 user library `~/R/x86_64-pc-linux-gnu-library/4.6`. At Phase 1, check what is actually installed rather than assuming the install has finished.
- Use R where an R implementation is the established reference or the only implementation. Examples: Seurat/Signac WNN as a cross-check of the Python WNN baseline, reading `.rds`/`.rda` inputs, or an R-only statistical method. Python remains the orchestrator. R steps run as `Rscript` stages called by the pipeline, with these rules:
  - inputs and outputs are files under Agent3;
  - seeds are fixed;
  - the command line is logged;
  - `sessionInfo()` is captured per run;
  - heavy R work runs in Slurm like everything else.
  
  The registry naming rules apply unchanged: an R adapter must implement the method it is named after.
- If a needed R package is missing, install it into the **home R user library** above. Use `BiocManager::install()` for CRAN and Bioconductor packages, and `remotes::install_github()` pinned to a commit SHA for GitHub packages. Command-line tools a method genuinely needs may be installed under the user's Sycamore home directory (for example `~/tools/<name>/`, exposed through `~/bin` or an explicit path in the Agent3 config).
- Rules for these home-directory installs, which the user's other projects share:
  - never install into the R module's site library;
  - never remove, downgrade or force-upgrade existing packages without asking (`update = FALSE`, `upgrade = "never"`);
  - check the home quota before large installs;
  - do not edit `~/.Renviron`, `~/.Rprofile` or shell startup files unless the change is required and recorded.
- Record every R package or tool install in `docs/PROGRESS.md` (name, version, source, commit, install path). Export an R lock file into Agent3 (for example `environment.R.lock.csv` from `installed.packages()`, plus `sessionInfo()` output) whenever R is used.

### 2.2 Cluster

- Heavy work runs through Slurm. Use login nodes only for inspection, installs, small unit tests, job submission and status checks.
- GPU work uses **Sycamore H100** with account `h100_harlin` and an explicit partition: `h100_short` (up to 2 h), `h100_priority` (up to 5 days), or `h100_preempt` (up to 5 days, preemptible). Recheck partitions, node CPU/memory/GPU counts, and current availability (`sinfo`, `scontrol show partition`, `sacctmgr`) before writing resource requests. Jobs on `h100_preempt` must checkpoint and resume safely.
- The user runs this agent session on Sycamore, and `/vast` is shared with Longleaf. Confirm with `hostname` and `sinfo` in Phase 0. Do not move the GPU workload to Longleaf queues without approval.
- CPU-only stages (PBMC preprocessing, small models, metrics, reports) may use CPU resources on the same cluster. Choose them after inspecting the partitions; do not inherit obsolete assumptions.
- Maintain a GPU-hour ledger (`logs/gpu_ledger.json`) with job ID, partition, GPUs, elapsed time, and purpose, reconciled against `sacct`. Stop and ask before exceeding the cap in Section 0.
- Assume compute nodes may lack internet access. Download model weights and annotation models on a login node into the project cache, record their revision/hash, and run jobs with `HF_HUB_OFFLINE=1`.
- The TCGA slides sit on storage used by another project. Limit concurrent slide readers to a declared number, read only the pyramid levels and regions you need, and never copy the full 1.16 TB.

### 2.3 Other policies

- Do not push to GitHub, upload to Canvas, or publish anything unless the user explicitly asks. Make the repository Git-ready.
- Never commit raw data, tile images, feature arrays, checkpoints, caches, credentials or API keys.
- Never send raw data, slide images, or record-level clinical data to an external API.
- Ask the user only when blocked by credentials, gated model access, cluster access, an ambiguous official requirement, a destructive change, or a budget overrun. Routine installation and debugging inside the paths above are authorized.

## 3. Project objective

Build an **objective-aware, evidence-driven multimodal representation learning agent** that accepts:

1. a multimodal dataset, described by a configuration that names its modalities, files, and pairing keys; and
2. a user objective, given as structured YAML/JSON or plain language.

The agent must automatically:

1. inspect each modality and the cross-modal pairing structure;
2. interpret the user objective;
3. recommend and perform modality-specific preprocessing, fitted only within the correct scope;
4. decide whether and how the modalities should be integrated, choosing the fusion strategy, encoders and training objectives;
5. plan a fair, resource-aware pilot of eligible strategies;
6. train or fine-tune the eligible candidates;
7. evaluate the learned representations on structure, alignment, downstream utility, modality contribution and stability;
8. select and refine a small final set on validation evidence, freeze it, then evaluate on held-out data;
9. run downstream analyses (classification, clustering, retrieval, cross-modal prediction, survival as appropriate);
10. issue an evidence-based **integration verdict**, in which "a single modality is sufficient" is a legitimate outcome;
11. export the learned representations with full provenance and generate validated reports.

The central strategy is:

> **Inspect modalities and pairing + interpret objective → preprocess per modality → decide whether integration is warranted → identify eligible strategy/encoder/objective combinations → establish single-modality and simple-fusion baselines → micro-pilot eligible candidates → evaluate structure, alignment, utility, modality contribution and stability → shortlist → tune → freeze → held-out evaluation → downstream analysis → integration verdict → report.**

This must not be two hand-written pipelines behind one command. The data profile and the objective must **materially change** preprocessing, eligible strategies, encoders, losses, metrics and downstream analyses. Planner rules branch on profile properties (modality type, pairing level, sample size, sparsity, label availability, missingness), **never on the dataset name**.

## 4. Lessons carried over from Agent 1 and Agent 2

Agent 1 audit findings to avoid:

- Sub-second runtime noise decided a shortlist. Runtime is a tiebreaker or constraint only, measured on fit time, never on import time.
- Seed stability was not comparable across methods. Define stability so deterministic and stochastic methods are scored on the same basis.
- Failures went undetected: non-convergence, disconnected graphs, collapsed layouts. Add explicit collapse and validity diagnostics.
- Preprocessing branched on the dataset name. Preprocessing must branch on the data profile.
- Only local metrics were used. Include global structure and downstream metrics.
- The report contained a draft notice, and example figures were chosen alphabetically. Choose figures by a declared rule, and never ship draft text.

Agent 2 practices to keep, plus the corrections from its report revision:

- A smoke gate that is written automatically by a passing smoke job and tied to the current analysis-code hash. A hand-written success flag is never valid.
- Separate fingerprints for analysis code and report code, so report-only edits do not invalidate analysis stages.
- Pre-registering the protocol and selection rules before any real pilot, committed locally.
- A one-command Slurm launcher with `afterok` dependencies; failed jobs block their dependents.
- An unseen-dataset regression test showing the planner generalizes beyond the two demonstrations.
- Reports must **lead with the objective, the decisions taken, what was executed, the results and the interpretation**. No raw metric dictionaries in tables. Include an architecture figure and a claim-to-source map. Visually inspect every rendered page.

## 5. Demonstration datasets and default objectives

The two reports must show the same agent choosing different preprocessing, strategies, encoders, objectives, metrics and downstream analyses because the data and objectives differ.

### 5.1 Dataset 1: PBMC Multiome (RNA + ATAC), `generated_report_1`

Default objective (`objectives/pbmc_multiome_joint.yaml`):

```text
Learn a joint RNA+ATAC cell representation that preserves PBMC cell-type structure
at least as well as the better single modality, aligns the two modalities so that a
cell's RNA profile retrieves its own ATAC profile, and supports cross-modal prediction
of chromatin accessibility from expression. Cell-type labels may be used for
evaluation only, not for fitting the primary representation.
```

This demonstration should exercise:

- two high-dimensional, sparse count modalities with exact cell-level pairing;
- per-modality QC: RNA library size, detected genes, mitochondrial fraction; ATAC fragments, TSS enrichment (from `atac_TSS_fragments` / fragments, or computed from the fragment file), fraction of fragments in peaks, nucleosome signal if computed; and a documented doublet decision. Prefer data-driven thresholds (for example MAD-based) recorded with their values;
- RNA preprocessing: raw counts kept for count likelihoods; library-size normalization and log1p for Gaussian/MLP encoders; highly variable genes chosen on training cells only;
- ATAC preprocessing: binarization or count handling declared per model; peak filtering by training-cell frequency; TF-IDF with IDF weights fitted on training cells; LSI fitted on training cells, with the depth-correlated first component diagnosed rather than dropped by habit; optional gene-activity scores from `atac_peak_annotation.tsv` for gene-level cross-modal analyses;
- strategies such as WNN, MultiVI, CLIP-style dual encoders and MOFA+ (see Section 9), compared with RNA-only PCA, ATAC-only LSI and concatenation baselines;
- evaluation of biological structure, cross-modal alignment (FOSCTTM, retrieval recall@k), cross-modal prediction, neighborhood preservation of each modality, and QC/depth confounding.

**Curated reference labels.** Use the Satija lab's annotations (`celltype` in `annotations/pbmc_10x_annotation_FixFlip.csv`; see Section 1.2). How to treat them:

- They are **curated expert labels, not measured ground truth.** They come from manual annotation of Seurat WNN clusters of these same cells, named using marker genes. A WNN representation therefore has a structural advantage when scored against them, and RNA-informative representations may be favored. The report must state this, and must complement label-based metrics with label-free ones (alignment, retrieval, cross-modal prediction, neighborhood preservation).
- The primary evaluation uses the 19 fine labels. A secondary coarse level (for example T / NK / B / Plasma / Monocyte / DC / HSPC) comes from a declared, versioned mapping. Report per-class results; rare classes (HSPC 26, Plasma 18) make macro metrics noisy.
- Cells labeled `filtered` failed the vignette's QC (nCount_ATAC 5e3–7e4, nCount_RNA 1e3–2.5e4, percent.mt < 20). The agent still performs and documents its own data-driven QC. Label-based metrics use only cells that both pass the agent's QC and carry a real label, and the report shows how the agent's QC compares with the vignette's.
- Labels never enter preprocessing or representation fitting, except in a run explicitly marked label-informed (`reference_only`). The split may be stratified by label, for balance only.
- Never present the vendor clusters in `analysis.tar.gz` as truth.

### 5.2 Dataset 2: TCGA-BRCA (H&E WSI + clinical variables), `generated_report_2`

Default objective (`objectives/tcga_brca_pam50.yaml`):

```text
Learn a patient-level joint representation of diagnostic H&E whole-slide images and
baseline clinical variables that predicts PAM50 intrinsic subtype on held-out patients,
quantify whether histology adds information beyond clinical variables (and vice versa),
and assess secondary utility for overall-survival risk stratification and
subtype-consistent patient retrieval. Training labels may be used.
```

This demonstration should exercise:

- a gigapixel image modality paired with a low-dimensional, partially missing tabular modality at the **patient level**, with several slides for some patients;
- WSI preprocessing: read slide metadata (objective power, MPP, levels) per slide; tissue segmentation on a thumbnail (for example Otsu on saturation plus morphology), with background, pen-mark and blur filtering where feasible; tiling at a declared physical resolution (default about 0.5 µm/px, 224 px, or the encoder's native input) using the best pyramid level plus resizing; a seeded per-slide tile cap chosen from a measured throughput pilot and the GPU budget. Stain normalization is a recorded decision, not a default;
- store tile **coordinates and features**, not tile images, except a bounded, documented cache needed for an approved fine-tuning arm, and a small QC/attention gallery;
- patch encoders: a pathology foundation model (frozen features), an ImageNet backbone baseline, and optionally parameter-efficient fine-tuning (Section 9);
- slide-to-patient aggregation by mean pooling (baseline) or attention-based MIL (ABMIL or gated attention), with attention weights saved;
- clinical encoding: categorical one-hot or embeddings, standardized numerics, and explicit missingness indicators. Imputers and scalers are fitted on training patients only;
- fusion: early (concatenation), late (decision-level), and intermediate (attention or gated) fusion, image–tabular contrastive pretraining, and modality dropout for missing-modality robustness;
- evaluation: held-out PAM50 macro-F1, balanced accuracy, per-class recall and one-vs-rest AUROC with bootstrap CIs; modality ablations; site confounding; exploratory survival C-index; retrieval; attention heatmaps.

**Cohort and leakage rules:**

- Unit of analysis: patient. Use only primary-tumor diagnostic slides (sample type `01`). Exclude the metastatic slide and any non-primary slides, and record the exclusions. Aggregate all of a patient's eligible slides.
- Build Agent3's clinical table from `raw/clinical/*.json` and the Xena matrix, linked by patient barcode and sample code `01`. Document parsing, category harmonization (Positive/Negative/Equivocal/Indeterminate/Not evaluated → declared levels), conflict resolution and missingness. Cross-check counts against the other project's audit and report any disagreement; do not copy its tables.
- **Inputs versus targets:** never use as inputs the target, anything derived from molecular assays (`PAM50*`, `Integrated_Clusters*`, other RNA-based calls), follow-up or outcome variables (vital status, days to death or last follow-up), or post-diagnosis treatment variables. For the survival analysis, outcome fields are targets only.
- **Splits:** define patient-level train/validation/test splits for **all** eligible patients (labeled and unlabeled) before any modeling, so that test patients never enter any training, including self-supervised or contrastive pretraining. Stratify by the target where it is available. Keep every slide of a patient in one split.
- **Site confounding:** TCGA histology features are known to encode the tissue source site (`TCGA-<TSS>-...`). Record the TSS for every patient, report how well each representation predicts the site (balanced accuracy or AUROC with a chance reference), and run a site-held-out evaluation of the final selected representation as a sensitivity analysis.
- Do not assume complete covariates or labels. Report the cohort flow: slides → eligible slides → patients → patients with PAM50 → patients with complete clinical inputs.
- About 800 labeled patients give wide confidence intervals for a single held-out split. Use bootstrap CIs and repeated seeds, and do not overstate small differences.

## 6. Objective interpreter

Create a validated `ObjectiveSpec` schema containing at least:

```text
objective_id
original_text
objective_types              # e.g. cell_state_structure, cross_modal_alignment,
                             #      cross_modal_prediction, supervised_prediction,
                             #      survival_risk, patient_similarity
target_variable
label_source_and_quality     # ground truth | curated | silver | none, plus caveats
allowed_supervision          # none | evaluation_only | train_labels
modalities_required
unit_of_analysis             # cell | slide | patient
downstream_tasks
primary_metrics
secondary_metrics
inapplicable_metrics
clinical_inputs_and_exclusions   # where applicable
sensitivity_analyses
compute_budget
prohibited_uses
parser_mode                  # structured | deterministic_text | llm
parser_confidence
ambiguities
clarification_needed
```

Support structured YAML/JSON objectives and plain-language objectives. The plain-language parser must have a deterministic fallback for the supported objective categories. An LLM parser is optional and runs only with an approved key. Validate every parse. When ambiguity would materially change modeling (for example, whether labels may train the representation), return a clarification request instead of guessing.

The objective must visibly drive planning. For example, `allowed_supervision: evaluation_only` makes the supervised family ineligible for the primary representation; it may run only as a clearly labeled "label-informed reference". `allowed_supervision: train_labels` makes it eligible.

## 7. Dataset inspector and data profile

Create a `DataProfile` schema with a `ModalityProfile` per modality and a `PairingProfile` across modalities:

```text
ModalityProfile:
  modality_id, modality_type   # count_matrix_gene | count_matrix_peak | wsi_image |
                               # tabular_clinical | dense_image | continuous_tabular
  n_samples, n_features, feature_types
  storage_format, shape_and_channels, value_range, dtype
  sparsity, library_size_distribution, binary_or_count_indicators
  missingness_by_feature       # distinct from biological zeros
  wsi: n_slides, objective_power, mpp, levels, estimated_tissue_area, estimated_tiles
  qc_metrics_available
  estimated_memory, estimated_preprocessing_cost

PairingProfile:
  pairing_key, pairing_level   # cell barcode | patient barcode
  n_paired, n_unpaired_by_modality, multiplicity (e.g. slides per patient)
  label_availability_by_split_candidate
  group_or_batch_variables     # donor, tissue source site, slide, patient
  missing_modality_rates
  licensing_and_release_constraints
```

The inspector must distinguish at least count matrices, binary-like accessibility matrices, gigapixel images, small images, and tabular data with mixed types. Biological zeros are not missingness. All inspection outputs are written inside Agent3.

Every preprocessing action is recorded with:

```text
action, modality_id, parameters, reason, fit_scope, data_leakage_check, rows_and_features_before_after
```

## 8. Preprocessing planner

The planner maps modality profiles to preprocessing tools, for example:

- count gene matrix → QC, doublet decision, train-fitted HVG selection, normalization/log for Gaussian encoders, raw counts kept for count likelihoods;
- peak matrix → QC, train-fitted peak filtering, binarization or count handling, train-fitted TF-IDF/LSI, optional gene activity;
- WSI → slide metadata QC, tissue segmentation, resolution-matched tiling, tile QC, feature extraction with a declared encoder, a resumable per-slide feature store (HDF5 or zarr with coordinates, level, MPP, encoder ID and hash);
- tabular clinical → type inference, category harmonization, train-fitted imputation and scaling, missingness indicators, leakage-exclusion rules from the `ObjectiveSpec`.

Stages exist because the profile requires them. A WSI modality inserts `tile` and `extract-features` stages; a count modality does not. Each tool has its own unit tests on synthetic data.

## 9. Strategy, encoder, and objective registry

The registry declares genuine, smoke-tested adapters for each method family covered in class. "Capable of" means a callable adapter, declared capabilities and constraints, input validation, genuine training, embedding export (`encode`) for new samples, checkpoint save/load, at least one synthetic smoke test, and an honest failure mode.

| Family | Expected real implementations | Training objective | Typical suitability |
|---|---|---|---|
| `baseline_unimodal` | RNA PCA; ATAC TF-IDF+LSI; clinical one-hot/standardized; histology mean-pooled frozen features | none / linear | Required reference for every dataset |
| `baseline_fusion` | concatenated (block-scaled) PCA/LSI; WNN (muon); early-fusion concatenation for histology+clinical | none / linear | Required simple integration reference |
| `self_supervised_generative` | MultiVI (scvi-tools; count/accessibility likelihoods; handles missing modalities); MOFA+ (mofapy2) as a linear factor model | ELBO / factor model | Paired count modalities |
| `contrastive_unimodal` | SimCLR-style NT-Xent with declared augmentations (count thinning, feature dropout, tile-bag subsampling) | NT-Xent | Unlabeled single-modality pretraining before fusion |
| `clip_cross_modal` | Dual encoders with projection heads, symmetric InfoNCE, learnable temperature; optional reconstruction term declared explicitly | symmetric InfoNCE | Paired samples with shared information across modalities |
| `supervised_fusion` | Early, late and intermediate (attention/gated) fusion networks with MIL aggregation for WSI; modality dropout | cross-entropy (classification); Cox partial likelihood (survival, if selected) | Labels allowed for training |
| `pretrained_foundation` | Frozen pathology FM tile encoder (UNI2-h, Virchow2, CONCH-Z's image tower, or MUSK's image tower) with a linear/MIL probe; parameter-efficient fine-tuning (LoRA or last-block unfreezing) of the tile encoder on a bounded tile subset; optionally a single-cell FM for RNA if accessible | task loss for fine-tuning | Gigapixel histology; RNA only when a compatible model exists |
| `pretrained_multimodal_finetune` | Fine-tuning a pretrained image–text pathology model (CONCH-Z, whose text tower was aligned on pathology reports; MUSK as the alternative). Clinical variables are rendered into text by a deterministic, versioned template that applies the same input exclusions, and the image and text towers are aligned at patient level with a contrastive loss (LoRA or partial unfreezing; bounded tile subset) | symmetric InfoNCE, optionally plus a task head | Histology paired with clinical variables when an image–text model is available. Compare against the tabular-encoder CLIP arm, because the templated text adds no information beyond the table. |

Do not relabel one method as another:

- concatenated PCA or WNN is not contrastive or deep learning;
- a frozen encoder with a probe is **not fine-tuning**; call it "frozen pretrained encoder";
- a deterministic autoencoder is not MultiVI or a VAE;
- an ImageNet backbone is not a pathology foundation model;
- CLIP-style alignment trained with labels is supervised, not self-supervised;
- a model trained on any cell-type labels is "label-informed", not unsupervised.

### 9.1 Applicability planner

For each dataset/objective pair, assign every registered family (and each major encoder option) one status:

```text
eligible_for_pilot
reference_only          # e.g. supervised arm when labels are evaluation-only
not_applicable
infeasible_under_budget
failed_validation
```

Each non-eligible status carries a concrete reason tied to the profile, the objective, or the measured budget. The rules are encoded as testable functions with rule IDs, for example:

- both modalities high-dimensional and paired at the sample level with enough pairs → `clip_cross_modal` eligible;
- count-valued modalities → count-likelihood generative models eligible; never feed log-normalized values into a model while claiming a count likelihood;
- labels with `allowed_supervision: train_labels` → `supervised_fusion` eligible, otherwise `reference_only`;
- a gigapixel image modality without dense labels → tile encoder + MIL required; end-to-end whole-slide training `infeasible_under_budget`;
- a low-dimensional tabular modality with few hundred pairs → cross-modal contrastive pretraining eligible but flagged "small-pair regime", with supervised and late fusion favored a priori; the pilot decides;
- missing-modality rate above a declared threshold → only strategies that tolerate missing modalities (MultiVI, modality dropout) for the affected samples;
- pathology FM weights inaccessible → `failed_validation` with the access reason, and continue with accessible encoders.

Pathology foundation models are taken from the local store (Section 1.3): UNI2-h and Virchow2 (vision-only), plus CONCH-Z and MUSK (image–text). Do not download them yourself, and never read or embed credentials. For each model, record the model ID, local revision, license, embedding dimension, expected input resolution and normalization, and whether it is vision-only or image–text. Running every FM on every slide is not required. The planner picks a small set of encoders to pilot (for example, one strong vision-only FM, one image–text FM, and the ImageNet baseline) using declared criteria: availability, input-resolution match, measured throughput and GPU budget. It records why the others were not piloted.

### 9.2 Adapter contract

```python
class RepresentationAdapter:
    family: str
    model_id: str

    def capabilities(self): ...
    def assess_applicability(self, data_profile, objective_spec, budget): ...
    def estimate_cost(self, data_profile, params): ...
    def validate_input(self, train_data, val_data): ...
    def fit(self, train_data, val_data, params, seed): ...
    def encode(self, data, modality=None): ...       # joint and, where defined, per-modality embeddings
    def predict_cross_modal(self, data, source, target): ...
    def predict(self, data): ...                      # supervised heads only
    def attention(self, data): ...                    # MIL models only
    def save(self, path): ...
    def load(self, path): ...
```

Unsupported operations raise an explicit capability error; they never return fabricated or empty results.

Every run result records:

```text
dataset_id, objective_id, family, model_id, encoders, fusion_type, training_objective,
status, eligibility_reason, hyperparameters, embedding_dim, seed, parameter_count,
trainable_parameter_count, epochs_or_steps_completed, early_stopping_reason,
training_curves, checkpoint_path, embedding_paths, runtime_seconds (fit only),
peak_memory_mb, peak_gpu_memory_mb, hardware, slurm_job_id, warnings, error_type,
error_message, retry_count, package_versions, git_commit, config_hash, analysis_code_hash
```

## 10. Split discipline and leakage control

- Create deterministic, manifested train/validation/test splits before fitting anything. PBMC: cell-level (default 70/15/15), optionally stratified by the curated label for balance only. TCGA: patient-level as in Section 5.2.
- Fit every data-dependent transform (QC thresholds where they are model-relevant, HVG and peak selection, TF-IDF/IDF, LSI/PCA, scalers, imputers, category vocabularies) on training samples only, then apply it to validation and test samples.
- Representations are trained on training samples only. Embeddings for validation and test samples come from `encode`, not from refitting.
- Use validation data for early stopping, selection and tuning. The test split stays untouched until the final plan is frozen and hashed.
- Downstream probes are fitted on training embeddings, tuned on validation, and scored once on test.
- Retrieval and FOSCTTM are computed within a declared candidate pool (for example all test cells); always report the pool size.
- The validator checks split disjointness, fit scopes, and that no test identifier appears in any training artifact.

## 11. Baselines first, then a comparable micro-pilot

### 11.1 Baselines

Run these before any learned multimodal model, and tune them as carefully as the learned models:

- PBMC: RNA-only PCA, ATAC-only LSI, concatenated PCA+LSI, and WNN.
- TCGA: clinical-only (logistic regression or a small MLP), histology-only (mean-pooled frozen FM features with a linear probe; ABMIL as the stronger histology-only reference), and early-fusion concatenation.

### 11.2 Common evaluation probe

Every representation, baselines included, is scored with the same frozen-embedding probe protocol: kNN classifier and logistic regression (C tuned on validation), Leiden clustering, and the same alignment and retrieval code. End-to-end supervised predictions are reported separately from probe results. When embedding dimensions differ, report the dimensions and, where practical, fix a common dimension for the learned models.

### 11.3 Micro-pilot

1. Build a representative pilot subset from the training split only (or use all training data when it is cheap, as for PBMC).
2. Declare a common budget policy: maximum wall time, family-appropriate maximum epochs or steps, early stopping, identical data access, and recorded hardware and parameter counts. Equal epochs are not automatically fair across unlike objectives.
3. Train each eligible candidate with one heuristic configuration and a few seeds.
4. Measure throughput for TCGA (tiles/s for reading and for encoding) on a pilot of about 20 slides, then fix the final tile cap from the GPU budget. Record the calculation.
5. Isolate failures and timeouts without stopping the pipeline.

### 11.4 Failure recovery

Record the original configuration and error. Allow at most one predefined, scientifically valid retry (smaller batch after a measured OOM, gradient clipping, lower learning rate after divergence, a compatible package version). Record the change and its rationale. If the retry fails, keep the terminal status and continue. Never substitute another family under the failed model's name. Any capacity or data reduction for a single candidate must be documented because it changes comparability.

## 12. Selection, tuning, and freezing

Do not select by training loss, popularity, or an unexplained weighted average. Write the selection rules in `docs/METHOD_NOTES.md` and commit them before any real pilot. Use a structured evidence table with separate columns:

```text
biological_or_label_structure
cross_modal_alignment            # where defined
downstream_utility               # objective's primary metric
modality_contribution            # gain over best single modality
stability_across_seeds
confounding_diagnostics          # depth/QC (PBMC), site (TCGA)
validity_and_collapse_flags
runtime_and_memory
```

Default logic:

1. reject invalid, non-converged or collapsed representations (low effective rank, near-constant dimensions, degenerate neighbor graphs);
2. identify an ε-Pareto set over the objective's primary metrics, with pre-registered equivalence margins; within a margin, prefer the simpler or cheaper model;
3. advance at most two candidates per dataset to tuning unless evidence strongly justifies more, and keep the best baseline as a reference;
4. tune only shortlisted models with small, meaningful grids (learning rate, latent dimension, temperature, dropout, MIL attention size, fusion type, LoRA rank, tile cap). Keep the default unless the gain exceeds the margin;
5. freeze the final plan (configuration, seeds, checkpoints, hashes) before opening the test split;
6. permit the outcome "no learned multimodal model beats the simple baselines".

Runtime breaks ties only when quality differences fall within the margin, and it is measured on fit time, never imports.

## 13. Evaluation framework

### 13.1 Common layer, applicable to both datasets

- **Structure:** label ARI and NMI of Leiden clusters (with a declared resolution or a small resolution scan), label silhouette, kNN label accuracy and macro-F1 (train → test), and graph connectivity per label.
- **Global structure:** correlation of pairwise distances or a similar global measure against each modality's reference space.
- **Modality neighborhood preservation:** kNN overlap between the joint embedding and each single-modality reference space, to show what each modality contributes and what alignment discards.
- **Alignment (per-modality embeddings):** FOSCTTM, recall@1/5/10 in both directions, and the modality-mixing silhouette.
- **Downstream utility:** the objective's primary task metrics with bootstrap 95% CIs and paired differences against the baselines.
- **Modality contribution:** joint minus best single modality, by ablation (drop each modality at test time for models that allow it).
- **Validity and collapse:** effective rank, per-dimension variance, and the fraction of near-duplicate embeddings.
- **Confounding:** R² of QC or depth variables predicted from the embedding (PBMC), and site predictability (TCGA).
- **Stability:** repeated seeds for stochastic models, reported as the mean and SD of the key metrics, plus subsample stability for deterministic baselines, so both are comparable.
- **Compute:** fit time, peak memory, GPU-hours and parameter counts.

Never collapse these categories into one unexplained score. Label diagnostics honestly.

### 13.2 PBMC-specific evaluation

- All structure metrics against the curated labels, at both the fine (19-type) and coarse levels, with the caveat that the labels derive from WNN clustering (WNN's structural advantage is flagged wherever WNN is compared).
- Cross-modal retrieval and FOSCTTM on test cells for models with per-modality embeddings, plus kNN-based retrieval baselines in PCA/LSI space.
- Cross-modal prediction (default RNA → ATAC): predict binarized accessibility of held-out test cells for train-selected peaks, and report per-cell and per-peak AUROC/AUPRC against a kNN-in-RNA-PCA baseline and a peak-frequency baseline. Optionally run ATAC (gene activity) → RNA with correlation metrics. Use MultiVI's own cross-modal prediction where available.
- Depth and QC confounding: correlation of the embedding with log UMI count, log fragment count, TSS enrichment and mitochondrial fraction.
- Qualitative: UMAPs of the joint and single-modality embeddings colored by curated label, modality (for co-embeddings) and QC; marker-gene expression and gene-activity panels for sanity checks. UMAP is a visualization, not evidence.

### 13.3 TCGA-specific evaluation

- PAM50 on held-out test patients: macro-F1, balanced accuracy, per-class recall, worst-class recall, one-vs-rest AUROC, and a confusion matrix, with bootstrap CIs. Compare clinical-only, histology-only, early fusion, the selected fusion model, and the contrastive-pretrained representation with a linear probe.
- Primary versus sensitivity clinical input sets (with and without receptor status), and the specific question of whether histology improves the LumA-versus-LumB distinction beyond receptor status.
- Missing-modality robustness: performance with clinical inputs dropped or masked at test time, for models trained with modality dropout.
- Site confounding and a site-held-out sensitivity evaluation of the final representation.
- Exploratory survival: Cox model on embeddings (or a Cox head) fitted on training patients, with test C-index and bootstrap CI, and Kaplan–Meier curves for train-defined risk groups. State the small number of events.
- Retrieval: subtype precision@k for patient-to-patient retrieval, and image → clinical retrieval recall@k for the contrastive model.
- Qualitative: attention heatmaps and top-attended tiles for a declared selection of test slides (for example, the highest-confidence correct and incorrect prediction per class), plus UMAPs of patient embeddings by subtype and by site.

## 14. Integration verdict

For each dataset/objective, produce `verdict.json`:

```text
integration_verdict: integrate | integrate_with_caution | single_modality_sufficient
selected_representation
selected_strategy, encoders, fusion_type, training_objective
comparison_to_best_single_modality       # paired difference, CI
comparison_to_simple_fusion_baseline
structure_evidence, alignment_evidence, utility_evidence
modality_contribution_evidence
confounding_and_validity_flags
approved_downstream_uses
discouraged_or_prohibited_uses
uncertainty_and_limitations
conditions_for_reconsideration
```

The verdict is computed from rules linked to validated metrics and configurable thresholds, never from prose. A suggested rule:

- `integrate`: the selected joint representation beats both the best single modality and the simple-fusion baseline on the primary metric by more than the margin, with a paired bootstrap CI excluding zero, and no major validity or confounding flag;
- `integrate_with_caution`: there is a gain, but the CI includes zero, secondary metrics conflict, or a confounding flag is present;
- `single_modality_sufficient`: otherwise.

Distinguish project decision thresholds from established external standards. Prohibited uses should be concrete, for example: "agreement with WNN-derived labels is not independent evidence that a representation is biologically superior to WNN"; "single-donor results do not support donor-level generalization"; "site-confounded embeddings should not be used for cross-site claims".

## 15. Repository structure

Build incrementally around the existing files. Do not create empty complexity.

```text
Agent3_RepresentationLearning/
├── prompt/prompt.md
├── README.md
├── .gitignore
├── pyproject.toml
├── environment.yml
├── environment.lock.txt
├── environment.conda-explicit.txt
├── Makefile
├── config/
│   ├── base.yaml
│   ├── pbmc_multiome.yaml
│   └── tcga_brca.yaml
├── objectives/
│   ├── pbmc_multiome_joint.yaml
│   ├── tcga_brca_pam50.yaml
│   └── examples/                      # alternative/plain-language objectives for robustness demos
├── docs/
│   ├── PROGRESS.md
│   ├── ASSIGNMENT_REQUIREMENTS.md
│   ├── METHOD_NOTES.md                # pre-registered protocol and selection rules
│   ├── DATA_PROVENANCE.md
│   └── REQUIREMENTS_TRACEABILITY.md
├── scripts/
│   ├── download_multiome.sh           # existing; do not rerun
│   ├── inspect_multiome.py            # existing; do not rerun
│   ├── 00_preflight.sh
│   ├── 01_setup_environment.sh
│   ├── 02_fetch_models.sh             # weights/annotation models into the project cache (login node)
│   ├── 03_smoke_test.sh
│   ├── submit_workflow.sh             # one command → dependency-chained Slurm jobs
│   └── slurm/
│       ├── common.sh
│       ├── smoke.sbatch
│       ├── wsi_features.sbatch        # resumable job array over slide shards
│       ├── run.sbatch
│       └── finalize.sbatch
├── src/agent3_replearn/
│   ├── __init__.py, __main__.py, cli.py, orchestrator.py
│   ├── config.py, schemas.py, utils.py
│   ├── inspector.py, objective_parser.py, planner.py, policy.py
│   ├── splits.py, preprocessing/{counts.py, atac.py, wsi.py, clinical.py}
│   ├── data/{multiome.py, tcga_brca.py, synthetic.py}
│   ├── registry.py
│   ├── models/{base.py, baselines.py, wnn.py, multivi_adapter.py, mofa_adapter.py,
│   │           contrastive.py, clip_dual.py, mil.py, fusion.py, pathology_fm.py}
│   ├── pilot.py, selector.py, tuning.py, final.py, worker.py, execution.py
│   ├── evaluation/{structure.py, alignment.py, downstream.py, survival.py,
│   │               confounding.py, stability.py}
│   ├── verdict.py, reporting.py, project_report.py, validation.py
├── templates/
│   ├── generated_report.html.j2
│   ├── project_report.html.j2
│   └── report.css
├── tests/
├── .cache/                            # ignored
├── checkpoints/                       # ignored
├── outputs/
│   ├── pbmc_multiome/
│   └── tcga_brca/
└── reports/
    ├── generated_report_1.pdf
    ├── generated_report_2.pdf
    └── report.pdf
```

Use an installable `src/` package. Avoid one giant notebook or a single multi-thousand-line script. Keep to one orchestrating agent and modular deterministic tools. Do not introduce LangChain, a multi-agent framework, a web application, cloud deployment or a database.

## 16. Single-command interface and resumability

The complete pipeline runs with minimal intervention once a dataset configuration and an objective are supplied:

```bash
python -m agent3_replearn run \
  --config config/pbmc_multiome.yaml \
  --objective objectives/pbmc_multiome_joint.yaml \
  --resume

python -m agent3_replearn run \
  --config config/tcga_brca.yaml \
  --objective objectives/tcga_brca_pam50.yaml \
  --resume
```

The second command changes the data and the objective, not the orchestration. On the cluster, `bash scripts/submit_workflow.sh` submits the dependency-chained jobs (smoke → PBMC → TCGA feature extraction → TCGA modeling → finalize). Do not submit a second copy while one is active.

Stage commands for debugging:

```text
inspect, interpret-objective, plan, split, preprocess, tile, extract-features,
baseline, pilot, select, tune, final, embed, evaluate, downstream, verdict,
report, project-report, validate, run
```

Each stage writes a manifest and a completion marker only after validation. `--resume` skips work whose configuration, analysis-code hash and input checksums match, and never merely because a file exists. `--force <stage>` reruns that stage and invalidates dependent stages. Real-data runs refuse to start without a current-code smoke gate. WSI feature extraction is resumable per slide.

## 17. Output contract

Each dataset/objective pair produces:

```text
outputs/<dataset>/
├── dataset_profile.json
├── objective_spec.json
├── preprocessing_plan.json
├── applicability_matrix.json
├── decision_log.jsonl                 # every decision: rule_id, alternatives, evidence, reason
├── splits/split_manifest.json
├── preprocess/                        # fitted transforms, QC tables, cohort flow
├── features/                          # TCGA tile coordinates and features (ignored by Git, excluded from zips)
├── baseline/metrics.csv, manifest.json
├── pilot/budget_manifest.json, pilot_metrics.csv, failures.json, model_runs/
├── selection/decisions.json
├── tuning/tuning_results.csv, chosen_parameters.json
├── final/final_plan.json (frozen, hashed), model_card.json, training_curves/
├── representations/
│   ├── PBMC: representations.h5ad     # obs: barcode, split, curated labels, QC; obsm: X_<model>
│   └── TCGA: patient_embeddings.parquet, slide_embeddings.parquet, attention/
├── evaluation/metrics.csv, evidence.json, figures/
├── downstream/<analysis>/metrics.csv, predictions, figures/
├── verdict.json
├── evidence_bundle/
├── run_manifest.json
└── validation_results.json
```

Every exported representation carries sample identifiers, split membership, model ID, checkpoint hash, preprocessing hash and embedding definition (joint versus per-modality, normalization). Never write unlabeled arrays whose meaning cannot be reconstructed.

## 18. Reporting

Generate automatically:

```text
reports/generated_report_1.pdf   # PBMC Multiome
reports/generated_report_2.pdf   # TCGA-BRCA
reports/report.pdf               # project report: at most 4 body pages, excluding references/appendices
```

Each generated dataset report covers, in this order and leading with decisions and findings:

1. objective as interpreted, and the integration verdict in one paragraph;
2. modality inspection and pairing profile;
3. preprocessing per modality, with fit scopes and cohort flow;
4. strategy decisions: the applicability matrix with reasons;
5. selected encoders, fusion type and training objectives, and why;
6. pilot design, budget and selection evidence;
7. final training (curves, compute);
8. representation quality: structure, alignment, modality contribution, stability, confounding;
9. downstream analyses with CIs;
10. qualitative figures (UMAPs, retrieval examples, attention heatmaps);
11. approved and prohibited uses, limitations, reproducibility information.

Reports are built only from a validated evidence bundle. Numbers are template variables read from saved artifacts, and a claim-to-source check verifies them. Use compact tables, never full metric dictionaries. Choose figures by declared rules. Deterministic templated reporting is the baseline; an LLM narrative is optional, needs an approved key, and fails clearly when unavailable.

The four-page project report explains the agent rather than benchmarking methods: system architecture (with a figure), how the agent makes representation learning decisions (profile + objective → rules → pilot evidence → selection → verdict), tools and AI models used, results on both datasets, and strengths and limitations. A strong narrative is:

> The agent profiles each modality and the pairing structure, interprets the objective's supervision and downstream needs, and uses explicit rules to decide which integration strategies, encoders and objectives are eligible. It then pilots only those against single-modality and simple-fusion baselines under a declared budget, selects on validation evidence, and issues an integration verdict that can conclude that a single modality suffices.

## 19. Validation checklist

Implement an automated validator covering at least:

```text
[ ] no stale legacy-root path (/proj/...) in project files
[ ] nothing written inside either shared data directory (compare manifests/mtimes before and after)
[ ] Agent1/Agent2 projects and environments unmodified
[ ] local model store unmodified; every FM used matches its DOWNLOAD_MANIFEST.json SHA256 and revision
[ ] PYTHONNOUSERSITE enforced; no ~/.local path on sys.path in any job
[ ] assignment requirements captured in docs/ASSIGNMENT_REQUIREMENTS.md
[ ] both dataset configurations and objectives reproducibly defined
[ ] splits disjoint, manifested, patient-level for TCGA, created before modeling
[ ] all data-dependent transforms fitted on training data only
[ ] no test identifier present in any training or pretraining artifact
[ ] TCGA inputs exclude targets, molecular-derived fields, outcomes and post-diagnosis treatment
[ ] every registry family has a genuine adapter that passes a synthetic smoke test
[ ] every family has an applicability status with a reason for each dataset/objective
[ ] every eligible candidate was attempted in the pilot; failures remain visible
[ ] selection and tuning used validation evidence only; final plan frozen before test access
[ ] baselines evaluated with the same probe protocol and comparable tuning
[ ] structure, alignment, utility, modality contribution, stability, confounding reported separately
[ ] collapse/validity diagnostics pass for selected representations
[ ] verdict.json recomputed by the validator matches the saved file
[ ] representations exported with full provenance and identifiers
[ ] GPU ledger reconciled with sacct and within budget
[ ] generated_report_1.pdf and generated_report_2.pdf exist, open, and pass the claim-to-source check
[ ] report.pdf exists, opens, and has ≤ 4 body pages
[ ] environment, hardware, package versions, configs, seeds, model weight revisions recorded
[ ] if R or external tools were used: R lock file, per-run sessionInfo, and every home-directory install logged
```

Run unit tests, ruff, stage validation and a clean end-to-end smoke run on synthetic multimodal fixtures. Render and visually inspect every report page and figure. A file existing is not validation.

## 20. Smoke fixtures and robustness

- Synthetic paired multimodal fixtures with a known shared latent structure (two count modalities; an image-bag + tabular modality with a small synthetic tiled TIFF for the WSI tools). On these, alignment metrics must behave as expected (near-zero FOSCTTM for the true pairing, near-random for shuffled pairs) before any real data are used.
- An unseen-configuration regression test: a third synthetic dataset with a different modality combination must produce a valid, different plan without code changes.
- An objective-swap demonstration at the `plan` stage (cheap, no training required): for example, PBMC with a label-trainable objective, or TCGA with survival as the primary objective. It must change the applicability matrix and metrics, and is reported in an appendix.

## 21. Progress logging and handoff

Append to `docs/PROGRESS.md` after each phase: date and time, completed work, commands and Slurm job IDs, outputs and checkpoints, tests and validators passed, failures and fixes, scientific decisions and their evidence, GPU hours used, and the next action. Do not hide unresolved problems. Distinguish the states: implementation complete, smoke-tested, pilot-complete, final-training complete, scientifically validated, report-ready.

## 22. Execution order

### Phase 0: inspect and reconcile

1. Read this prompt fully. Check `hostname`, Slurm reachability and Sycamore account access (Section 2.2).
2. Inspect the Agent3 directory, logs and environment. Inspect Agent1 and Agent2 read-only for reusable infrastructure.
3. Write `docs/ASSIGNMENT_REQUIREMENTS.md` from the official assignment text supplied by the user (Project 3), plus the approved defaults in Section 0.
4. `git init` in the project root, add `.gitignore` (data, features, checkpoints, caches, outputs except small manifests), and commit the existing files. Append to `docs/PROGRESS.md`.

### Phase 1: environment

1. Install the real dependencies into `agent3`, including CUDA PyTorch for H100. Redirect the caches. Enforce `PYTHONNOUSERSITE=1`. If the plan includes R steps, run `module load r/4.6.0`, check the required R packages, install missing ones into the home R user library (Section 2.1), and export the R lock file.
2. Verify the PBMC label file against `annotations/SHA256SUMS` and the data against `SHA256SUMS`. Inventory the local pathology model store read-only and verify each populated folder against its `DOWNLOAD_MANIFEST.json`.
3. Validate the GPU runtime with a short `h100_short` job (CUDA available, bf16 matmul, a tiny forward/backward pass, one forward pass per available FM on random tiles, checking embedding dimensions), and the PDF toolchain. Export the lock files.

### Phase 2: schemas, CLI, inspectors, objectives, splits

1. Implement `DataProfile`, `ObjectiveSpec`, run, evidence and verdict schemas, and the CLI and stage manifests.
2. Implement the loaders and inspectors. Profile both datasets read-only, writing outputs into Agent3 only. Build the TCGA cohort flow and Agent3's own clinical table.
3. Implement the structured and deterministic-text objective parsers, the deterministic splits, and the leakage checks.

### Phase 3: registry and smoke tests

1. Implement the preprocessing tools and every registry adapter.
2. Pass synthetic smoke tests (training, encode, save/load, finite outputs, alignment sanity) through the automatic smoke gate.
3. Do not run real pilots before this suite passes.

### Phase 4: protocol pre-registration, baselines, and planning

1. Write `docs/METHOD_NOTES.md` with the rules, margins, thresholds, budgets, the probe protocol and the verdict rules, and commit it before any real pilot.
2. Generate the applicability matrices and cost estimates. Run the WSI throughput pilot on about 20 slides and fix the tile cap.
3. Run the baselines.

### Phase 5: production feature extraction (TCGA)

1. Run tiling and feature extraction as a resumable Slurm job array within the GPU budget, with bounded concurrent slide readers.
2. Verify feature-store completeness per slide; failures remain listed, never silently dropped.

### Phase 6: micro-pilots, selection, tuning, final training

1. Pilot every eligible candidate under the declared budget, preserving job IDs and logs.
2. Apply the pre-registered selection rules, tune at most two candidates per dataset, and freeze the final plan.
3. Train the final models and export the representations.

### Phase 7: held-out evaluation, downstream analyses, verdict

1. Open the test split once. Compute all evaluation categories and downstream analyses.
2. Run the sensitivity analyses (clinical input sets, site-held-out, missing modality) and the exploratory survival analysis.
3. Produce `verdict.json`.

### Phase 8: reports, validation, packaging

1. Build the evidence bundles. Render `generated_report_1.pdf`, `generated_report_2.pdf` and `report.pdf`.
2. Run the validators and visual QA on every page. Fix the presentation without touching the frozen analysis evidence.
3. Package `submission/source_code.zip` (source, configs, objectives, environment locks, docs, the three PDFs) and `submission/generated_outputs.zip` (representations, metrics, downstream results, figures, decision logs, verdicts; no raw data, tile features or checkpoints), with checksums and a manifest, and verify the archive sizes are reasonable for Canvas upload.
4. Review the Git status and prepare the handoff. Do not push or submit without the user's instruction.

## 23. Definition of done

The project is complete only when:

1. one command runs each dataset/objective pipeline with resume support;
2. modality inspection and objective interpretation visibly change the preprocessing, strategy, encoder, objective and metric decisions;
3. every registered family has a genuine, smoke-tested implementation, and every eligible candidate receives an empirical pilot;
4. selection uses validation evidence under pre-registered rules, and the test data are touched once, after freezing;
5. single-modality and simple-fusion baselines are evaluated fairly with the same probe protocol;
6. learned representations are exported with complete provenance;
7. structure, alignment, utility, modality contribution, stability and confounding are reported separately;
8. each dataset receives an evidence-based integration verdict with approved and prohibited uses;
9. reports are generated from validated artifacts and contain no invented values;
10. `generated_report_1.pdf`, `generated_report_2.pdf`, `report.pdf` (≤ 4 body pages) and the source code satisfy the Canvas deliverables;
11. the shared data directories, Agent1, Agent2 and their environments are untouched;
12. the repository is reproducible, documented and Git-ready.

Begin with Phase 0. Preserve existing work, record what is already complete, and move forward from the current validated state.
