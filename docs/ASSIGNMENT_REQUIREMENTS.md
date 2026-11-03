# Project 3 requirements and approved execution contract

The official user-supplied assignment is **Building an AI Agent for Multimodal
Representation Learning**. Its emphasis is an intelligent, reusable agent, not
an exhaustive benchmark or the highest benchmark score.

## Official requirements

Given a multimodal biomedical dataset and user objective, automatically inspect
modalities; recommend and execute preprocessing; choose integration strategies,
encoders and training objectives; train or fine-tune; evaluate representations;
perform downstream analysis; and produce a report explaining decisions and findings.
Selection must adapt to the data and objective rather than replay a fixed pipeline.

The system must be capable of appropriate class methods, including contrastive,
self-supervised, supervised, CLIP-style multimodal learning and pretrained
multimodal foundation-model fine-tuning. It need not apply every method everywhere.
Evaluate on at least two multimodal biomedical datasets.

Canvas deliverables:

1. Source code: a complete single-command pipeline after supplying a dataset and
   objective, with minimal manual intervention.
2. For each dataset: learned representations, quantitative evaluation, downstream
   results, method/preprocessing decisions, and generated_report_1 or
   generated_report_2.
3. report.pdf: at most four body pages (references/appendices excluded), describing
   system architecture, agent decisions, tools/models, results on both datasets,
   strengths and limitations. Focus on agent design and behavior.

Evaluation covers automation/autonomy, analysis and modeling decisions, correct
representation-learning methods, representation and downstream quality, reports
and interpretations, robustness, software design and reproducibility.

## User-approved implementation defaults

The detailed contract is `prompt/prompt.md`, read in full before implementation.
The user authorized execution on November 2, 2026 and supplied a deadline of
**November 10, 2026**. Current client timezone is America/Indianapolis (Eastern);
no exact submission time was supplied or assumed.

- Report 1: granulocyte-sorted PBMC Multiome; curated FixFlip labels used for
  evaluation only. Label provenance and WNN annotation bias must be reported.
- Report 2: TCGA-BRCA diagnostic primary-tumor WSI plus baseline clinical data;
  patient-level PAM50 LumA/LumB/Basal/Her2 prediction. Normal-like excluded from
  primary classification with its count reported; survival is exploratory.
- Clinical inputs: age, stage, histological type and ER/PR/HER2, with missingness;
  sensitivity analysis excludes receptors. Molecular-derived, outcome and
  post-diagnosis treatment fields are never input features.
- Only four pathology FMs: local UNI2-h, Virchow2, CONCH-Z and MUSK, verified
  against manifests. Include an explicitly ImageNet-pretrained baseline.
- Sycamore H100: account h100_harlin, at most two concurrent GPUs, aggregate
  48 GPU-hours including tests and pilots. Reconcile a ledger against sacct.
- A deterministic planner and templated reporting form the reproducible core.
  No external LLM without an approved key; do not claim LLM use.
- Python packages only in the existing agent3 Conda environment, with explicit
  PYTHONNOUSERSITE=1; all caches and derivatives stay within Agent3. R 4.6.0 may
  use the existing personal home library, with package/session records.
- Shared PBMC/TCGA inputs, the model store, Agent1/Agent2 and their environments
  are read-only. Never reuse the other TCGA project's cohorts or splits.
- Train-only preprocessing and fitting; validation-only model selection; freeze
  the final plan before held-out test evaluation. Failures remain visible.
- Source/data/code hashes govern resume and smoke gates. Evidence governs
  reports and the integration verdict; single-modality sufficiency is allowed.
- Local Git commits are authorized; no GitHub push, Canvas submission or weight
  redistribution is authorized. Submission archives exclude raw data, tiles,
  feature stores, FM weights and fine-tuned derivatives.

Execution states must distinguish implementation, smoke testing, pilots, final
training, scientific validation and report readiness. Phase 8 packaging and all
12 definition-of-done conditions in the contract are required for completion.
