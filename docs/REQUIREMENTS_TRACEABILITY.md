# Requirement-to-implementation traceability

The authoritative implementation contract is prompt/prompt.md. This table maps
requirements to code/evidence locations. As of November 3, both datasets have
completed real baselines, every eligible pilot, final fitting, held-out analyses,
evidence-based verdicts and scientific validation. Frozen export checks cover
both plans, all 11 TCGA checkpoints, 7,427 patient-model rows and 1,132 slides.
Final report/visual acceptance and archive hashes are recorded in the release
artifacts described below; implementation presence alone is not acceptance.

| Requirement | Implementation | Evidence/acceptance |
|---|---|---|
| Modality/pairing inspection | inspector.py; data/multiome.py; data/tcga_brca.py | dataset_profile.json; clinical lineage/cohort flow |
| Structured/text objective | schemas.py; objective_parser.py | objective_spec.json; ambiguity and supervision tests |
| Objective/profile decisions | planner.py; registry.py | applicability_matrix.json; decision_log.jsonl; unseen-profile/objective-swap tests |
| Train-only fitting | splits.py; preprocessing/; data/prepared.py | fit-scope/split hashes; no test IDs in fit artifacts |
| Real method families | models/; registry.build_adapter | per-adapter synthetic training/save/load/new-sample tests |
| Frozen/actual fine-tuning distinction | pathology_fm.py; foundation_finetune.py | strict state matching; actual pretrained-parameter change test |
| Single command/resume | cli.py; orchestrator.py; scripts/agent3_cli.py | two end-to-end synthetic runs; content-validated resume; lossless mixed-dtype Parquet adapter |
| Failure isolation/retry | worker.py; execution.py | worker requests/logs; visible failure rows; OOM-only half-batch retry |
| Fair probes/selection | evaluation/downstream.py; selector.py | common grids; validation-only Pareto evidence; pre-registered protocol commit |
| Frozen held-out access | execution.final; data.prepared.require_test_access | final_plan.json and frozen.sha256 |
| Representation export | final.py; scripts/agent3_cli.py | h5ad/parquet + identifiers/splits/provenance |
| Separate evaluation categories | evaluation/; final.py; sensitivity.py | evidence.json; downstream/results.json; figures |
| Integration verdict | verdict.py | verdict recomputation matches validated metrics |
| Generated reports | reporting.py; templates/ | PDFs, claim_sources.json and PDF text verification; frozen-output QC/modality/site display maps; visual_review_validation.json |
| Four-page project report | project_report.py | report.pdf and page-count validation |
| Reproducibility/storage boundaries | utils.py; common.sh; preflight_integrity.py | environment locks, protected snapshot, checksum receipts, GPU ledger |
| Submission archives | scripts/package_submission.py | checksums/archive manifest; raw/features/weights excluded; both real runs scientifically validated; explicit visual-record hash checks |

Phase status, real job IDs and unresolved issues are appended to PROGRESS.md.
