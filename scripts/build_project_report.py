"""Reuse the accepted PDF bytes only when every report input/output still matches."""
from pathlib import Path

from agent3_replearn.config import load_config
from agent3_replearn.project_report import build_project_report
from agent3_replearn.utils import ROOT, checksum, code_hash, digest, read_json, require_compute, write_json

require_compute()
sources = [ROOT / 'docs/REFERENCES.json', ROOT / 'docs/PUBLICATION_LINKS.json',
           ROOT / 'outputs/robustness/objective_swap_summary.json']
for filename in ['pbmc_multiome.yaml', 'tcga_brca.yaml']:
    config = load_config(ROOT / 'config' / filename)
    sources.extend(config.out / name for name in ['validation_results.json', 'evaluation/evidence.json',
                   'verdict.json', 'dataset_profile.json', 'applicability_matrix.json',
                   'final/final_plan.json', 'final/frozen.sha256', 'downstream/results.json',
                   'evidence_bundle/pilot_summary.json', 'preprocess/preprocessing_manifest.json'])
sources.extend([ROOT / 'logs/gpu_ledger.json', ROOT / 'outputs/tcga_brca/features/encoder_selection.json'])
sources.extend(ROOT / f'outputs/pbmc_multiome/pilot/model_runs/multivi/{seed}/attempt_0/worker.log'
               for seed in [774, 775, 776])
inputs = dict(sources={str(path): checksum(path) for path in sources}, report_code_hash=code_hash(report=True))
signature = digest(inputs)
receipt = ROOT / 'reports/project_report_resume.json'
cached = read_json(receipt) if receipt.is_file() else {}
if cached.get('signature') == signature and all(Path(path).is_file() and checksum(path) == sha
                                               for path, sha in cached.get('artifacts', {}).items()):
    assert cached['artifacts'], 'Empty report cache cannot establish acceptance'
    print('Validated resume: project report (all input and output hashes match)')
else:
    build_project_report()
    assert inputs['sources'] == {str(path): checksum(path) for path in sources}, 'Report sources changed during render'
    artifacts = [ROOT / 'reports' / name for name in ['report.pdf', 'report.html', 'architecture.png',
                 'report_audit_context.json', 'report_claim_sources.json', 'project_report_validation.json']]
    write_json(receipt, dict(signature=signature, inputs=inputs,
               artifacts={str(path): checksum(path) for path in artifacts},
               scope='Presentation cache only. Never changes scientific evidence or grants visual approval.'))
    print('Built and fingerprinted project report')
