"""Explicit deliverable allowlists; never redistribute raw inputs or FM derivatives."""
import argparse
import json
import subprocess
import zipfile

from visual_review_status import visual_review_status

from agent3_replearn.utils import ROOT, checksum, read_json, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--require-visual-review', action='store_true')
args = parser.parse_args()
review = visual_review_status(ROOT)
write_json(ROOT / 'reports/visual_review_validation.json', review)
if args.require_visual_review:
    assert review['passed'], 'An explicit inspection record must match every current PDF and page'

out = ROOT / 'submission'
out.mkdir(exist_ok=True)
project = read_json(ROOT / 'reports/project_report_validation.json')
assert project['passed'] and project['body_pages'] <= 4
for dataset in ['pbmc_multiome', 'tcga_brca']:
    assert read_json(ROOT / 'outputs' / dataset / 'validation_results.json')['passed']
tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, check=True, capture_output=True).stdout.decode().split('\0')
source = [ROOT / p for p in tracked if p and not p.startswith(('logs/', 'outputs/', '.cache/', 'checkpoints/'))]
source += [ROOT / 'reports' / p for p in ['generated_report_1.pdf', 'generated_report_2.pdf', 'report.pdf']]
generated = []
for dataset in ['pbmc_multiome', 'tcga_brca']:
    directory = ROOT / 'outputs' / dataset
    for folder in ['representations', 'evaluation', 'downstream', 'selection', 'evidence_bundle', 'splits', 'stages', 'final/training_curves']:
        generated.extend(p for p in (directory / folder).rglob('*') if p.is_file())
    generated.extend(directory / p for p in ['dataset_profile.json', 'objective_spec.json', 'preprocessing_plan.json',
                                           'applicability_matrix.json', 'decision_log.jsonl', 'verdict.json',
                                           'run_manifest.json', 'validation_results.json'])
    for stage, name in [('baseline', 'baseline_metrics.json'), ('pilot', 'pilot_metrics.json'), ('pilot', 'failures.json'),
                        ('pilot', 'budget_manifest.json'), ('tuning', 'chosen_parameters.json'),
                        ('tuning', 'tuning_results.json'), ('tuning', 'decisions_after_tuning.json')]:
        generated.append(directory / stage / name)
    generated.append(directory / 'preprocess/preprocessing_manifest.json')
    for name in ['cohort_flow.json', 'clinical_lineage.json', 'clinical_audit_crosscheck.json', 'os_discrepancies.json',
                 'selected_genes.csv', 'selected_peaks.csv', 'cell_qc.parquet']:
        path = directory / 'preprocess' / name
        if path.is_file():
            generated.append(path)
    for name in ['manifest.json', 'sessionInfo.txt', 'reference_metadata.csv', 'seurat_weights.csv']:
        path = directory / 'baseline/r_wnn_crosscheck' / name
        if path.is_file():
            generated.append(path)
    for name in ['encoder_selection.json', 'tiling_manifest.json', 'throughput_measurement.json']:
        path = directory / 'features' / name
        if path.is_file():
            generated.append(path)
    # Final plan references checkpoint hashes; checkpoint files are intentionally absent.
    generated.append(directory / 'final/final_plan.json')
    generated.append(directory / 'final/model_card.json')
generated.extend(p for p in (ROOT / 'outputs/robustness').rglob('*.json') if p.is_file())
for filename in ['logs/frozen_before_vector_export_fix.json', 'logs/frozen_vector_export_verification.json',
                 'reports/visual_review.json', 'reports/visual_review_validation.json',
                 'reports/qa_preview_manifest.json', 'reports/project_report_validation.json',
                 'reports/report_claim_sources.json', 'reports/project_report_resume.json',
                 'reports/report_audit_context.json', 'logs/report_revision_frozen_snapshot.json',
                 'logs/report_revision_release.json']:
    path = ROOT / filename
    if path.is_file():
        generated.append(path)
for path in generated:
    assert '/checkpoints/' not in str(path)
    if '/features/' in str(path):
        assert path.name in {'encoder_selection.json', 'tiling_manifest.json', 'throughput_measurement.json'} and path.suffix == '.json'
    assert path.suffix not in {'.pt', '.pth', '.safetensors', '.joblib', '.svs', '.tif', '.tiff'}
manifests = {}
for name, paths in [('source_code.zip', source), ('generated_outputs.zip', generated)]:
    paths = sorted(set(paths))
    assert all(p.is_file() for p in paths)
    with zipfile.ZipFile(out / name, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(ROOT)))
    with zipfile.ZipFile(out / name) as archive:
        assert archive.testzip() is None
    manifests[name] = dict(bytes=(out / name).stat().st_size, sha256=checksum(out / name),
                           files={str(p.relative_to(ROOT)): checksum(p) for p in paths})
    assert manifests[name]['bytes'] < 1024**3, 'Review archive >1 GB before declaring Canvas-ready'
write_json(out / 'manifest.json', manifests)
(out / 'SHA256SUMS').write_text(''.join(f'{v["sha256"]}  {k}\n' for k, v in manifests.items()))
print(json.dumps({name: {'bytes': value['bytes'], 'sha256': value['sha256']} for name, value in manifests.items()}, indent=2))
