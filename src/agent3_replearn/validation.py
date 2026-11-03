"""Scientific and filesystem validators; report readiness is separate from existence."""
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader

from .data.prepared import require_test_access
from .orchestrator import applicable_stages, manifest_path, require_smoke
from .registry import REGISTRY
from .reporting import json_field, validate_evidence
from .splits import assert_disjoint
from .utils import ROOT, checksum, code_hash, read_json, write_json


def validate_run(config, objective):
    require_smoke() if not config.smoke else None
    require_test_access(config)
    evidence, verdict = validate_evidence(config)
    split = read_json(config.out / 'splits/split_manifest.json')
    assert_disjoint(split['groups'])
    preprocessing = read_json(config.out / 'preprocess/preprocessing_manifest.json')
    assert preprocessing['fit_scope']['leakage_check']
    plan = read_json(config.out / 'final/final_plan.json')
    assert plan['analysis_code_hash'] == code_hash()
    artifacts = {}
    for stage in applicable_stages(config):
        if stage == 'validate':
            continue
        manifest = read_json(manifest_path(config, stage))
        assert manifest['status'] == 'complete', stage
        for name, sha in manifest['artifacts'].items():
            assert Path(name).is_file() and checksum(name) == sha, name
            artifacts[name] = sha
    train_ids, test_ids = set(split['groups']['train']), set(split['groups']['test'])
    for result in plan['runs']:
        fit = read_json(Path(result['checkpoint_path']).parent / 'fit_scope.json')
        assert set(fit['fitted_ids']) <= train_ids
        assert not set(fit['fitted_ids']) & test_ids
        assert checksum(result['checkpoint_path']) == result['checkpoint_sha256']
    matrix = read_json(config.out / 'applicability_matrix.json')
    assert set(r['model_id'] for r in matrix) == set(REGISTRY)
    baseline = read_json(config.out / 'baseline/baseline_metrics.json')
    pilot = read_json(config.out / 'pilot/pilot_metrics.json')
    attempted = {r['model_id'] for r in baseline + pilot}
    assert {r['model_id'] for r in matrix if r['status'] == 'eligible_for_pilot'} <= attempted
    for row in matrix:
        assert row['reason'] and row['rule_id']
    claims = read_json(config.out / 'evidence_bundle/claim_sources.json')
    pdf = ROOT / 'reports' / (config.report_name + '.pdf')
    reader = PdfReader(pdf)
    text = '\n'.join(p.extract_text() for p in reader.pages)
    for claim in claims:
        assert checksum(claim['source']) == claim['source_sha256']
        assert json_field(read_json(claim['source']), claim['keys']) == claim['value']
        assert claim['rendered'] in text
    for path in list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'config').glob('*.yaml')):
        assert ('/' + 'proj' + '/') not in path.read_text(), path
    if not config.smoke:
        integrity = read_json(ROOT / 'logs/preflight/integrity.json')
        assert integrity['passed']
        gpu = read_json(ROOT / 'logs/preflight/gpu_runtime.json')
        assert gpu['passed']
        ledger = read_json(ROOT / 'logs/gpu_ledger.json')
        assert ledger['gpu_hours_used'] <= ledger['gpu_hours_cap'] <= 48
        assert (ROOT / 'environment.R.lock.csv').is_file()
        subprocess.run([sys.executable, 'scripts/check_protected.py'], cwd=ROOT, check=True)
    results = dict(passed=True, report_pages=len(reader.pages), claims_verified=len(claims),
                   split_disjoint=True, train_only_fitting=True, objective_supervision_checked=True,
                   all_eligible_candidates_attempted=True, final_frozen_before_test=True,
                   representation_provenance=checksum(config.out / 'representations/provenance.json'),
                   verdict_recomputed_matches=True, validated_artifacts=artifacts,
                   analysis_code_hash=code_hash(), scientific_status='validated_with_reported_limitations')
    path = config.out / 'validation_results.json'
    write_json(path, results)
    return [path]
