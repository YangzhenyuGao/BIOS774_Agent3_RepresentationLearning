"""One dependency-chained Slurm workflow, with no duplicate live GPU work."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from visual_review_status import visual_review_status

from agent3_replearn.utils import ROOT, checksum, code_hash, digest, read_json, write_json

STATE = ROOT / 'logs/workflow_state.json'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED'}


def phase_artifacts(name):
    """Validate and fingerprint outputs before trusting a completed Slurm job."""
    paths = set()

    def stages(dataset, through):
        from agent3_replearn.config import load_config
        from agent3_replearn.orchestrator import applicable_stages, manifest_path
        cfg = load_config(ROOT / 'config' / (dataset + '.yaml'))
        order = applicable_stages(cfg)
        for stage in order[:order.index(through) + 1]:
            path = manifest_path(cfg, stage)
            manifest = read_json(path)
            if manifest['status'] != 'complete':
                raise RuntimeError(f'Incomplete stage output: {path}')
            for filename, sha in manifest['artifacts'].items():
                target = ROOT / filename
                if not target.is_file() or checksum(target) != sha:
                    raise RuntimeError(f'Missing or changed stage output: {target}')
                paths.add(target)
            paths.add(path)

    if name == 'smoke':
        from agent3_replearn.orchestrator import require_smoke
        require_smoke()
        path = ROOT / 'logs/smoke_acceptance.json'
        paths.add(path)
        paths.update(ROOT / p for p in read_json(path)['evidence'])
    elif name == 'gpu_runtime':
        paths.add(ROOT / 'logs/preflight/gpu_runtime.json')
    elif name == 'preprocess':
        for dataset in ['pbmc_multiome', 'tcga_brca']:
            stages(dataset, 'preprocess')
        path = ROOT / 'outputs/pbmc_multiome/baseline/r_wnn_crosscheck/manifest.json'
        paths.add(path)
        reference = read_json(path)
        for field in ['input_hashes', 'output_hashes']:
            for filename, sha in reference[field].items():
                if not Path(filename).is_file() or checksum(filename) != sha:
                    raise RuntimeError(f'Missing or changed R WNN artifact: {filename}')
        paths.update(ROOT / p for field in ['input_hashes', 'output_hashes'] for p in reference[field])
        paths.update((ROOT / 'outputs/robustness').glob('*.json'))
    elif name == 'tiling':
        stages('tcga_brca', 'tile')
        paths.update((ROOT / 'outputs/tcga_brca/features/coordinate_qc').glob('*'))
    elif name == 'throughput':
        directory = ROOT / 'outputs/tcga_brca/features'
        snapshot = directory / 'throughput_measurement.json'
        paths.add(snapshot if snapshot.exists() else directory / 'encoder_selection.json')
    elif name == 'features':
        directory = ROOT / 'outputs/tcga_brca/features'
        selection = read_json(directory / 'encoder_selection.json')
        tiling = read_json(directory / 'tiling_manifest.json')
        for shard in range(2):
            path = directory / f'extraction_shard_{shard}.json'
            if read_json(path)['failures']:
                raise RuntimeError(f'Unresolved feature failures: {path}')
            # The full CLI subsequently writes an aggregate shard-0 receipt.
            # Stable feature hashes, rather than that mutable progress file,
            # establish content-valid resume of the completed array phase.
        paths.update([directory / 'encoder_selection.json', directory / 'tiling_manifest.json'])
        paths.update(directory / encoder / (Path(meta).stem + '.h5')
                     for encoder in selection['encoders'] for meta in tiling['coordinate_manifests'])
    elif name in {'pbmc', 'tcga'}:
        stages('pbmc_multiome' if name == 'pbmc' else 'tcga_brca', 'verdict')
    elif name == 'finalize':
        for dataset in ['pbmc_multiome', 'tcga_brca']:
            stages(dataset, 'validate')
        paths.update(ROOT / 'reports' / p for p in ['report.pdf', 'project_report_validation.json'])
        preview = ROOT / 'reports/qa_preview_manifest.json'
        paths.add(preview)
        paths.update(Path(page['image']) for report in read_json(preview)['reports'] for page in report['pages'])
        paths.update(ROOT / 'submission' / p for p in ['source_code.zip', 'generated_outputs.zip', 'manifest.json', 'SHA256SUMS'])
    if not paths:
        raise RuntimeError(f'No validated outputs for phase {name}')
    return {str(p): checksum(p) for p in sorted(paths)}


def valid_receipt(receipt):
    return bool(receipt) and all(Path(p).is_file() and checksum(p) == sha for p, sha in receipt.items())


def status(job):
    result = subprocess.run(['sacct', '--array', '-X', '-j', str(job), '-n', '-P', '--format=JobID,State,ExitCode'],
                             capture_output=True, text=True, check=True)
    rows = [line.split('|')[:3] for line in result.stdout.splitlines() if line.strip()]
    if not rows or any(r[1].split()[0].rstrip('+') in ACTIVE for r in rows):
        return 'active'
    if all(r[1] == 'COMPLETED' and r[2] == '0:0' for r in rows):
        return 'complete'
    return 'failed'


def wait(job):
    while True:
        state = status(job)
        if state == 'complete':
            return
        if state == 'failed':
            raise RuntimeError(f'Job {job} failed. Its logs/failure evidence are preserved; dependents are not submitted.')
        time.sleep(20)


def cpu_submit(script, dependency=None, minutes=120, cpus=8, memory='64G', arguments=()):
    if dependency and subprocess.run(['scontrol', 'show', 'job', str(dependency)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        dependency = None  # Already verified complete in sacct; controller has purged the old job.
    command = ['sbatch', '--parsable', '--account=rc_yunmli_pi', '--partition=small',
               f'--cpus-per-task={cpus}', f'--mem={memory}', f'--time={minutes}']
    if dependency:
        command.append(f'--dependency=afterok:{dependency}')
    command += [script, *arguments]
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip().split(';')[0]


def gpu_submit(script, purpose, dependency=None, minutes=240, array_tasks=1, array_parallel=2,
               arguments=(), partition='h100_priority'):
    if dependency and subprocess.run(['scontrol', 'show', 'job', str(dependency)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        dependency = None
    command = [sys.executable, 'scripts/gpu_jobs.py', '--partition', partition,
               '--purpose', purpose, '--minutes', str(minutes), '--memory', '96G',
               '--array-tasks', str(array_tasks), '--array-parallel', str(array_parallel)]
    if dependency:
        command += ['--dependency', dependency]
    command += ['--submit', script, *arguments]
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]


def tcga_submit(dependency):
    """Reserve full training time or a shorter, content-verified frozen continuation."""
    from agent3_replearn.config import load_config
    from agent3_replearn.objective_parser import interpret
    from agent3_replearn.orchestrator import applicable_stages, manifest_path, stage_signature

    config = load_config(ROOT / 'config/tcga_brca.yaml')
    objective = interpret(ROOT / 'objectives/tcga_brca_pam50.yaml')
    cached = True
    try:
        stages = applicable_stages(config)
        for stage in stages[:stages.index('final') + 1]:
            manifest = read_json(manifest_path(config, stage))
            if manifest['status'] != 'complete' or manifest['signature'] != stage_signature(config, objective, stage):
                cached = False
                break
            if not all(Path(path).is_file() and checksum(path) == sha for path, sha in manifest['artifacts'].items()):
                cached = False
                break
        plan = config.out / 'final/final_plan.json'
        cached = cached and checksum(plan) == (plan.parent / 'frozen.sha256').read_text().strip()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError):
        cached = False
    minutes = 60 if cached else 720
    purpose = ('TCGA frozen export/evaluation/sensitivity continuation' if cached else
               'TCGA baselines, all eligible pilots, final and sensitivities')
    print('TCGA allocation minutes:', minutes, '; content-verified final cache:', cached, flush=True)
    return gpu_submit('scripts/slurm/run.sbatch', purpose, dependency, minutes,
                      arguments=('config/tcga_brca.yaml', 'objectives/tcga_brca_pam50.yaml', 'verdict'))


def main():
    with (ROOT / 'logs/workflow_execution.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(STATE) if STATE.exists() else {'phases': {}}
        config_paths = ['config/base.yaml', 'config/pbmc_multiome.yaml', 'config/tcga_brca.yaml',
                        'objectives/pbmc_multiome_joint.yaml', 'objectives/tcga_brca_pam50.yaml']
        config_hash = digest({p: checksum(ROOT / p) for p in config_paths})
        if state.get('analysis_code_hash') != code_hash() or state.get('workflow_config_hash') != config_hash:
            active = [job for job in state['phases'].values() if status(job) == 'active']
            if active:
                raise RuntimeError(f'Code/configuration changed while prior jobs remain active: {active}; stop those jobs before resuming')
            state = dict(phases={}, receipts={}, analysis_code_hash=code_hash(),
                         workflow_config_hash=config_hash, prior_state=state)
        state.setdefault('receipts', {})
        # The raw throughput receipt remains byte-identical when an operational
        # queue decision derives a smaller production cap from that measurement.
        old = str(ROOT / 'outputs/tcga_brca/features/encoder_selection.json')
        snapshot = ROOT / 'outputs/tcga_brca/features/throughput_measurement.json'
        receipt = state['receipts'].get('throughput', {})
        if old in receipt and snapshot.exists() and checksum(snapshot) == receipt[old]:
            receipt[str(snapshot)] = receipt.pop(old)
        visual_review_path = ROOT / 'reports/visual_review.json'
        visual_review_hash = checksum(visual_review_path) if visual_review_path.is_file() else None
        if (state.get('report_code_hash') != code_hash(report=True)
                or state.get('visual_review_record_hash') != visual_review_hash):
            state['phases'].pop('finalize', None)
            state['complete'] = False
        state['visual_review_record_hash'] = visual_review_hash
        state['report_code_hash'] = code_hash(report=True)
        state['controller_job_id'] = os.environ['SLURM_JOB_ID']
        write_json(STATE, state)
        previous = None

        def queued(name, submit, dependency):
            if name in state['phases']:
                job = state['phases'][name]
                job_state = status(job)
                if job_state == 'failed' or (job_state == 'complete' and name in state['receipts']
                                              and not valid_receipt(state['receipts'][name])):
                    job = submit(dependency)
                    state['phases'][name] = job
            else:
                job = submit(dependency)
                state['phases'][name] = job
            write_json(STATE, state)
            return job

        def phase(name, submit):
            nonlocal previous
            job = queued(name, submit, previous)
            print(name, job, flush=True)
            wait(job)
            state['receipts'][name] = phase_artifacts(name)
            write_json(STATE, state)
            previous = job
            subprocess.run([sys.executable, 'scripts/gpu_jobs.py'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

        if 'smoke' not in state['phases']:
            try:
                from agent3_replearn.orchestrator import require_smoke
                require_smoke()
                accepted = read_json(ROOT / 'logs/smoke_acceptance.json')
                if accepted.get('slurm_job_id') and status(accepted['slurm_job_id']) == 'complete':
                    state['phases']['smoke'] = accepted['slurm_job_id']
            except (RuntimeError, FileNotFoundError):
                pass
        phase('smoke', lambda dep: cpu_submit('scripts/slurm/smoke.sbatch', dep, minutes=60, memory='64G'))
        # Reuse and wait for the real H100 acceptance test already in the ledger.
        runtime = ROOT / 'logs/preflight/gpu_runtime.json'
        if not runtime.exists() or not read_json(runtime).get('passed'):
            ledger = read_json(ROOT / 'logs/gpu_ledger.json')
            candidates = [j for j in ledger['jobs'] if 'Phase 1 H100' in j['purpose'] and status(j['job_id']) != 'failed']
            if candidates:
                wait(candidates[-1]['job_id'])
            else:
                phase('gpu_runtime', lambda dep: gpu_submit('scripts/slurm/gpu_runtime.sbatch', 'Phase 1 H100 acceptance retry', dep, 20))
        assert read_json(runtime)['passed']
        phase('preprocess', lambda dep: cpu_submit('scripts/slurm/inspect_preprocess.sbatch', dep))

        def pbmc_submit(dep):
            return gpu_submit('scripts/slurm/run.sbatch', 'PBMC baselines, eligible pilots, selection, final and held-out evaluation',
                              dep, 360, arguments=('config/pbmc_multiome.yaml', 'objectives/pbmc_multiome_joint.yaml', 'verdict'))

        def throughput_submit(dep):
            return gpu_submit('scripts/slurm/throughput.sbatch', 'Training-only 20-slide encoder throughput and fixed tile cap', dep, 60)

        queued('pbmc', pbmc_submit, previous)
        # Independent CPU tiling and the one-GPU throughput pilot can advance
        # while PBMC is queued/running. The ledger reserves at most two GPUs.
        phase('tiling', lambda dep: cpu_submit('scripts/slurm/wsi_tile.sbatch', dep, minutes=180))
        phase('throughput', throughput_submit)

        def features_submit(dep):
            return gpu_submit('scripts/slurm/wsi_features.sbatch',
                'Production WSI features; two serial shards, one GPU alongside PBMC',
                dep, 120, array_tasks=2, array_parallel=1, partition='h100_short')

        queued('features', features_submit, previous)
        phase('pbmc', pbmc_submit)
        # PBMC has released its GPU. Increase the existing array's throttle only
        # after the locked ledger proves the total remains at most two GPUs.
        subprocess.run([sys.executable, 'scripts/gpu_jobs.py', '--array-parent', state['phases']['features'],
                        '--set-array-parallel', '2'], cwd=ROOT, check=True)
        ledger = read_json(ROOT / 'logs/gpu_ledger.json')
        parallel = max(job.get('array_parallel', 2) for job in ledger['jobs']
                       if str(job.get('array_parent')) == state['phases']['features'])
        selection_path = ROOT / 'outputs/tcga_brca/features/encoder_selection.json'
        selection = read_json(selection_path)
        if selection['operational_constraints']['array_parallel'] != parallel:
            selection['operational_constraints']['array_parallel'] = parallel
            write_json(selection_path, selection)
        phase('features', features_submit)
        phase('tcga', tcga_submit)
        phase('finalize', lambda dep: cpu_submit('scripts/slurm/finalize.sbatch', dep, minutes=120))
        state['complete'] = True
        state['completion_scope'] = 'Automated computation, evidence validation and packaging'
        state['manual_visual_review'] = read_json(ROOT / 'reports/qa_preview_manifest.json')['manual_visual_review']
        review = visual_review_status(ROOT)
        state['visual_review_validation'] = review
        if review['passed']:
            state['manual_visual_review'] = 'completed; explicit inspection records match all current PDF/page hashes'
        write_json(STATE, state)


if __name__ == '__main__':
    main()
