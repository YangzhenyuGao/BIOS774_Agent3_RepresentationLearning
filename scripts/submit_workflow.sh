#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/slurm/common.sh"
python - <<'PY'
import fcntl
import subprocess
from agent3_replearn.utils import ROOT, read_json, write_json
with (ROOT / 'logs/workflow_submission.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    path = ROOT / 'logs/workflow_controller_job.json'
    if path.exists():
        old = read_json(path)
        result = subprocess.run(['squeue', '-h', '-j', str(old['job_id']), '-o', '%T'], capture_output=True, text=True)
        if result.stdout.strip():
            raise SystemExit('Existing workflow controller is active: ' + str(old['job_id']))
    result = subprocess.run(['sbatch', '--parsable', 'scripts/slurm/controller.sbatch'], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(result.stderr.strip())
    job = result.stdout.strip().split(';')[0]
    write_json(path, {'job_id': job})
    print('Workflow controller:', job)
PY
