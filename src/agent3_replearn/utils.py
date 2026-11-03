"""Provenance, filesystem boundaries and deterministic manifests."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default).encode()).hexdigest()


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    assert_project_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default, allow_nan=False))
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def assert_project_output(path):
    if not Path(path).resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Output escapes Agent3: {path}")


def code_hash(report=False):
    source = list((ROOT / "src").rglob("*.py")) + [ROOT / "pyproject.toml"]
    if report:
        source += list((ROOT / "templates").rglob("*"))
    else:
        source = [p for p in source if p.name not in {"reporting.py", "project_report.py"}]
    return digest({str(p.relative_to(ROOT)): checksum(p) for p in sorted(source) if p.is_file()})


def provenance():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    return dict(
        time_utc=datetime.now(timezone.utc).isoformat(),
        host=platform.node(),
        python=sys.version,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        git_commit=commit.stdout.strip(),
        analysis_code_hash=code_hash(),
    )


def check_environment():
    if os.environ.get("PYTHONNOUSERSITE") != "1" or any("/.local/" in p for p in sys.path):
        raise RuntimeError("Explicit PYTHONNOUSERSITE=1 and an isolated sys.path are required")


def require_compute():
    check_environment()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training and heavy data processing require a Slurm allocation")
    if os.environ.get('SLURM_JOB_GPUS') or int(os.environ.get('SLURM_GPUS_ON_NODE', '0')) > 0:
        identifier = (os.environ['SLURM_ARRAY_JOB_ID'] + '_' + os.environ['SLURM_ARRAY_TASK_ID']
                      if os.environ.get('SLURM_ARRAY_JOB_ID') else os.environ['SLURM_JOB_ID'])
        path = ROOT / 'logs/gpu_ledger.json'
        if not path.exists() or identifier not in {str(j['job_id']) for j in read_json(path)['jobs']}:
            raise RuntimeError('GPU allocation is not recorded in the project ledger; submit through scripts/gpu_jobs.py')


def seed_all(seed):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
