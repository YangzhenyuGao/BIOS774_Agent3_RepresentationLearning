"""Read-only shared-input checks; write receipts only inside Agent3."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = Path('/vast/som/yunligrp/users/ygao')
PBMC = USER / 'BIOS774Agent/data/pbmc_granulocyte_sorted_10k'
TCGA = USER / 'imaging_fi/data/TCGA-BRCA'
MODELS = ['UNI2-h', 'Virchow2', 'CONCH-Z', 'MUSK']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(root):
    rows = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            p = Path(directory) / name
            st = p.lstat()
            rows[str(p.relative_to(root))] = [st.st_mode, st.st_size, st.st_mtime_ns]
    return rows


def main():
    out = ROOT / 'logs/preflight'
    out.mkdir(parents=True, exist_ok=True)
    initial = out / 'protected_initial.json'
    if not initial.exists():
        roots = [PBMC, TCGA]
        roots += [USER / 'models' / m for m in MODELS]
        roots += [USER / 'BIOS774Agent' / p for p in ['Agent1_DR', 'Agent2_Generative']]
        roots += [USER / 'conda/envs' / e for e in ['agent', 'agent2']]
        receipt = {}
        for root in roots:
            assert root.is_dir(), root
            print('Snapshot', root, flush=True)
            receipt[str(root)] = snapshot(root)
        initial.write_text(json.dumps(receipt, sort_keys=True))
    checks = []
    for sums in [PBMC / 'SHA256SUMS', PBMC / 'annotations/SHA256SUMS']:
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            expected, name = line.split(maxsplit=1)
            path = sums.parent / name.lstrip('*')
            assert path.resolve().is_relative_to(PBMC.resolve()), path
            actual = sha256(path)
            checks.append(dict(path=str(path), sha256=actual, matches=actual == expected))
            print(path.name, actual == expected, flush=True)
    manifests = []
    for name in MODELS:
        folder = USER / 'models' / name
        manifest = json.loads((folder / 'DOWNLOAD_MANIFEST.json').read_text())
        for row in manifest['files']:
            path = folder / row['path']
            assert path.resolve().is_relative_to(folder.resolve()), path
            actual = sha256(path)
            checks.append(dict(path=str(path), sha256=actual, matches=actual == row['sha256']))
            print(name, row['path'], actual == row['sha256'], flush=True)
        manifests.append(manifest)
    result = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  passed=all(r['matches'] for r in checks), checks=checks, models=manifests)
    (out / 'integrity.json').write_text(json.dumps(result, indent=2))
    assert result['passed'], 'Input/model checksum mismatch'


if __name__ == '__main__':
    main()
