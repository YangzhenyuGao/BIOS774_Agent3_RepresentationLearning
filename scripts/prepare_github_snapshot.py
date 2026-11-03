"""Prepare a data-free public tree; never read credentials or publish a branch."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
raw = subprocess.check_output(['git', 'ls-files', '-s', '-z'], cwd=ROOT).decode()
allowed_reports = {'reports/report.pdf', 'reports/generated_report_1.pdf', 'reports/generated_report_2.pdf'}
forbidden_roots = {'data', 'models', 'features', 'outputs', 'logs', 'submission', 'checkpoints', '.cache', '.git'}
forbidden_suffixes = {'.h5', '.h5ad', '.h5mu', '.parquet', '.npy', '.npz', '.pt', '.pth',
                      '.safetensors', '.joblib', '.svs', '.tif', '.tiff', '.zip', '.mtx', '.bin', '.gz'}
manifest, text_entries = [], []
secret_patterns = [rb'(?m)^-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
                   rb'gh[pousr]_[A-Za-z0-9]{36}', rb'github_pat_[A-Za-z0-9_]{50,}']
for entry in raw.split('\0'):
    if not entry:
        continue
    metadata, relative = entry.split('\t', 1)
    mode = metadata.split()[0]
    path = Path(relative)
    if path.parts[0] in forbidden_roots or relative == 'prompt/install.md':
        continue
    if path.parts[0] == 'reports' and relative not in allowed_reports:
        continue
    assert mode in {'100644', '100755'}, f'Unsupported publication file mode: {relative}'
    assert path.suffix not in forbidden_suffixes, f'Forbidden artifact: {relative}'
    assert path.suffix not in {'.csv', '.tsv'} or relative == 'environment.R.lock.csv', relative
    source = ROOT / path
    assert source.is_file() and not source.is_symlink(), relative
    data = source.read_bytes()
    assert len(data) < 25 * 1024**2, f'Review large public file: {relative}'
    git_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    manifest.append(dict(path=relative, mode=mode, bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest(), git_blob_sha1=git_sha))
    if relative not in allowed_reports:
        assert not any(re.search(pattern, data) for pattern in secret_patterns), f'Credential-like content: {relative}'
        text_entries.append(dict(path=relative, mode=mode, type='blob', content=data.decode('utf-8')))
selected = {row['path'] for row in manifest}
expected_source = {str(path.relative_to(ROOT)) for path in (ROOT / 'src').rglob('*.py')}
assert expected_source <= selected, f'Missing Python sources: {sorted(expected_source - selected)}'
assert allowed_reports <= selected
out = ROOT / 'logs'
out.mkdir(exist_ok=True)
(out / 'github_publication_manifest.json').write_text(json.dumps(dict(
    scope='Core source, environment/configuration, documentation, aggregate-only examples and three PDFs',
    excluded='Raw data, individual sample records, embeddings, tile features, checkpoints, weights and private credentials',
    python_source_modules=len(expected_source), files=manifest), indent=2) + '\n')
(out / 'github_publication_text.json').write_text(json.dumps(text_entries, ensure_ascii=False))
print(json.dumps(dict(public_files=len(manifest), python_modules=len(expected_source),
                      bytes=sum(row['bytes'] for row in manifest), inline_text_bytes=(out / 'github_publication_text.json').stat().st_size)))
