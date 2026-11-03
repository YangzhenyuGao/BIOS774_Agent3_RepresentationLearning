"""Compare read-only boundary snapshots; never restore or edit protected files."""
from pathlib import Path

from preflight_integrity import snapshot

from agent3_replearn.utils import ROOT, read_json, write_json

before = read_json(ROOT / 'logs/preflight/protected_initial.json')
differences = {}
for directory, records in before.items():
    after = snapshot(Path(directory))
    changed = [p for p in records if p not in after or records[p] != after[p]]
    created = sorted(set(after) - set(records))
    if changed or created:
        differences[directory] = dict(changed_or_removed=changed, created=created)
result = dict(passed=not differences, differences=differences,
              scope='names, modes, sizes and nanosecond mtimes; access-time changes excluded',
              concurrent_external_changes_are_not_reverted=True)
write_json(ROOT / 'logs/preflight/protected_comparison.json', result)
if differences:
    raise RuntimeError('Protected-directory changes detected; inspect attribution before declaring storage-boundary validation')
print('All protected directories and environments match their initial metadata snapshots.')
