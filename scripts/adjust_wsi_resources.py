"""Choose a predeclared tile cap that fits measured short-queue wall time."""
import json

from agent3_replearn.config import load_config
from agent3_replearn.utils import checksum, provenance, read_json, write_json


def choose_cap(estimates, shards=2, wall_hours=2., startup_seconds=90):
    feasible = [int(cap) for cap, hours in estimates.items()
                if hours / shards + startup_seconds / 3600 <= wall_hours]
    if not feasible:
        raise RuntimeError('No declared tile cap fits the queue wall time with measured safety/startup allowance')
    return max(feasible)


def main():
    cfg = load_config('config/tcga_brca.yaml')
    path = cfg.out / 'features/encoder_selection.json'
    snapshot = path.with_name('throughput_measurement.json')
    current = read_json(path)
    if 'operational_constraints' not in current:
        snapshot.write_bytes(path.read_bytes())  # Preserve the exact verified raw measurement.
    measured = read_json(snapshot)
    cap = choose_cap(measured['projected_hours_by_cap'])
    selected = {**measured, 'tile_cap': cap}
    selected['operational_constraints'] = dict(partition='h100_short', shards=2, array_parallel=1,
        wall_hours_per_task=2., startup_seconds_per_task=90,
        measurement_sha256=checksum(snapshot), original_budget_only_tile_cap=measured['tile_cap'],
        projected_hours_per_task=measured['projected_hours_by_cap'][str(cap)] / 2,
        reason='Predeclared cap chosen before TCGA fitting to retain measured twofold safety within the short-queue time limit',
        **provenance())
    write_json(path, selected)
    with (cfg.out / 'decision_log.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage='feature_resource_plan', rule_id='MEASURED_CAP_WITHIN_SHORT_WALL',
            alternatives=measured['projected_hours_by_cap'], selected_tile_cap=cap,
            evidence=selected['operational_constraints'],
            reason=selected['operational_constraints']['reason'])) + '\n')
    print('Operational WSI cap:', cap, '; short queue; two serial tasks; raw measurement preserved:', checksum(snapshot))


if __name__ == '__main__':
    main()
