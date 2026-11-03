"""Pre-registered epsilon-Pareto validation selection; no test inputs."""
import numpy as np


def aggregate(rows, primary_metrics):
    grouped = {}
    for row in rows:
        if row['status'] == 'success' and row.get('validity', {}).get('valid', False):
            grouped.setdefault(row['model_id'], []).append(row)
    result = []
    for model_id, runs in sorted(grouped.items()):
        metrics = {key: float(np.mean([r['validation'][key] for r in runs if key in r['validation']]))
                   for key in primary_metrics if any(key in r['validation'] for r in runs)}
        sd = {key: float(np.std([r['validation'][key] for r in runs if key in r['validation']], ddof=0)) for key in metrics}
        result.append(dict(model_id=model_id, metrics=metrics, seed_sd=sd,
                           seeds=[r['seed'] for r in runs], successful_runs=len(runs),
                           runtime_seconds=float(np.mean([r['runtime_seconds'] for r in runs])),
                           trainable_parameters=max(r.get('trainable_parameter_count', 0) for r in runs),
                           embedding_dim=max(r['embedding_dim'] for r in runs),
                           family=runs[0]['family']))
    return result


def select(rows, primary_metrics, margin=.01, maximum=2):
    candidates = aggregate(rows, primary_metrics)
    if not candidates:
        raise ValueError('No valid successful candidates remain')
    pareto = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if candidate is other:
                continue
            common = [k for k in primary_metrics if k in candidate['metrics'] and k in other['metrics']]
            if not common:
                continue
            if all(other['metrics'][k] >= candidate['metrics'][k] - margin for k in common) and any(
                    other['metrics'][k] > candidate['metrics'][k] + margin for k in common):
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    # No weighted score: primary metric first, then equivalence grouping and simplicity.
    primary = primary_metrics[0]
    best = max(c['metrics'].get(primary, -np.inf) for c in pareto)
    equivalent = [c for c in pareto if c['metrics'].get(primary, -np.inf) >= best - margin]
    for metric in primary_metrics[1:]:
        available = [c for c in equivalent if metric in c['metrics']]
        if available:
            threshold = max(c['metrics'][metric] for c in available) - margin
            equivalent = [c for c in available if c['metrics'][metric] >= threshold]
    equivalent.sort(key=lambda c: (c['trainable_parameters'], c['embedding_dim'],
                                    round(c['runtime_seconds']), c['model_id']))
    remaining = [c for c in pareto if c not in equivalent]
    remaining.sort(key=lambda c: -c['metrics'].get(primary, -np.inf))
    shortlist = [c['model_id'] for c in (equivalent + remaining)[:maximum]]
    return dict(rule_id='EPSILON_PARETO_SEQUENTIAL_QUALITY_CONSTRAINTS_THEN_SIMPLICITY', primary_metrics=primary_metrics,
                equivalence_margin=margin, candidate_evidence=candidates,
                pareto_set=[c['model_id'] for c in pareto], shortlist=shortlist,
                selected=shortlist[0], used_test_data=False)
