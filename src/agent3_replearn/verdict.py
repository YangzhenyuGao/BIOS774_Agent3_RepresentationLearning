"""Evidence-linked integration verdict with explicit project thresholds."""
from .schemas import Verdict


def compute_verdict(evidence, margin=.01):
    selected = evidence['selected']
    runs = evidence['models']
    singles = [m for m, row in runs.items() if row['fusion_type'] == 'none']
    best_single = max(singles, key=lambda m: runs[m]['classification']['macro_f1'])
    comparisons = evidence['bootstrap']['paired_differences']
    zero = dict(estimate=0., lower=0., upper=0.)
    single = comparisons.get(f'{selected}__minus__{best_single}', zero)
    fusion = comparisons.get(f'{selected}__minus__concatenation', zero)
    primary = evidence.get('primary_metrics', ['macro_f1'])
    metric_differences = {}
    for metric in primary:
        values = evidence['bootstrap'] if metric == 'macro_f1' else evidence['bootstrap'].get('secondary', {}).get(metric, {})
        pairwise = values.get('paired_differences', {})
        metric_differences[metric] = dict(
            single=pairwise.get(f'{selected}__minus__{best_single}', zero),
            fusion=pairwise.get(f'{selected}__minus__concatenation', zero))
    all_differences = [v for item in metric_differences.values() for v in item.values()]
    positive_primary_gain = any(item['single'].get('estimate', 0) > margin for item in metric_differences.values())
    no_major_primary_loss = all(item['single'].get('estimate', -1) >= -margin for item in metric_differences.values())
    flags = list(evidence.get('validity_flags', []))
    if selected in singles:
        verdict = 'single_modality_sufficient'
    elif all(v.get('estimate', 0) > margin and v.get('lower', -1) > 0 for v in all_differences) and not flags:
        verdict = 'integrate'
    elif positive_primary_gain and no_major_primary_loss:
        verdict = 'integrate_with_caution'
    else:
        verdict = 'single_modality_sufficient'
    row = runs[selected]
    return Verdict(integration_verdict=verdict, selected_representation=selected,
                   selected_strategy=row['family'], encoders=row['encoders'], fusion_type=row['fusion_type'],
                   training_objective=row['training_objective'],
                   comparison_to_best_single_modality={'model': best_single, **single,
                       'primary_metric_differences': {k: v['single'] for k, v in metric_differences.items()}},
                   comparison_to_simple_fusion_baseline={'model': 'concatenation', **fusion,
                       'primary_metric_differences': {k: v['fusion'] for k, v in metric_differences.items()}},
                   structure_evidence=row['structure'], alignment_evidence=row.get('alignment', {}),
                   utility_evidence=row['classification'], modality_contribution_evidence=evidence.get('ablations', {}),
                   confounding_and_validity_flags=flags,
                   approved_downstream_uses=['Exploratory within-cohort analysis under the recorded preprocessing/split'],
                   discouraged_or_prohibited_uses=evidence['prohibited_uses'],
                   uncertainty_and_limitations=evidence['limitations'],
                   conditions_for_reconsideration=['Independent donors/sites', 'Larger held-out cohort',
                                                   'Prospectively verified labels and external validation'])

