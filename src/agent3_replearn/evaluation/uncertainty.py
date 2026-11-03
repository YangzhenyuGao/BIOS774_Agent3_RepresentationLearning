"""Paired uncertainty for secondary classification and fixed-gallery retrieval."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .downstream import classification_metrics


def bootstrap_secondary(y, predictions, classes, probabilities=None, query_scores=None,
                        replicates=500, seed=774):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    strata = [np.flatnonzero(y == c) for c in classes]
    indices = [np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata if len(ix)])
               for _ in range(replicates)]
    output = {}
    for metric in ['balanced_accuracy', 'worst_class_recall'] + (['ovr_auroc'] if probabilities else []):
        draws = {name: [] for name in predictions}
        class_draws = {name: {c: [] for c in classes} for name in predictions}
        for ix in indices:
            for name, pred in predictions.items():
                if metric in {'balanced_accuracy', 'worst_class_recall'}:
                    recalls = []
                    for c in classes:
                        mask = y[ix] == c
                        recall = float(np.mean(np.asarray(pred)[ix][mask] == c)) if mask.any() else np.nan
                        class_draws[name][c].append(recall)
                        if np.isfinite(recall):
                            recalls.append(recall)
                    value = balanced_accuracy_score(y[ix], np.asarray(pred)[ix]) if metric == 'balanced_accuracy' else min(recalls)
                else:
                    values = []
                    for j, c in enumerate(classes):
                        value = roc_auc_score(y[ix] == c, probabilities[name][ix, j]) if len(np.unique(y[ix] == c)) == 2 else np.nan
                        class_draws[name][c].append(value)
                        if np.isfinite(value):
                            values.append(value)
                    value = np.mean(values) if values else np.nan
                draws[name].append(value)
        intervals, differences = {}, {}
        for name, pred in predictions.items():
            point = classification_metrics(y, pred, probabilities.get(name) if probabilities else None, classes)[metric]
            finite = np.asarray(draws[name])[np.isfinite(draws[name])]
            intervals[name] = dict(estimate=point, lower=float(np.quantile(finite, .025)) if len(finite) else None,
                                   upper=float(np.quantile(finite, .975)) if len(finite) else None,
                                   valid_resamples=len(finite))
        for a in predictions:
            for b in predictions:
                if a == b:
                    continue
                values = np.asarray(draws[a]) - draws[b]
                finite = values[np.isfinite(values)]
                differences[f'{a}__minus__{b}'] = dict(
                    estimate=intervals[a]['estimate'] - intervals[b]['estimate']
                             if intervals[a]['estimate'] is not None and intervals[b]['estimate'] is not None else None,
                    lower=float(np.quantile(finite, .025)) if len(finite) else None,
                    upper=float(np.quantile(finite, .975)) if len(finite) else None)
        output[metric] = dict(intervals=intervals, paired_differences=differences,
                               resampling='paired patients/samples stratified by reference label')
        per_class = {}
        key = 'per_class_auroc' if metric == 'ovr_auroc' else 'per_class_recall'
        for name, pred in predictions.items():
            point = classification_metrics(y, pred, probabilities.get(name) if probabilities else None, classes)[key]
            per_class[name] = {}
            for c in classes:
                finite = np.asarray(class_draws[name][c])[np.isfinite(class_draws[name][c])]
                per_class[name][c] = dict(estimate=point[c], lower=float(np.quantile(finite, .025)) if len(finite) else None,
                                        upper=float(np.quantile(finite, .975)) if len(finite) else None,
                                        valid_resamples=len(finite))
        output[metric]['per_class_intervals'] = per_class
    if query_scores:
        n = len(next(iter(query_scores.values())))
        ix = rng.integers(0, n, (replicates, n))
        draws = {name: np.asarray(values)[ix].mean(1) for name, values in query_scores.items()}
        intervals = {name: dict(estimate=float(np.mean(query_scores[name])), lower=float(np.quantile(v, .025)),
                               upper=float(np.quantile(v, .975))) for name, v in draws.items()}
        differences = {f'{a}__minus__{b}': dict(estimate=intervals[a]['estimate']-intervals[b]['estimate'],
                       lower=float(np.quantile(draws[a]-draws[b], .025)), upper=float(np.quantile(draws[a]-draws[b], .975)))
                       for a in draws for b in draws if a != b}
        output['retrieval_recall_at_10'] = dict(intervals=intervals, paired_differences=differences,
                                               resampling='paired queries; gallery/candidate pool fixed', pool_size=n)
    return output
