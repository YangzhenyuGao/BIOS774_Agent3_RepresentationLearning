"""Declared sensitivity analyses; refit patient/site models within their own scope."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data.prepared import load_split, subset
from .evaluation.downstream import fit_probes, score_probe
from .evaluation.structure import neighborhood_overlap
from .final import model_from_run, representative_runs
from .preprocessing.clinical import ClinicalTransform
from .registry import build_adapter
from .splits import site_sensitivity_split

COARSE = {'CD14 Mono': 'Monocyte', 'CD16 Mono': 'Monocyte', 'cDC': 'DC', 'pDC': 'DC',
          'HSPC': 'HSPC', 'Plasma': 'Plasma', 'NK': 'NK', 'Naive B': 'B',
          'Memory B': 'B', 'Intermediate B': 'B', 'MAIT': 'T', 'gdT': 'T', 'Treg': 'T'}


def coarse(labels):
    return np.asarray([COARSE.get(v, 'T' if v.startswith(('CD4', 'CD8')) else '__unlabeled__') for v in labels])


def merge_data(blocks):
    result = {}
    for key in blocks[0]:
        if key == 'views':
            result[key] = [np.concatenate([d[key][i] for d in blocks]) for i in range(2)]
        elif key in {'bags', 'tile_metadata', 'tile_images'}:
            result[key] = sum([d[key] for d in blocks], [])
        elif key == 'obs':
            result[key] = pd.concat([d[key] for d in blocks])
        elif isinstance(blocks[0][key], np.ndarray):
            result[key] = np.concatenate([d[key] for d in blocks])
        else:
            result[key] = blocks[0][key]
    return result


def sensitivities(config, objective, plan, train, test):
    selected_row = representative_runs(plan)[plan['selected']]
    model = model_from_run(selected_row)
    encoder = selected_row['hyperparameters'].get('input_encoder')
    val = load_split(config, 'validation', encoder=encoder)
    fine_tuned = plan['selected'] in {'fm_last_block', 'fm_image_text'}
    if fine_tuned:
        from .preprocessing.wsi import cache_finetune_tiles
        for data in [train, val, test]:
            cache_finetune_tiles(config, data)
    result = {}
    if 'counts' in train:
        a, b, c = model.encode(train), model.encode(val), model.encode(test)
        probes, _ = fit_probes(a, coarse(train['labels']), b, coarse(val['labels']))
        result['coarse_labels'] = score_probe(probes['logistic'], c, coarse(test['labels']))[0]
        result['coarse_labels']['mapping_version'] = 'v1 canonical PBMC families'
        result['coarse_labels']['mapping'] = COARSE
    elif 'site' in train['obs']:
        # Receptor removal refits the representation, not just masks its inputs.
        tr = load_split(config, 'train', encoder=encoder, clinical_set='without_receptors')
        va = load_split(config, 'validation', encoder=encoder, clinical_set='without_receptors')
        te = load_split(config, 'test', encoder=encoder, clinical_set='without_receptors', allow_test=True)
        if fine_tuned:
            for data in [tr, va, te]:
                cache_finetune_tiles(config, data)
        params = {**selected_row['hyperparameters'], 'clinical_columns': config.preprocessing['clinical_sensitivity_columns']}
        alternate = build_adapter(plan['selected'], params)
        alternate.fit(tr, va, seed=config.seed)
        probe, _ = fit_probes(alternate.encode(tr), tr['labels'], alternate.encode(va), va['labels'])
        result['without_receptors'] = dict(status='success',
            metrics=score_probe(probe['logistic'], alternate.encode(te), te['labels'])[0],
            actual_selected_family_refit=True, fit_ids=list(getattr(alternate, 'fit_ids', tr['ids'])),
            input_columns=config.preprocessing['clinical_sensitivity_columns'])
        combined = merge_data([train, val, test])
        held = site_sensitivity_split(combined['ids'], combined['obs'].site.to_numpy(), combined['labels'], config.seed)
        indexed = {v: i for i, v in enumerate(combined['ids'])}
        pieces = {s: subset(combined, [indexed[v] for v in ids]) for s, ids in held['groups'].items()}
        transform = ClinicalTransform(config.preprocessing['clinical_columns']).fit(pieces['train']['obs'])
        for split_name, data in pieces.items():
            data['split'] = split_name
            data['views'][1] = transform.encode(data['obs'])
            if fine_tuned:
                cache_finetune_tiles(config, data)
        mid = plan['selected']
        alternate = build_adapter(mid, selected_row['hyperparameters'].copy())
        alternate.fit(pieces['train'], pieces['validation'], seed=config.seed)
        fitted = set(getattr(alternate, 'fit_ids', pieces['train']['ids']))
        assert not fitted & set(pieces['test']['ids'])
        probe, _ = fit_probes(alternate.encode(pieces['train']), pieces['train']['labels'],
                              alternate.encode(pieces['validation']), pieces['validation']['labels'])
        metrics = score_probe(probe['logistic'], alternate.encode(pieces['test']), pieces['test']['labels'])[0]
        result['site_held_out'] = dict(status='success', exact_selected_family_refit=True,
                                       split=held, fit_ids=sorted(fitted), metrics=metrics,
                                       preprocessing='clinical refit; original local FM initialization for any tower refit')
        # Luminal A/B distinction uses the frozen main classifier, without selecting
        # a new binary model after seeing held-out labels.
        path = config.out / 'evaluation' / f'{plan["selected"]}_predictions.csv'
        pred = pd.read_csv(path)
        luminal = pred.truth.isin(['LumA', 'LumB'])
        result['luminal_a_vs_b'] = dict(n_test=int(luminal.sum()),
            macro_f1=float(f1_score(pred.loc[luminal, 'truth'], pred.loc[luminal, 'prediction'],
                                     labels=['LumA', 'LumB'], average='macro', zero_division=0)))
    # Seed stability on held-out data uses the frozen predeclared checkpoints;
    # deterministic models also receive training-subsample refits on validation.
    seed_rows = []
    for row in plan['runs']:
        if row['model_id'] == plan['selected']:
            seed_model = model_from_run(row)
            from pathlib import Path

            import joblib
            probes = joblib.load(Path(row['checkpoint_path']).parent / 'probes.joblib')
            metrics = score_probe(probes['logistic'], seed_model.encode(test), test['labels'])[0]
            seed_rows.append(dict(seed=row['seed'], validation_macro_f1=row['validation']['macro_f1'],
                                  test_macro_f1=metrics['macro_f1'], test_balanced_accuracy=metrics['balanced_accuracy']))
    result['seed_stability'] = dict(rows=seed_rows,
        validation_macro_f1_sd=float(np.std([r['validation_macro_f1'] for r in seed_rows])) if len(seed_rows) > 1 else None,
        test_macro_f1_mean=float(np.mean([r['test_macro_f1'] for r in seed_rows])),
        test_macro_f1_sd=float(np.std([r['test_macro_f1'] for r in seed_rows])) if len(seed_rows) > 1 else None,
        interpretation='Predeclared seed variation; subsample stability reported separately')
    stability = []
    original = model.encode(val)
    for seed in config.budget.seeds:
        rng = np.random.default_rng(seed)
        ix = rng.choice(len(train['ids']), int(.8 * len(train['ids'])), replace=False)
        adapter = build_adapter(plan['selected'], selected_row['hyperparameters'].copy())
        adapter.fit(subset(train, ix), val, seed=seed)
        stability.append(dict(seed=seed, training_fraction=.8,
                              validation_neighbor_overlap=neighborhood_overlap(original, adapter.encode(val))))
    result['subsample_stability'] = dict(rows=stability, same_protocol_for_deterministic_and_stochastic=True)
    baseline_stability = {}
    for mid in ['view_a', 'view_b', 'concatenation', 'wnn']:
        rows = [r for r in plan['runs'] if r['model_id'] == mid]
        if not rows or mid == plan['selected']:
            continue
        own_train = load_split(config, 'train', encoder=rows[0]['hyperparameters'].get('input_encoder'))
        own_val = load_split(config, 'validation', encoder=rows[0]['hyperparameters'].get('input_encoder'))
        reference = model_from_run(rows[0])
        original = reference.encode(own_val)
        values = []
        for seed in config.budget.seeds:
            ix = np.random.default_rng(seed).choice(len(own_train['ids']), int(.8 * len(own_train['ids'])), replace=False)
            candidate = build_adapter(mid, rows[0]['hyperparameters'].copy())
            candidate.fit(subset(own_train, ix), own_val, seed=seed)
            values.append(dict(seed=seed, training_fraction=.8,
                               validation_neighbor_overlap=neighborhood_overlap(original, candidate.encode(own_val))))
        baseline_stability[mid] = values
    result['baseline_subsample_stability'] = dict(models=baseline_stability,
        interpretation='Adapter refits conditional on the same fixed train-only preprocessing; not whole-preprocessing stability')
    if 'counts' in train:
        directory = config.out / 'baseline/r_wnn_crosscheck'
        if (directory / 'seurat_wsnn.mtx').exists():
            from scipy.io import mmread
            r_graph = mmread(directory / 'seurat_wsnn.mtx').tocsr()
            wnn_rows = [r for r in plan['runs'] if r['model_id'] == 'wnn']
            if wnn_rows:
                graph = model_from_run(wnn_rows[0]).training_graph
                values = []
                for i in range(graph.shape[0]):
                    left_row, right_row = graph.getrow(i), r_graph.getrow(i)
                    left = set(left_row.indices[np.argsort(left_row.data)[-15:]])
                    right = set(right_row.indices[np.argsort(right_row.data)[-15:]])
                    values.append(len(left & right) / max(1, min(len(left), len(right))))
                result['r_wnn_crosscheck'] = dict(train_only=True, training_cells=len(values),
                    top15_graph_overlap=float(np.mean(values)),
                    caveat='Canonical Seurat/Signac reference and muon implementation differ; curated labels are WNN-derived')
    return result
