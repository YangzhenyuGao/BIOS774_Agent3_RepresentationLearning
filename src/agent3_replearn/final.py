"""Held-out representation export, evaluation, downstream evidence and verdict."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .data.prepared import load_split, require_test_access
from .evaluation.alignment import alignment
from .evaluation.confounding import confounding
from .evaluation.downstream import bootstrap, classification_metrics, labeled, score_probe
from .evaluation.structure import structure, validity
from .evaluation.uncertainty import bootstrap_secondary
from .registry import REGISTRY, build_adapter
from .utils import checksum, read_json, write_json
from .verdict import compute_verdict


def model_from_run(row):
    cls = type(build_adapter(row['model_id'], row['hyperparameters']))
    return cls.load(row['checkpoint_path'])


def representative_runs(plan):
    result = {}
    for row in plan['runs']:
        result.setdefault(row['model_id'], row)
    return result


def embed(config, plan):
    import anndata as ad
    require_test_access(config)
    runs = representative_runs(plan)
    data = {s: load_split(config, s, allow_test=s == 'test') for s in ['train', 'validation', 'test']}
    out = config.out / 'representations'
    out.mkdir(parents=True, exist_ok=True)
    metadata, embeddings = {}, {}
    obs = pd.concat([d['obs'].assign(split=s) for s, d in data.items()])
    paths = []
    for mid, row in runs.items():
        model = model_from_run(row)
        current = data
        if row['hyperparameters'].get('input_encoder'):
            current = {s: load_split(config, s, encoder=row['hyperparameters']['input_encoder'], allow_test=s == 'test')
                       for s in ['train', 'validation', 'test']}
        if mid in {'fm_last_block', 'fm_image_text'}:
            from .preprocessing.wsi import cache_finetune_tiles
            for d in current.values():
                cache_finetune_tiles(config, d)
        values = {s: model.encode(d) for s, d in current.items()}
        z = np.concatenate(list(values.values()))
        embeddings[mid] = z
        metadata[mid] = dict(model_id=mid, seed=row['seed'], checkpoint_sha256=row['checkpoint_sha256'],
                             preprocessing_sha256=row['preprocessing_sha256'],
                             embedding_definition='single modality' if REGISTRY[mid].fusion == 'none' else 'joint',
                             normalized=False, family=row['family'], dimension=z.shape[1])
        np.savez_compressed(out / f'{mid}.npz', ids=obs.index.to_numpy(dtype=str), embedding=z,
                            split=obs['split'].to_numpy(dtype=str))
        paths.append(out / f'{mid}.npz')
    if 'counts' in data['train'] or all(m.format == 'paired_npz' for m in config.modalities):
        a = ad.AnnData(sparse.csr_matrix((len(obs), 0), dtype=np.float32), obs=obs)
        for mid, z in embeddings.items():
            a.obsm['X_' + mid] = z
        a.uns['representation_provenance'] = {k: str(v) for k, v in metadata.items()}
        with ad.settings.override(allow_write_nullable_strings=True):
            a.write_h5ad(out / 'representations.h5ad')
        paths.append(out / 'representations.h5ad')
    else:
        frames = []
        for mid, z in embeddings.items():
            frame = obs[['split']].reset_index()
            frame['model_id'] = mid
            frame['embedding'] = list(z)
            frame['checkpoint_sha256'] = metadata[mid]['checkpoint_sha256']
            frame['preprocessing_sha256'] = metadata[mid]['preprocessing_sha256']
            frame['embedding_definition'] = metadata[mid]['embedding_definition']
            frames.append(frame)
        pd.concat(frames).to_parquet(out / 'patient_embeddings.parquet', index=False)
        paths.append(out / 'patient_embeddings.parquet')
        # Slide-level frozen means retain identifiers and the encoder receipt.
        import h5py
        selection = read_json(config.out / 'features/encoder_selection.json')
        inventory = pd.read_parquet(config.out / 'preprocess/slide_inventory.parquet')
        split_lookup = obs['split'].to_dict()
        slides = []
        for slide in inventory[inventory.primary_eligible].itertuples():
            with h5py.File(config.out / 'features' / selection['primary_encoder'] / f'{slide.slide_id}.h5') as f:
                mean = f['features'][:].mean(0)
            slides.append(dict(slide_id=slide.slide_id, patient_id=slide.patient_id,
                               split=split_lookup[slide.patient_id], embedding=mean.tolist(),
                               encoder=selection['primary_encoder'], encoder_receipt=selection['records'][selection['primary_encoder']]))
        pd.DataFrame(slides).to_parquet(out / 'slide_embeddings.parquet', index=False)
        paths.append(out / 'slide_embeddings.parquet')
    write_json(out / 'provenance.json', metadata)
    return paths + [out / 'provenance.json']


def evaluate(config, objective, plan):
    require_test_access(config)
    train, test = load_split(config, 'train'), load_split(config, 'test', allow_test=True)
    evidence = dict(selected=plan['selected'], primary_metrics=objective.primary_metrics, models={}, prohibited_uses=objective.prohibited_uses,
                    limitations=read_json(config.out / 'dataset_profile.json')['label_caveats'],
                    validity_flags=[], ablations={}, test_access=dict(final_plan_sha256=checksum(config.out / 'final/final_plan.json'),
                                                                     frozen_before_test=True))
    predictions = {}
    probabilities, query_scores = {}, {}
    out = config.out / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    for mid, row in representative_runs(plan).items():
        train = load_split(config, 'train', encoder=row['hyperparameters'].get('input_encoder'))
        test = load_split(config, 'test', encoder=row['hyperparameters'].get('input_encoder'), allow_test=True)
        model = model_from_run(row)
        directory = Path(row['checkpoint_path']).parent
        probes = joblib.load(directory / 'probes.joblib')
        if mid in {'fm_last_block', 'fm_image_text'}:
            from .preprocessing.wsi import cache_finetune_tiles
            cache_finetune_tiles(config, train)
            cache_finetune_tiles(config, test)
        a, b = model.encode(train), model.encode(test)
        metrics, pred = score_probe(probes['logistic'], b, test['labels'])
        knn_metrics, _ = score_probe(probes['knn'], b, test['labels'])
        predictions[mid] = pred['prediction']
        probabilities[mid] = pred['probabilities']
        saved = pd.DataFrame(dict(sample_id=test['ids'][pred['mask']], truth=pred['y'], prediction=pred['prediction']))
        for j, c in enumerate(pred['classes']):
            saved['probability_' + str(c)] = pred['probabilities'][:, j]
        saved.to_csv(out / f'{mid}_predictions.csv', index=False)
        record = dict(classification=metrics, knn=knn_metrics, validity=validity(b),
                       structure=structure(b, test['labels'], test['views']),
                       confounding=confounding(a, b, train['obs'], test['obs']),
                       family=row['family'], encoders=row['encoders'], fusion_type=row['fusion_type'],
                       training_objective=row['training_objective'], seed=row['seed'],
                       validation_macro_f1=row['validation']['macro_f1'])
        if not record['validity']['valid']:
            evidence['validity_flags'].append(mid + ': invalid held-out representation')
        try:
            za, zb = model.encode(test, modality=0), model.encode(test, modality=1)
            if (directory / 'alignment_probe.joblib').exists():
                za = joblib.load(directory / 'alignment_probe.joblib').predict(za)
            record['alignment'] = alignment(za, zb)
            query_scores[mid] = record['alignment']['per_sample_recall_at_10']
        except NotImplementedError:
            record['alignment'] = dict(applicable=False)
        if model.capabilities().get('supervised'):
            keep = labeled(test['labels'])
            prob = model.predict(test)[keep]
            cls = model.classes
            record['end_to_end_head'] = classification_metrics(np.asarray(test['labels'])[keep],
                np.asarray(cls)[prob.argmax(1)], prob, cls)
        if model.capabilities().get('missing_modality'):
            abl = {}
            for dropped in [0, 1]:
                masked = {**test, 'views': [v.copy() for v in test['views']]}
                masked['views'][dropped][:] = 0
                if 'counts' in masked:
                    masked['counts'] = [v.copy() for v in test['counts']]
                    masked['counts'][dropped] = sparse.csr_matrix(masked['counts'][dropped].shape, dtype=np.float32)
                if dropped == 0 and 'bags' in masked:
                    masked['bags'] = [np.zeros_like(v) for v in test['bags']]
                abl[str(dropped)] = score_probe(probes['logistic'], model.encode(masked), test['labels'])[0]
            evidence['ablations'][mid] = abl
        if model.capabilities().get('attention') and 'tile_metadata' in test:
            weights = model.attention(test)
            attention_dir = config.out / 'representations/attention'
            attention_dir.mkdir(exist_ok=True)
            for identifier, attention, coords in zip(test['ids'], weights, test['tile_metadata']):
                pd.DataFrame(coords).assign(attention=attention).to_parquet(attention_dir / f'{mid}_{identifier}.parquet', index=False)
        evidence['models'][mid] = record
    keep = labeled(test['labels'])
    classes = sorted(set(train['labels']) - {'filtered', '__unlabeled__'})
    evidence['bootstrap'] = bootstrap(np.asarray(test['labels'])[keep], predictions, classes,
                                       config.bootstrap_replicates, config.seed)
    evidence['bootstrap']['secondary'] = bootstrap_secondary(np.asarray(test['labels'])[keep], predictions, classes,
        probabilities=probabilities if 'site' in test['obs'] else None, query_scores=query_scores,
        replicates=config.bootstrap_replicates, seed=config.seed)
    if any('site' in r['confounding'] for r in evidence['models'].values()):
        selected_conf = evidence['models'][plan['selected']]['confounding'].get('site', {})
        if selected_conf.get('balanced_accuracy', 0) > selected_conf.get('chance_reference', 1) + .1:
            evidence['validity_flags'].append('Selected representation predicts tissue source site above the project caution margin')
        evidence['limitations'].append('Pathology FM pretraining overlap with TCGA cannot be excluded from local manifests')
    else:
        selected_conf = evidence['models'][plan['selected']]['confounding']
        if any(v.get('test_r_squared', 0) > .5 for v in selected_conf.values()):
            evidence['validity_flags'].append('Selected embedding predicts QC/depth with test R-squared >0.5; technical versus biological association unresolved')
    write_json(out / 'evidence.json', evidence)
    pd.DataFrame([dict(model_id=m, **{k: v for k, v in r['classification'].items() if isinstance(v, (int, float))})
                  for m, r in evidence['models'].items()]).to_csv(out / 'metrics.csv', index=False)
    return [out / 'evidence.json', out / 'metrics.csv'] + list(out.glob('*_predictions.csv'))


def downstream(config, objective, plan):
    selected_row = representative_runs(plan)[plan['selected']]
    encoder = selected_row['hyperparameters'].get('input_encoder')
    train, test = load_split(config, 'train', encoder=encoder), load_split(config, 'test', encoder=encoder, allow_test=True)
    models = {mid: model_from_run(row) for mid, row in representative_runs(plan).items()}
    out = config.out / 'downstream'
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    if 'cross_modal_prediction' in objective.downstream_tasks:
        from .evaluation.cross_modal import cross_modal_prediction
        results['cross_modal_prediction'] = cross_modal_prediction(train, test, models)
    selected = models[plan['selected']]
    if plan['selected'] in {'fm_last_block', 'fm_image_text'}:
        from .preprocessing.wsi import cache_finetune_tiles
        cache_finetune_tiles(config, train)
        cache_finetune_tiles(config, test)
    a, b = selected.encode(train), selected.encode(test)
    keep = labeled(test['labels'])
    nn = NearestNeighbors(n_neighbors=min(11, len(b))).fit(b)
    ix = nn.kneighbors(b, return_distance=False)[:, 1:]
    y = np.asarray(test['labels'])
    results['retrieval'] = dict(candidate_pool_size=len(b),
        subtype_precision_at_5=float(np.mean([np.mean(y[v[:5]] == y[i]) for i, v in enumerate(ix) if keep[i]])),
        subtype_precision_at_10=float(np.mean([np.mean(y[v[:10]] == y[i]) for i, v in enumerate(ix) if keep[i]])))
    if 'survival' in objective.downstream_tasks:
        from .evaluation.survival import survival
        results['survival'] = survival(a, b, train['obs'], test['obs'], out / 'survival', config.bootstrap_replicates)
    from .sensitivity import sensitivities
    results['sensitivity'] = sensitivities(config, objective, plan, train, test)
    write_json(out / 'results.json', results)
    from .figures import make_figures
    make_figures(config, objective, plan, train, test, a, b)
    return [p for p in out.rglob('*') if p.is_file()] + list((config.out / 'evaluation/figures').glob('*.png'))


def execute_final_stage(config, objective, stage):
    plan = read_json(config.out / 'final/final_plan.json')
    if stage == 'embed':
        return embed(config, plan)
    if stage == 'evaluate':
        return evaluate(config, objective, plan)
    if stage == 'downstream':
        return downstream(config, objective, plan)
    if stage == 'verdict':
        evidence = read_json(config.out / 'evaluation/evidence.json')
        verdict = compute_verdict(evidence, config.equivalence_margin)
        write_json(config.out / 'verdict.json', verdict)
        return [config.out / 'verdict.json']
    raise ValueError(stage)
