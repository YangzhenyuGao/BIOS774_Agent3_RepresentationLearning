"""Isolated candidate fitting, train-only probes and validation evidence."""
import argparse
import importlib
import resource
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import torch

from .config import Config
from .data.prepared import load_split
from .evaluation.alignment import alignment
from .evaluation.downstream import fit_probes, score_probe
from .evaluation.structure import structure, validity
from .registry import REGISTRY, build_adapter
from .schemas import ObjectiveSpec, RunResult
from .splits import assert_train_scope
from .utils import checksum, code_hash, provenance, read_json, require_compute, write_json


def work(request):
    require_compute()
    config = Config.model_validate(request['config'])
    objective = ObjectiveSpec.model_validate(request['objective'])
    model_id = request['model_id']
    spec = REGISTRY[model_id]
    if spec.supervised and objective.allowed_supervision != 'train_labels':
        raise ValueError('Worker refuses label-informed representation fitting for this objective')
    directory = Path(request['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    row = dict(model_id=model_id, family=spec.family, status='failed', seed=request['seed'],
               encoders=list(spec.encoders), fusion_type=spec.fusion, training_objective=spec.loss,
               hyperparameters=request['params'], objective_id=objective.objective_id,
               dataset_id=config.dataset_id, config_hash=config.config_hash,
               eligibility_reason=next(r['reason'] for r in read_json(config.out / 'applicability_matrix.json') if r['model_id'] == model_id),
               package_versions={p: version(p) for p in ['torch', 'scvi-tools', 'muon', 'mofapy2', 'numpy']},
               **provenance())
    try:
        imports = {'multivi': ['scvi'], 'wnn': ['muon', 'scanpy'], 'mofa': ['mofapy2.run.entry_point']}
        for package in imports.get(model_id, []):
            importlib.import_module(package)
        if code_hash() != request['analysis_code_hash']:
            raise RuntimeError('Worker code hash differs from submitted request')
        input_encoder = request['params'].get('input_encoder')
        train, val = load_split(config, 'train', encoder=input_encoder), load_split(config, 'validation', encoder=input_encoder)
        if 'encoder' in train:
            row['input_encoder'] = train['encoder']
            row['foundation_model_records'] = read_json(config.out / 'features/encoder_selection.json')['records'][train['encoder']]
        inputs = ['RNA PCA', 'ATAC TF-IDF/LSI'] if 'counts' in train else [
            'Frozen ' + train.get('encoder', 'synthetic image feature') + ' with patient pooling',
            'Train-fitted clinical/tabular encoder']
        row['encoders'] = ([inputs[0]] if model_id in {'view_a', 'histology_conch', 'histology_imagenet'} else
                           [inputs[1]] if model_id == 'view_b' else inputs + list(spec.encoders))
        if model_id == 'abmil':
            row['encoders'] = [inputs[0], 'gated attention MIL']
        if model_id in {'fm_last_block', 'fm_image_text'}:
            from .preprocessing.wsi import cache_finetune_tiles
            cache_finetune_tiles(config, train)
            cache_finetune_tiles(config, val)
        model = build_adapter(model_id, request['params'])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        model.fit(train, val, seed=request['seed'])
        if model_id in {'fm_last_block', 'fm_image_text'}:
            row['encoders'] = ['CONCH-Z last vision block', 'CONCH-Z last text block' if model_id == 'fm_image_text' else 'clinical MLP']
            row['foundation_model_records'] = model.encoder.record()
            row['pretrained_parameter_squared_change'] = model.pretrained_parameter_squared_change
        row['runtime_seconds'] = time.monotonic() - start
        fitted = getattr(model, 'fit_ids', train['ids'])
        scope = assert_train_scope(fitted, read_json(config.out / 'splits/split_manifest.json'))
        write_json(directory / 'fit_scope.json', dict(**scope, fitted_ids=list(fitted), validation_ids=list(val['ids'])))
        train_z, val_z = model.encode(train), model.encode(val)
        diagnostics = validity(val_z)
        row['validity'] = diagnostics
        if not diagnostics['valid']:
            raise ValueError('Collapsed/invalid representation: ' + str(diagnostics['flags']))
        probes, tuning = fit_probes(train_z, train['labels'], val_z, val['labels'])
        scores, _ = score_probe(probes['logistic'], val_z, val['labels'])
        knn, _ = score_probe(probes['knn'], val_z, val['labels'])
        row['validation'] = scores
        row['validation']['knn_macro_f1'] = knn['macro_f1']
        row['structure'] = structure(val_z, val['labels'], val['views'])
        try:
            a, b = model.encode(val, modality=0), model.encode(val, modality=1)
            if model_id in {'view_a', 'view_b', 'concatenation', 'simclr'} or a.shape[1] != b.shape[1]:
                from sklearn.linear_model import Ridge
                mapper = Ridge(alpha=1).fit(model.encode(train, modality=0), model.encode(train, modality=1))
                a = mapper.predict(a)
                joblib.dump(mapper, directory / 'alignment_probe.joblib')
                row['alignment_definition'] = 'train-fitted ridge cross-view alignment probe'
            else:
                row['alignment_definition'] = 'adapter per-modality embeddings in shared latent coordinates'
            row['alignment'] = alignment(a, b)
            row['validation']['retrieval_recall_at_10'] = row['alignment']['retrieval_recall_at_10']
        except NotImplementedError:
            row['alignment'] = dict(applicable=False, reason='No per-modality encoder')
        row.update(status='success', embedding_dim=int(val_z.shape[1]),
                   parameter_count=getattr(model, 'parameter_count', 0),
                   trainable_parameter_count=getattr(model, 'trainable_parameter_count', getattr(model, 'parameter_count', 0)),
                   epochs_or_steps_completed=getattr(model, 'epochs_completed', len(getattr(model, 'history', []))),
                   training_curves=getattr(model, 'history', []),
                   early_stopping_reason=getattr(model, 'stopping_reason', 'family-specific convergence/max iterations'),
                   peak_memory_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                   hardware=dict(device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'))
        checkpoint = directory / 'model.joblib'
        model.save(checkpoint)
        joblib.dump(probes, directory / 'probes.joblib')
        np.savez_compressed(directory / 'validation_embeddings.npz', ids=val['ids'], embedding=val_z)
        write_json(directory / 'probe_tuning.json', tuning)
        write_json(directory / 'training_curves.json', getattr(model, 'history', []))
        row['checkpoint_path'] = str(checkpoint)
        row['checkpoint_sha256'] = checksum(checkpoint)
        row['preprocessing_sha256'] = checksum(config.out / 'preprocess/preprocessing_manifest.json')
        row['embedding_paths'] = {'validation': str(directory / 'validation_embeddings.npz')}
    except Exception as exc:
        traceback.print_exc()
        row.update(status='failed', error_type=type(exc).__name__, error_message=str(exc))
    core = RunResult.model_validate({key: row[key] for key in RunResult.model_fields if key in row})
    row.update(core.model_dump(mode='json'))
    write_json(directory / 'result.json', row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    work(read_json(args.request))


if __name__ == '__main__':
    main()
