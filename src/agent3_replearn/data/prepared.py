"""Manifested split loading; held-out data require a frozen final-plan receipt."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from ..splits import assert_train_scope
from ..utils import checksum, digest, read_json, write_json
from ..preprocessing.clinical import ClinicalTransform


def preprocess_clinical(config):
    frame = pd.read_parquet(config.out / 'preprocess/clinical_cohort.parquet')
    manifest = read_json(config.out / 'splits/split_manifest.json')
    train_ids = manifest['groups']['train']
    primary = config.preprocessing.get('clinical_columns', ['age', 'stage', 'histological_type', 'er', 'pr', 'her2'])
    sensitivity = config.preprocessing.get('clinical_sensitivity_columns', ['age', 'stage', 'histological_type'])
    transforms = {name: ClinicalTransform(cols).fit(frame.loc[train_ids])
                  for name, cols in [('primary', primary), ('without_receptors', sensitivity)]}
    out = config.out / 'preprocess'
    joblib.dump(transforms, out / 'clinical_transforms.joblib')
    for split, ids in manifest['groups'].items():
        directory = out / split
        directory.mkdir(exist_ok=True)
        frame.loc[ids].to_parquet(directory / 'obs.parquet')
        for name, transform in transforms.items():
            np.save(directory / f'clinical_{name}.npy', transform.encode(frame.loc[ids]))
    write_json(out / 'preprocessing_manifest.json', dict(
        fit_scope=assert_train_scope(train_ids, manifest), input_columns=primary,
        sensitivity_columns=sensitivity, missingness_indicators='all original variables',
        imputation='train median numeric; missing category categorical',
        target_and_outcome_columns_are_inputs=False))


def require_test_access(config):
    p = config.out / 'final/final_plan.json'
    frozen = config.out / 'final/frozen.sha256'
    if not p.exists() or not frozen.exists() or checksum(p) != frozen.read_text().strip():
        raise RuntimeError('Test data locked until the final plan is frozen and hash-verified')


def load_split(config, split, encoder=None, clinical_set='primary', allow_test=False):
    if split == 'test':
        if not allow_test:
            raise RuntimeError('Explicit held-out evaluation access required')
        require_test_access(config)
    directory = config.out / 'preprocess' / split
    obs = pd.read_parquet(directory / 'obs.parquet')
    data = dict(ids=obs.index.astype(str).to_numpy(), labels=obs.label.fillna('__unlabeled__').astype(str).to_numpy(),
                obs=obs, split=split)
    if (directory / 'view_0.npy').exists():
        from .synthetic import load_arrays
        return load_arrays(directory, data)
    if (directory / 'rna_pca.npy').exists():
        data['views'] = [np.load(directory / 'rna_pca.npy'), np.load(directory / 'atac_lsi.npy')]
        data['counts'] = [sparse.load_npz(directory / f'{m}_counts.npz') for m in ['rna', 'atac']]
    else:
        encoder = encoder or read_json(config.out / 'features/encoder_selection.json')['primary_encoder']
        from ..preprocessing.wsi import patient_features
        means, bags, tile_metadata = patient_features(config, obs.index, encoder)
        data['views'] = [means, np.load(directory / f'clinical_{clinical_set}.npy')]
        data['bags'], data['tile_metadata'], data['encoder'] = bags, tile_metadata, encoder
        data['os_days'] = obs.os_days.to_numpy(dtype=float)
        data['os_event'] = obs.os_event.to_numpy(dtype=float)
    data['data_hash'] = digest(dict(ids=data['ids'], split=split,
                                    preprocessing=checksum(config.out / 'preprocess/preprocessing_manifest.json')))
    return data


def subset(data, indices):
    indices = np.asarray(indices)
    result = {}
    for key, value in data.items():
        if key in {'views', 'counts'}:
            result[key] = [x[indices] for x in value]
        elif key in {'bags', 'tile_metadata', 'tile_images'}:
            result[key] = [value[i] for i in indices]
        elif key == 'obs':
            result[key] = value.iloc[indices]
        elif isinstance(value, np.ndarray) and len(value) == len(data['ids']):
            result[key] = value[indices]
        else:
            result[key] = value
    return result
