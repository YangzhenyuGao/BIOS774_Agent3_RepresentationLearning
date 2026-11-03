"""Known-latent fixtures and generic local paired arrays for regression tests."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..schemas import DataProfile, ModalityProfile, PairingProfile
from ..splits import assert_train_scope
from ..utils import checksum, read_json, write_json


def inspect_arrays(config):
    path = Path(config.modalities[0].path)
    arrays = np.load(path, allow_pickle=False)
    ids, labels = arrays['ids'], arrays['labels']
    modalities = [ModalityProfile(modality_id=m.modality_id, modality_type=m.modality_type,
                                  n_samples=len(ids), n_features=arrays['view_' + str(i)].shape[-1],
                                  storage_format='paired_npz', shape_and_channels=list(arrays['view_' + str(i)].shape),
                                  dtype=str(arrays['view_' + str(i)].dtype)) for i, m in enumerate(config.modalities)]
    profile = DataProfile(dataset_id=config.dataset_id, modalities=modalities,
        pairing=PairingProfile(pairing_key='sample_id', pairing_level='sample', n_paired=len(ids),
                               multiplicity={'tiles_per_sample': int(arrays['bags'].shape[1])} if 'bags' in arrays else {},
                               label_availability_by_split_candidate=pd.Series(labels).value_counts().to_dict()),
        source_checksums={str(path): checksum(path)}, label_caveats=['Synthetic known latent labels; not real biomedical evidence'],
        cohort_flow={'synthetic_samples': len(ids)})
    write_json(config.out / 'dataset_profile.json', profile)
    write_json(config.out / 'sample_index.json', dict(ids=ids.tolist(), labels=labels.tolist()))
    return profile


def preprocess_arrays(config):
    arrays = np.load(config.modalities[0].path, allow_pickle=False)
    ids = arrays['ids']
    index = {v: i for i, v in enumerate(ids)}
    split = read_json(config.out / 'splits/split_manifest.json')
    train = np.asarray([index[v] for v in split['groups']['train']])
    views = [arrays['view_0'], arrays['view_1']]
    transforms = [StandardScaler().fit(x[train]) for x in views]
    out = config.out / 'preprocess'
    out.mkdir(exist_ok=True)
    joblib.dump(transforms, out / 'array_transforms.joblib')
    for name, samples in split['groups'].items():
        ix = np.asarray([index[v] for v in samples])
        directory = out / name
        directory.mkdir(exist_ok=True)
        pd.DataFrame({'label': arrays['labels'][ix]}, index=pd.Index(ids[ix], name='sample_id')).to_parquet(directory / 'obs.parquet')
        for i, (x, transform) in enumerate(zip(views, transforms)):
            np.save(directory / f'view_{i}.npy', transform.transform(x[ix]).astype(np.float32))
        if 'bags' in arrays:
            np.save(directory / 'bags.npy', arrays['bags'][ix])
    write_json(out / 'preprocessing_manifest.json', dict(fit_scope=assert_train_scope(split['groups']['train'], split),
        input_columns=['synthetic continuous features'], sensitivity_columns=[],
        imputation='none; complete synthetic data', missingness_indicators='none'))


def load_arrays(directory, data):
    data['views'] = [np.load(directory / f'view_{i}.npy') for i in range(2)]
    if (directory / 'bags.npy').exists():
        data['bags'] = list(np.load(directory / 'bags.npy'))
    return data
