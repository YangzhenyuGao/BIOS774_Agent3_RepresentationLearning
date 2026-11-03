"""10x sparse count loading with exact barcode and curated-label pairing."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from ..utils import read_json


def decode(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def load_multiome(config):
    matrices, metadata = {}, {}
    barcode_reference = None
    for modality in config.modalities:
        with h5py.File(modality.path, 'r') as f:
            g = f['matrix']
            matrix = sparse.csc_matrix((g['data'][:], g['indices'][:], g['indptr'][:]), shape=g['shape'][:])
            barcodes = decode(g['barcodes'][:])
            feature_types = decode(g['features/feature_type'][:])
            keep = feature_types == modality.parameters['feature_type']
            matrices[modality.modality_id] = matrix[keep].T.tocsr().astype(np.float32)
            metadata[modality.modality_id] = pd.DataFrame(
                {'id': decode(g['features/id'][:])[keep], 'name': decode(g['features/name'][:])[keep]})
        if barcode_reference is not None:
            assert np.array_equal(barcodes, barcode_reference), 'Modalities are not exactly paired'
        barcode_reference = barcodes
    obs = pd.DataFrame(index=pd.Index(barcodes, name='sample_id'))
    if config.labels.get('path'):
        labels = pd.read_csv(config.labels['path'], index_col=0)
        assert np.array_equal(labels.index.astype(str), barcodes), 'Label barcode mismatch'
        obs['label'] = labels[config.labels['column']].astype(str).values
        provenance = read_json(config.labels['provenance'])
        assert provenance['recommended_label_column'] == config.labels['column']
        assert Path(config.labels['path']).name == provenance['recommended_label_file']
    else:
        obs['label'] = '__unlabeled__'
    metrics_path = config.preprocessing.get('metrics_path')
    if metrics_path:
        metrics = pd.read_csv(metrics_path, index_col=0, low_memory=False)
        if not obs.index.isin(metrics.index).all():
            raise ValueError('QC metric barcodes do not contain every matrix barcode')
        metrics = metrics.reindex(obs.index)
        obs = obs.join(metrics, rsuffix='_vendor')
    return matrices, obs, metadata

