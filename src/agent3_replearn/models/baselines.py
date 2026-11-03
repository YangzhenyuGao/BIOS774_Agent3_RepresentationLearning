"""Shared block-scaling for all reference representations."""
import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import RepresentationAdapter


class BaselineAdapter(RepresentationAdapter):
    def __init__(self, model_id, params):
        self.model_id, self.params = model_id, params
        self.history = []

    def fit(self, train_data, val_data, params=None, seed=774):
        self.validate_input(train_data, val_data)
        self.scalers = [StandardScaler().fit(x) for x in train_data['views']]
        self.dimensions = [x.shape[1] for x in train_data['views']]
        return self

    def encode(self, data, modality=None):
        blocks = [s.transform(x).astype(np.float32) / np.sqrt(d)
                  for s, x, d in zip(self.scalers, data['views'], self.dimensions)]
        if modality is not None:
            return blocks[int(modality)]
        if self.model_id == 'view_a':
            return blocks[0]
        if self.model_id == 'view_b':
            return blocks[1]
        return np.concatenate(blocks, axis=1)

