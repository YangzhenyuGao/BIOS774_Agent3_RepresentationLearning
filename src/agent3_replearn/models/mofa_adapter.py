"""MOFA+ Gaussian factors with a declared fixed-loading weighted-ridge extension."""
import numpy as np
from sklearn.preprocessing import StandardScaler

from ..utils import seed_all
from .base import RepresentationAdapter


class MOFAAdapter(RepresentationAdapter):
    def __init__(self, params):
        self.model_id, self.params = 'mofa', params
        self.history = []

    def fit(self, train_data, val_data, params=None, seed=774):
        from mofapy2.run.entry_point import entry_point
        self.validate_input(train_data, val_data)
        seed_all(seed)
        self.params.update(params or {})
        # MOFA on train-fitted PCA/LSI scores is a Gaussian multiview factor model.
        # This is explicitly not a count-likelihood model or a replacement for MultiVI.
        self.scalers = [StandardScaler().fit(x) for x in train_data['views']]
        blocks = [s.transform(x).astype(np.float64) for s, x in zip(self.scalers, train_data['views'])]
        entry = entry_point()
        entry.set_data_options(scale_groups=False, scale_views=False, center_groups=False)
        entry.set_data_matrix([[x] for x in blocks], likelihoods=['gaussian'] * len(blocks),
                              views_names=['view_a', 'view_b'], samples_names=[list(train_data['ids'])])
        entry.set_model_options(factors=min(self.params.get('latent_dim', 32), min(x.shape[1] for x in blocks)),
                                spikeslab_weights=True, ard_factors=True, ard_weights=True)
        entry.set_train_options(iter=self.params.get('epochs', 50), convergence_mode='fast',
                                seed=seed, verbose=False, quiet=True)
        entry.build()
        entry.run()
        self.weights = entry.model.nodes['W'].getExpectations()
        self.weights = [np.asarray(w['E']) for w in self.weights]
        tau = entry.model.nodes['Tau'].getExpectations()
        self.precisions = [np.asarray(t['E']).mean(axis=0).clip(1e-5) for t in tau]
        self.fit_ids = list(train_data['ids'])
        self.parameter_count = sum(w.size for w in self.weights)
        stats = entry.model.getTrainingStats()
        self.epochs_completed = int(np.count_nonzero(np.asarray(stats['time']) > 0))
        self.history = [{'iteration': i + 1, 'elbo': float(value)}
                        for i, value in enumerate(np.asarray(stats['elbo']).ravel()) if np.isfinite(value)]
        return self

    def encode(self, data, modality=None):
        blocks = [s.transform(x) for s, x in zip(self.scalers, data['views'])]
        chosen = range(len(blocks)) if modality is None else [int(modality)]
        k = self.weights[0].shape[1]
        precision = np.eye(k)
        numerator = np.zeros((len(blocks[0]), k))
        for i in chosen:
            w, tau = self.weights[i], self.precisions[i]
            precision += w.T @ (tau[:, None] * w)
            numerator += blocks[i] @ (tau[:, None] * w)
        return np.linalg.solve(precision, numerator.T).T.astype(np.float32)

