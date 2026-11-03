"""Muon WNN training graph, with a declared inductive barycentric extension."""
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .base import RepresentationAdapter


class WNNAdapter(RepresentationAdapter):
    def __init__(self, params):
        self.model_id, self.params = 'wnn', params
        self.history = []

    def fit(self, train_data, val_data, params=None, seed=774):
        import anndata as ad
        import mudata as md
        import muon as mu
        import scanpy as sc
        self.validate_input(train_data, val_data)
        self.params.update(params or {})
        self.reference = [np.asarray(x, dtype=np.float32) for x in train_data['views']]
        n = len(self.reference[0])
        neighbors = min(self.params.get('n_neighbors', 15), n - 2)
        modalities = {}
        for name, values in zip(['a', 'b'], self.reference):
            a = ad.AnnData(values)
            a.obs_names = list(train_data['ids'])
            a.obsm['X_pca'] = values
            sc.pp.neighbors(a, n_neighbors=neighbors, use_rep='X_pca', random_state=seed)
            modalities[name] = a
        m = md.MuData(modalities)
        mu.pp.neighbors(m, n_neighbors=neighbors, n_bandwidth_neighbors=min(20, n - 2),
                        n_multineighbors=min(200, n - 1), random_state=seed, low_memory=True,
                        add_weights_to_modalities=True)
        self.training_graph = m.obsp['connectivities'].tocsr()
        sc.tl.diffmap(m, n_comps=min(self.params.get('latent_dim', 32) + 1, n - 2), random_state=seed)
        self.training_embedding = np.asarray(m.obsm['X_diffmap'][:, 1:], dtype=np.float32)
        if not np.isfinite(self.training_embedding).all():
            raise ValueError('WNN diffusion embedding is non-finite')
        self.weights = np.column_stack([m.mod[k].obs['mod_weight'].to_numpy() for k in ['a', 'b']])
        self.neighbor_models = [NearestNeighbors(n_neighbors=neighbors).fit(x) for x in self.reference]
        # Per-view projections into the same training WNN latent coordinates provide
        # a reproducible kNN retrieval reference; not a native WNN encoder claim.
        self.fit_ids = list(train_data['ids'])
        self.lookup = {v: i for i, v in enumerate(self.fit_ids)}
        return self

    def encode(self, data, modality=None):
        projections, weights = [], []
        for i, (nn, x) in enumerate(zip(self.neighbor_models, data['views'])):
            distance, index = nn.kneighbors(x)
            kernel = np.exp(-distance / np.maximum(np.median(distance, axis=1, keepdims=True), 1e-6))
            kernel /= kernel.sum(1, keepdims=True)
            projections.append((self.training_embedding[index] * kernel[:, :, None]).sum(1))
            weights.append((self.weights[index, i] * kernel).sum(1))
        if modality is not None:
            return projections[int(modality)].astype(np.float32)
        weights = np.column_stack(weights)
        weights /= np.maximum(weights.sum(1, keepdims=True), 1e-8)
        z = projections[0] * weights[:, :1] + projections[1] * weights[:, 1:]
        for i, identifier in enumerate(data['ids']):
            if identifier in self.lookup:
                z[i] = self.training_embedding[self.lookup[identifier]]
        return z.astype(np.float32)
