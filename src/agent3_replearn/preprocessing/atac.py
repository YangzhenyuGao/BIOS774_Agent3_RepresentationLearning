"""Training-fitted accessibility filtering, TF-IDF and depth-diagnosed LSI."""

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


class ATACTransform:
    def __init__(self, n_peaks=30000, n_components=32, seed=774):
        self.n_peaks, self.n_components, self.seed = n_peaks, n_components, seed

    def tfidf(self, counts):
        x = (counts[:, self.selected] > 0).astype(np.float32).tocsr()
        denominator = np.asarray(x.sum(1)).ravel().clip(1)
        x = sparse.diags(1 / denominator) @ x
        x = (x @ sparse.diags(self.idf)).tocsr()
        x.data = np.log1p(x.data * 1e4)
        return x.astype(np.float32)

    def fit(self, counts):
        binary = counts > 0
        frequency = np.asarray(binary.sum(0)).ravel()
        eligible = np.flatnonzero(frequency >= max(3, int(counts.shape[0] * 0.001)))
        if len(eligible) < 3:
            raise ValueError("Too few accessible peaks after training-frequency filtering")
        p = frequency[eligible] / counts.shape[0]
        self.selected = np.sort(eligible[np.argsort(-p * (1 - p), kind="stable")[: self.n_peaks]])
        self.idf = np.log1p(counts.shape[0] / (1 + frequency[self.selected])).astype(np.float32)
        self.svd = TruncatedSVD(
            n_components=min(self.n_components + 1, len(self.selected) - 1, counts.shape[0] - 1),
            random_state=self.seed,
        )
        z = self.svd.fit_transform(self.tfidf(counts))
        depth = np.log1p(np.asarray(counts.sum(1)).ravel())
        corr = np.corrcoef(z[:, 0], depth)[0, 1]
        self.first_component_depth_correlation = float(corr) if np.isfinite(corr) else 0
        self.drop_first = abs(self.first_component_depth_correlation) > 0.5
        self.scaler = StandardScaler().fit(z[:, int(self.drop_first) :][:, : self.n_components])
        return self

    def encode(self, counts):
        z = self.svd.transform(self.tfidf(counts))[:, int(self.drop_first) :][:, : self.n_components]
        return self.scaler.transform(z).astype(np.float32)
