import numpy as np
import pandas as pd
import pytest
from scipy import sparse


@pytest.fixture
def paired_data():
    rng = np.random.default_rng(774)
    blocks = []
    for split, n in [("train", 64), ("validation", 24), ("test", 24)]:
        z = rng.normal(size=(n, 6)).astype(np.float32)
        counts = rng.poisson(np.exp(np.clip(z @ rng.normal(size=(6, 24)), -2, 2))).astype(np.float32)
        peaks = rng.binomial(1, 1 / (1 + np.exp(-z @ rng.normal(size=(6, 28))))).astype(np.float32)
        ids = np.asarray([f"{split}_{i}" for i in range(n)])
        labels = np.where(z[:, 0] > 0, "class_a", "class_b")
        blocks.append(
            dict(
                ids=ids,
                labels=labels,
                views=[
                    z + rng.normal(0, 0.02, z.shape).astype(np.float32),
                    z + rng.normal(0, 0.02, z.shape).astype(np.float32),
                ],
                counts=[sparse.csr_matrix(counts), sparse.csr_matrix(peaks)],
                obs=pd.DataFrame({"label": labels}, index=ids),
            )
        )
    return blocks
