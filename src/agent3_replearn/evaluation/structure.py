import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .downstream import labeled


def validity(z):
    z = np.asarray(z)
    if not np.isfinite(z).all():
        return dict(valid=False, flags=["nonfinite"])
    variance = z.var(0)
    singular = np.linalg.svd(z - z.mean(0), compute_uv=False)
    p = singular / max(singular.sum(), 1e-30)
    rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    duplicate = 1 - len(np.unique(np.round(z, 6), axis=0)) / len(z)
    flags = []
    if rank < 1.5:
        flags.append("low_effective_rank")
    if np.max(variance) < 1e-8:
        flags.append("constant_embedding")
    if duplicate > 0.95:
        flags.append("near_duplicate_collapse")
    return dict(
        valid=not flags,
        flags=flags,
        effective_rank=rank,
        near_duplicate_fraction=float(duplicate),
        minimum_variance=float(variance.min()),
        maximum_variance=float(variance.max()),
    )


def neighborhood_overlap(a, b, k=15):
    k = min(k, len(a) - 1)
    ia = NearestNeighbors(n_neighbors=k + 1).fit(a).kneighbors(a, return_distance=False)[:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1).fit(b).kneighbors(b, return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ia, ib)]))


def structure(z, labels, reference_views, seed=774):
    import anndata as ad
    import scanpy as sc
    from scipy.sparse.csgraph import connected_components

    a = ad.AnnData(np.asarray(z, dtype=np.float32))
    sc.pp.neighbors(a, n_neighbors=min(15, len(z) - 1), use_rep="X", random_state=seed)
    sc.tl.leiden(a, resolution=1.0, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    keep = labeled(labels)
    y = np.asarray(labels)[keep]
    clusters = a.obs["leiden"].to_numpy()[keep]
    output = dict(
        leiden_resolution=1.0,
        ari=float(adjusted_rand_score(y, clusters)),
        nmi=float(normalized_mutual_info_score(y, clusters)),
        label_silhouette=None,
        neighborhood_overlap={str(i): neighborhood_overlap(z, ref) for i, ref in enumerate(reference_views)},
    )
    if 1 < len(set(y)) < len(y):
        output["label_silhouette"] = float(
            silhouette_score(z[keep], y, sample_size=min(1500, len(y)), random_state=seed)
        )
    connectivity = {}
    graph = a.obsp["connectivities"].tocsr()
    for label in sorted(set(y)):
        ix = np.flatnonzero(np.asarray(labels) == label)
        _, component = connected_components(graph[ix][:, ix], directed=False)
        connectivity[label] = float(np.bincount(component).max() / len(ix))
    output["per_class_graph_connectivity"] = connectivity
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(z), min(500, len(z)), replace=False)
    output["global_distance_correlation"] = {}
    for i, reference in enumerate(reference_views):
        value = spearmanr(pdist(z[ix]), pdist(reference[ix])).statistic
        output["global_distance_correlation"][str(i)] = float(value) if np.isfinite(value) else None
    return output
