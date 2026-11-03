import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score


def alignment(a, b, metric="cosine"):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or len(a) < 2:
        raise ValueError("Aligned views require matching nonempty shapes and paired ordering")
    distances = pairwise_distances(a, b, metric=metric)
    paired = np.diag(distances)
    output = dict(
        candidate_pool_size=len(a),
        distance=metric,
        same_sample_pairing="row matched",
        foscttm_denominator="N-1 other samples, half credit for exact ties",
    )
    directional = []
    query_scores = []
    for direction, matrix in [("a_to_b", distances), ("b_to_a", distances.T)]:
        truth = paired[:, None]
        less = (matrix < truth - 1e-8).sum(1)
        ties = (np.abs(matrix - truth) <= 1e-8).sum(1) - 1
        fraction = (less + 0.5 * np.maximum(ties, 0)) / (len(a) - 1)
        output[f"{direction}_foscttm"] = float(fraction.mean())
        order = np.argsort(matrix, axis=1, kind="stable")
        for k in [1, 5, 10]:
            hits = (order[:, :min(k, len(a))] == np.arange(len(a))[:, None]).any(1)
            output[f"{direction}_recall_at_{k}"] = float(hits.mean())
            if k == 10:
                query_scores.append(hits.astype(float))
        directional.append(fraction.mean())
    output["foscttm"] = float(np.mean(directional))
    output["retrieval_recall_at_10"] = (output["a_to_b_recall_at_10"] + output["b_to_a_recall_at_10"]) / 2
    output['per_sample_recall_at_10'] = np.mean(query_scores, axis=0).tolist()
    output["modality_mixing_silhouette"] = float(
        silhouette_score(
            np.concatenate([a, b]),
            np.repeat([0, 1], len(a)),
            metric="cosine",
            sample_size=min(2000, 2 * len(a)),
            random_state=774,
        )
    )
    return output
