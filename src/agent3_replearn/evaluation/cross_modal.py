"""RNA→accessibility prediction without supplying held-out accessibility to predictors."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors


def binary_prediction_metrics(target, scores):
    per_cell, per_peak = [], []
    for t, p in zip(target, scores):
        if len(np.unique(t)) == 2:
            per_cell.append([roc_auc_score(t, p), average_precision_score(t, p)])
    for t, p in zip(target.T, scores.T):
        if len(np.unique(t)) == 2:
            per_peak.append([roc_auc_score(t, p), average_precision_score(t, p)])
    return dict(per_cell_auroc=float(np.mean(per_cell, axis=0)[0]) if per_cell else None,
                per_cell_auprc=float(np.mean(per_cell, axis=0)[1]) if per_cell else None,
                per_peak_auroc=float(np.mean(per_peak, axis=0)[0]) if per_peak else None,
                per_peak_auprc=float(np.mean(per_peak, axis=0)[1]) if per_peak else None,
                eligible_cells=len(per_cell), eligible_peaks=len(per_peak))


def cross_modal_prediction(train, test, models):
    # Peaks are selected entirely by training frequency; evaluation cap makes the
    # metrics tractable and includes a broad deterministic frequency range.
    frequency = np.asarray((train['counts'][1] > 0).mean(0)).ravel()
    eligible = np.flatnonzero((frequency >= .01) & (frequency <= .99))
    if len(eligible) > 1000:
        order = eligible[np.argsort(frequency[eligible])]
        eligible = order[np.linspace(0, len(order) - 1, 1000).astype(int)]
    truth = (test['counts'][1][:, eligible] > 0).toarray().astype(int)
    nn = NearestNeighbors(n_neighbors=min(30, len(train['ids']))).fit(train['views'][0])
    ix = nn.kneighbors(test['views'][0], return_distance=False)
    binary_train = (train['counts'][1][:, eligible] > 0).astype(np.float32)
    knn = np.stack([np.asarray(binary_train[v].mean(0)).ravel() for v in ix])
    scores = {'peak_frequency': np.broadcast_to(frequency[eligible], truth.shape), 'knn_rna_pca': knn}
    source_only = {**test, 'views': [test['views'][0], np.zeros_like(test['views'][1])],
                   'counts': [test['counts'][0], test['counts'][1] * 0]}
    for name, model in models.items():
        if model.capabilities().get('cross_modal_prediction'):
            predicted = model.predict_cross_modal(source_only, source=0, target=1)
            scores[name] = np.asarray(predicted)[:, eligible]
        else:
            # A separate inductive kNN decoder in the model's RNA-only latent space.
            try:
                a, b = model.encode(train, modality=0), model.encode(source_only, modality=0)
                decoder = NearestNeighbors(n_neighbors=min(30, len(a))).fit(a)
                indices = decoder.kneighbors(b, return_distance=False)
                scores[name + '_knn_decoder'] = np.stack([np.asarray(binary_train[v].mean(0)).ravel() for v in indices])
            except NotImplementedError:
                continue
    return dict(direction='RNA to ATAC', target='binarized train-selected peaks', evaluated_peaks=eligible.tolist(),
                candidate_peak_selection='train frequency 1%-99%, at most 1000 stratified by frequency',
                target_modality_supplied_to_predictor=False,
                models={name: binary_prediction_metrics(truth, prediction) for name, prediction in scores.items()})

