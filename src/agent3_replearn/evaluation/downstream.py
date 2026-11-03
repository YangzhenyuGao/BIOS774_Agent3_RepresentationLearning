"""Identical frozen-embedding probes, classification metrics and paired bootstrap."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UNLABELED = {"filtered", "__unlabeled__", "nan", ""}


def labeled(labels):
    return np.asarray([v not in UNLABELED for v in labels])


def classification_metrics(y, prediction, probabilities=None, classes=None):
    y, prediction = np.asarray(y), np.asarray(prediction)
    classes = list(classes) if classes is not None else sorted(set(y))
    recall = recall_score(y, prediction, labels=classes, average=None, zero_division=0)
    output = dict(
        macro_f1=float(f1_score(y, prediction, labels=classes, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        accuracy=float(accuracy_score(y, prediction)),
        worst_class_recall=float(recall.min()),
        per_class_recall=dict(zip(classes, recall.tolist())),
                  per_class_support={c: int(np.sum(y == c)) for c in classes},
        confusion_matrix=confusion_matrix(y, prediction, labels=classes).tolist(),
        classes=classes,
    )
    if probabilities is not None:
        aucs = {}
        for j, c in enumerate(classes):
            target = y == c
            aucs[c] = (
                float(roc_auc_score(target, probabilities[:, j])) if len(np.unique(target)) == 2 else None
            )
        output["per_class_auroc"] = aucs
        values = [v for v in aucs.values() if v is not None]
        output["ovr_auroc"] = float(np.mean(values)) if values else None
    return output


def fit_probes(train_z, train_y, validation_z, validation_y):
    tr, va = labeled(train_y), labeled(validation_y)
    train_y, validation_y = np.asarray(train_y), np.asarray(validation_y)
    if len(set(train_y[tr])) < 2:
        raise ValueError("At least two labeled training classes required for probes")
    rows, fitted = [], {}
    for family, settings in [("logistic", [0.1, 1.0, 10.0]), ("knn", [5, 15, 30])]:
        best = -1
        for value in settings:
            estimator = (
                LogisticRegression(C=value, max_iter=2000, random_state=774)
                if family == "logistic"
                else KNeighborsClassifier(n_neighbors=min(int(value), int(tr.sum())))
            )
            probe = make_pipeline(StandardScaler(), estimator)
            probe.fit(train_z[tr], train_y[tr])
            prediction = probe.predict(validation_z[va])
            score = float(f1_score(validation_y[va], prediction, average="macro", zero_division=0))
            rows.append(dict(probe=family, parameter=value, validation_macro_f1=score))
            if score > best + 1e-12:
                best, fitted[family] = score, probe
    return fitted, rows


def score_probe(probe, z, labels):
    keep = labeled(labels)
    y = np.asarray(labels)[keep]
    pred = probe.predict(z[keep])
    probability = probe.predict_proba(z[keep])
    metrics = classification_metrics(y, pred, probability, probe.classes_)
    return metrics, dict(mask=keep, y=y, prediction=pred, probabilities=probability, classes=probe.classes_)


def bootstrap(y, predictions, classes, replicates=500, seed=774):
    """Paired index resamples reused across every model and difference."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    estimates = {name: [] for name in predictions}
    missing_class_resamples = 0
    for _ in range(replicates):
        ix = rng.integers(0, len(y), len(y))
        missing_class_resamples += len(set(y[ix])) < len(classes)
        for name, prediction in predictions.items():
            estimates[name].append(
                f1_score(y[ix], np.asarray(prediction)[ix], labels=classes, average="macro", zero_division=0)
            )
    intervals = {
        name: dict(
            estimate=float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
            lower=float(np.quantile(estimates[name], 0.025)),
            upper=float(np.quantile(estimates[name], 0.975)),
        )
        for name, pred in predictions.items()
    }
    differences = {}
    for left in predictions:
        for right in predictions:
            if left == right:
                continue
            delta = np.asarray(estimates[left]) - estimates[right]
            differences[f"{left}__minus__{right}"] = dict(
                estimate=intervals[left]["estimate"] - intervals[right]["estimate"],
                lower=float(np.quantile(delta, 0.025)),
                upper=float(np.quantile(delta, 0.975)),
            )
    return dict(
        metric="macro_f1",
        resampling_unit="sample; patients for histology, cells for multiome",
        replicates=replicates,
        seed=seed,
        intervals=intervals,
        paired_differences=differences,
        missing_class_resamples=missing_class_resamples,
    )
