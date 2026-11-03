"""Label-balanced deterministic sample splits and explicit leakage receipts."""

import numpy as np
from sklearn.model_selection import train_test_split

from .utils import digest


def make_splits(ids, labels, fractions=(0.7, 0.15, 0.15), seed=774):
    ids = np.asarray(ids, dtype=str)
    labels = np.asarray(labels, dtype=str)
    if len(ids) != len(set(ids)) or len(ids) != len(labels):
        raise ValueError("Split IDs must be unique, with one label per ID")
    # Split each label stratum independently, including unlabeled patients. This
    # keeps rare strata balanced and never excludes unlabelled samples from splits.
    groups = {"train": [], "validation": [], "test": []}
    rng = np.random.default_rng(seed)
    for label in sorted(set(labels)):
        positions = np.flatnonzero(labels == label)
        positions = positions[rng.permutation(len(positions))]
        n = len(positions)
        ntest = max(1, int(round(n * fractions[2]))) if n >= 3 else 0
        nval = max(1, int(round(n * fractions[1]))) if n >= 3 else 0
        while ntest + nval >= n:
            nval = max(0, nval - 1)
        groups["test"].extend(ids[positions[:ntest]].tolist())
        groups["validation"].extend(ids[positions[ntest : ntest + nval]].tolist())
        groups["train"].extend(ids[positions[ntest + nval :]].tolist())
    groups = {k: sorted(v) for k, v in groups.items()}
    assert_disjoint(groups)
    assert set(ids) == set(sum(groups.values(), []))
    return dict(
        seed=seed,
        fractions=list(fractions),
        stratification="target label; unlabeled is a stratum",
        labels_used_for_balance_only=True,
        groups=groups,
        id_hash=digest(sorted(ids)),
        split_hash=digest(groups),
    )


def assert_disjoint(groups):
    keys = list(groups)
    for i, left in enumerate(keys):
        if len(groups[left]) != len(set(groups[left])):
            raise ValueError("Duplicate split IDs")
        for right in keys[i + 1 :]:
            if set(groups[left]) & set(groups[right]):
                raise ValueError(f"Split leakage between {left} and {right}")


def assert_train_scope(fitted_ids, manifest):
    fitted = set(fitted_ids)
    if not fitted or not fitted <= set(manifest["groups"]["train"]):
        raise ValueError("Fitted transform contains non-training identifiers or has empty scope")
    return dict(
        fit_scope="train",
        fitted_n=len(fitted),
        fitted_ids_hash=digest(sorted(fitted)),
        split_hash=manifest["split_hash"],
        leakage_check=True,
    )


def site_sensitivity_split(ids, sites, labels, seed=774):
    """A fresh site-held-out training/validation/test split; no random-split model reuse."""
    ids, sites, labels = np.asarray(ids), np.asarray(sites), np.asarray(labels)
    unique, counts = np.unique(sites, return_counts=True)
    eligible = unique[counts >= 5]
    if len(eligible) < 3:
        raise ValueError("Too few sites for a site-held-out sensitivity analysis")
    rng = np.random.default_rng(seed)
    order = eligible[rng.permutation(len(eligible))]
    held = []
    target = max(1, int(round(0.2 * len(ids))))
    for site in order:
        held.append(site)
        if np.isin(sites, held).sum() >= target:
            break
    test = np.isin(sites, held)
    remainder = np.flatnonzero(~test)
    # Patient split only among remaining sites for early stopping, with safe fallback.
    labs = labels[remainder]
    values, freq = np.unique(labs, return_counts=True)
    strat = labs if freq.min() >= 2 else None
    tr, va = train_test_split(remainder, test_size=0.18, random_state=seed, stratify=strat)
    groups = dict(train=ids[tr].tolist(), validation=ids[va].tolist(), test=ids[test].tolist())
    assert_disjoint(groups)
    assert not (set(sites[tr]) & set(sites[test]))
    return dict(
        groups=groups,
        held_out_sites=held,
        seed=seed,
        split_hash=digest(groups),
        caveat="Fresh refitting required; wide uncertainty from a limited number of held-out sites",
    )
