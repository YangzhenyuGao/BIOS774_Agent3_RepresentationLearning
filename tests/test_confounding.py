import numpy as np
import pandas as pd

from agent3_replearn.evaluation.confounding import confounding


def test_constant_qc_and_single_site_are_explicitly_not_applicable():
    rng = np.random.default_rng(774)
    train, test = rng.normal(size=(40, 3)), rng.normal(size=(20, 3))
    result = confounding(train, test,
        pd.DataFrame({'mito_fraction': np.zeros(40), 'site': ['A'] * 40}),
        pd.DataFrame({'mito_fraction': np.zeros(20), 'site': ['A'] * 20}))
    assert not result['mito_fraction']['applicable']
    assert 'constant' in result['mito_fraction']['reason']
    assert 'test_r_squared' not in result['mito_fraction']
    assert not result['site']['applicable']
    assert 'balanced_accuracy' not in result['site']


def test_variable_qc_retains_predictability_and_site_chance_uses_training_classes():
    rng = np.random.default_rng(775)
    train, test = rng.normal(size=(120, 3)), rng.normal(size=(60, 3))
    result = confounding(train, test,
        pd.DataFrame({'mito_fraction': .3 + .05 * train[:, 0], 'site': np.repeat(['A', 'B', 'C'], 40)}),
        pd.DataFrame({'mito_fraction': .3 + .05 * test[:, 0], 'site': np.repeat(['A', 'B'], 30)}))
    assert result['mito_fraction']['applicable']
    assert result['mito_fraction']['test_r_squared'] > .9
    assert result['site']['applicable']
    assert result['site']['chance_reference'] == 1 / 3


def test_unseen_sites_are_not_given_a_fabricated_score():
    rng = np.random.default_rng(776)
    train, test = rng.normal(size=(40, 2)), rng.normal(size=(20, 2))
    result = confounding(train, test, pd.DataFrame({'site': ['A', 'B'] * 20}), pd.DataFrame({'site': ['C'] * 20}))
    assert not result['site']['applicable']
    assert result['site']['n_unseen_site_test'] == 20
    assert 'balanced_accuracy' not in result['site']
