import numpy as np

from agent3_replearn.evaluation.alignment import alignment
from agent3_replearn.evaluation.structure import validity


def test_true_and_shuffled_pair_alignment_and_collapse_detection():
    rng = np.random.default_rng(774)
    z = rng.normal(size=(300, 12))
    perfect = alignment(z, z)
    shuffled = alignment(z, z[rng.permutation(len(z))])
    assert perfect["foscttm"] < 0.001 and perfect["a_to_b_recall_at_1"] == 1
    assert 0.4 < shuffled["foscttm"] < 0.6
    assert shuffled["a_to_b_recall_at_1"] < 0.05
    assert not validity(np.zeros((50, 8)))["valid"]
    assert validity(z)["valid"]
