import numpy as np

from agent3_replearn.evaluation.uncertainty import bootstrap_secondary


def test_paired_intervals_have_correct_orientation_and_fixed_gallery():
    classes = ['a', 'b', 'c', 'd']
    y = np.repeat(classes, 10)
    target = np.repeat(np.arange(4), 10)
    perfect = np.eye(4)[target] * .9 + .025
    wrong = np.roll(perfect, 1, axis=1)
    predictions = {'perfect': y, 'wrong': np.asarray(classes)[(target + 1) % 4]}
    result = bootstrap_secondary(y, predictions, classes,
        probabilities={'perfect': perfect, 'wrong': wrong},
        query_scores={'perfect': np.ones(40), 'wrong': np.zeros(40)}, replicates=20)
    assert result['balanced_accuracy']['paired_differences']['perfect__minus__wrong']['lower'] == 1
    assert result['ovr_auroc']['per_class_intervals']['perfect']['a']['lower'] == 1
    assert result['retrieval_recall_at_10']['pool_size'] == 40
    assert result['retrieval_recall_at_10']['paired_differences']['perfect__minus__wrong']['lower'] == 1
