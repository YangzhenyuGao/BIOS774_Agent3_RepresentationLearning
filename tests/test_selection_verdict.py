from agent3_replearn.selector import select


def row(name, f1, alignment, parameters):
    return dict(model_id=name, status='success', validity={'valid': True}, seed=774,
                validation={'macro_f1': f1, 'retrieval_recall_at_10': alignment}, runtime_seconds=.1,
                trainable_parameter_count=parameters, embedding_dim=8, family='fixture')


def test_alignment_matters_after_structure_equivalence_and_timing_noise_does_not():
    rows = [row('rna', .90, .10, 0), row('joint', .895, .80, 100)]
    decision = select(rows, ['macro_f1', 'retrieval_recall_at_10'], margin=.01)
    assert decision['selected'] == 'joint'
    rows[1]['validation']['macro_f1'] = .85
    assert select(rows, ['macro_f1', 'retrieval_recall_at_10'], margin=.01)['selected'] == 'rna'
    a, b = row('a', .9, .5, 0), row('b', .9, .5, 0)
    a['runtime_seconds'], b['runtime_seconds'] = .2, .1
    first = select([a, b], ['macro_f1'], margin=.01)['selected']
    a['runtime_seconds'], b['runtime_seconds'] = .1, .2
    assert select([a, b], ['macro_f1'], margin=.01)['selected'] == first
