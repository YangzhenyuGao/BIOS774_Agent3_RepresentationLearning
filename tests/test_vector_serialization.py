import importlib.util
import json

import numpy as np
import pandas as pd

from agent3_replearn.utils import ROOT


def compatibility_module():
    spec = importlib.util.spec_from_file_location('agent3_cli', ROOT / 'scripts/agent3_cli.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mixed_float_vectors_round_trip_without_losing_float64_precision(tmp_path):
    vectors = [np.array([.1, -3.5], dtype=np.float32),
               np.array([1 + 2.**-40, np.pi, 1e-200], dtype=np.float64)]
    frame = pd.DataFrame(dict(sample=['first', 'second'], embedding=vectors))
    original = pd.DataFrame.to_parquet
    target = tmp_path / 'vectors.parquet'
    with compatibility_module().parquet_vector_compatibility():
        frame.to_parquet(target, index=False)
    restored = pd.read_parquet(target)
    assert restored['sample'].tolist() == ['first', 'second']
    for expected, actual in zip(vectors, restored['embedding']):
        np.testing.assert_array_equal(actual, expected)
    assert restored.embedding.iloc[1][0] > 1
    assert pd.DataFrame.to_parquet is original


def test_unrelated_parquet_columns_keep_their_original_types(tmp_path):
    frame = pd.DataFrame(dict(count=np.array([1, 2], dtype=np.int32), label=['a', 'b']))
    with compatibility_module().parquet_vector_compatibility():
        frame.to_parquet(tmp_path / 'ordinary.parquet', index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'ordinary.parquet'), frame)


def test_nested_empty_metadata_survives_as_json_without_dummy_fields(tmp_path):
    receipt = dict(model_id='vision', remote_code_sha256={}, strict_state_dict=True,
                   files=[dict(name='weights', sha256='abc')])
    frame = pd.DataFrame(dict(slide=['real'], embedding=[[1., 2.]], encoder_receipt=[receipt]))
    with compatibility_module().parquet_vector_compatibility():
        frame.to_parquet(tmp_path / 'slides.parquet', index=False)
    restored = pd.read_parquet(tmp_path / 'slides.parquet')
    assert json.loads(restored.encoder_receipt.iloc[0]) == receipt
    np.testing.assert_array_equal(restored.embedding.iloc[0], [1., 2.])
