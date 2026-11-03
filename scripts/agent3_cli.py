"""Production CLI with a lossless Parquet-vector compatibility adapter.

Model/preprocessing/selection code stays unchanged. Floating embedding vectors
use Python lists for Arrow inference; encoder-receipt dictionaries use canonical
JSON so empty nested dictionaries survive. Both conversions have exact checks.
"""
import json
from collections import Counter
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

from agent3_replearn.utils import ROOT, checksum, code_hash, write_json


@contextmanager
def parquet_vector_compatibility():
    original = pd.DataFrame.to_parquet

    @wraps(original)
    def compatible(frame, path=None, *args, **kwargs):
        vectors = 'embedding' in frame and any(isinstance(v, np.ndarray) for v in frame['embedding'])
        metadata = 'encoder_receipt' in frame and any(isinstance(v, dict) for v in frame['encoder_receipt'])
        if not vectors and not metadata:
            return original(frame, path, *args, **kwargs)
        converted = frame.copy(deep=False)
        arrays, receipts, columns = [], [], []
        if vectors:
            arrays = [np.asarray(v) for v in frame['embedding']]
            if not all(v.ndim == 1 and v.dtype.kind == 'f' for v in arrays):
                raise TypeError('Embedding vectors must be one-dimensional floating arrays')
            converted['embedding'] = [v.tolist() for v in arrays]
            # Float32 is exactly representable as Python float/Arrow double.
            # Preserve float64 precision rather than narrowing to float32.
            assert all(np.array_equal(a, np.asarray(b), equal_nan=True)
                       for a, b in zip(arrays, converted['embedding']))
            columns.append('embedding')
        if metadata:
            receipts = list(frame['encoder_receipt'])
            if not all(isinstance(v, dict) or v is None for v in receipts):
                raise TypeError('Encoder receipts must be dictionaries or null')
            converted['encoder_receipt'] = [json.dumps(v, sort_keys=True, allow_nan=False) if v is not None else None
                                            for v in receipts]
            columns.append('encoder_receipt')
        result = original(converted, path, *args, **kwargs)
        if isinstance(path, (str, Path)):
            target = Path(path)
            restored = pd.read_parquet(target, columns=columns)
            assert len(restored) == len(frame)
            if vectors:
                assert all(np.array_equal(a, np.asarray(b), equal_nan=True)
                           for a, b in zip(arrays, restored['embedding']))
            if metadata:
                assert [json.loads(v) if v is not None else None for v in restored.encoder_receipt] == receipts
            if target.resolve().is_relative_to(ROOT.resolve()):
                suffix = '.vector_serialization.json' if vectors else '.metadata_serialization.json'
                write_json(target.with_suffix(target.suffix + suffix), dict(
                    component='lossless Parquet export serialization', serialized_columns=columns,
                    serializer_source_sha256=checksum(Path(__file__)), analysis_code_hash=code_hash(),
                    pandas=pd.__version__, pyarrow=pyarrow.__version__, rows=len(frame),
                    original_row_dtypes=dict(Counter(str(v.dtype) for v in arrays)),
                    output_vector_type='list<double>' if vectors else None,
                    metadata_representation='canonical JSON string' if metadata else None,
                    exact_numeric_round_trip=True if vectors else None,
                    exact_metadata_round_trip=True if metadata else None,
                    output_path=str(target), output_sha256=checksum(target),
                    serializer_modifies_model_or_analysis_state=False))
        return result

    pd.DataFrame.to_parquet = compatible
    try:
        yield
    finally:
        pd.DataFrame.to_parquet = original


def main():
    from agent3_replearn.cli import main as cli_main

    with parquet_vector_compatibility():
        cli_main()


if __name__ == '__main__':
    main()
