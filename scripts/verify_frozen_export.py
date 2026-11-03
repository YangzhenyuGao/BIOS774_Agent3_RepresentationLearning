"""Check real mixed-vector export and the immutable state around its IO-only fix."""
import argparse
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
from agent3_cli import parquet_vector_compatibility

from agent3_replearn.utils import ROOT, checksum, code_hash, read_json, require_compute, write_json

require_compute()
parser = argparse.ArgumentParser()
parser.add_argument('--qa', action='store_true', help='Write an isolated export test, never production stage files')
args = parser.parse_args()
before = read_json(ROOT / 'logs/frozen_before_vector_export_fix.json')
assert code_hash() == before['analysis_code_hash']
assert checksum(ROOT / 'outputs/tcga_brca/final/final_plan.json') == before['tcga_final_plan_sha256']
assert checksum(ROOT / 'outputs/pbmc_multiome/final/final_plan.json') == before['pbmc_final_plan_sha256']
assert all(checksum(path) == sha for path, sha in before['checkpoints'].items())
directory = ROOT / 'outputs/tcga_brca/representations'
frames = []
sources = {}
plan = read_json(ROOT / 'outputs/tcga_brca/final/final_plan.json')
runs = {}
for row in plan['runs']:
    runs.setdefault(row['model_id'], row)
for filename, fields in before['representation_arrays'].items():
    with np.load(directory / filename) as arrays:
        for key, expected in fields.items():
            value = arrays[key]
            actual = dict(dtype=str(value.dtype), shape=list(value.shape),
                          raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())
            assert actual == expected, (filename, key)
        values = {key: arrays[key].copy() for key in arrays.files}
    sources[filename.removesuffix('.npz')] = values
    name = filename.removesuffix('.npz')
    frames.append(pd.DataFrame(dict(patient_id=values['ids'], split=values['split'],
        model_id=name, embedding=list(values['embedding']), checkpoint_sha256=runs[name]['checkpoint_sha256'],
        preprocessing_sha256=runs[name]['preprocessing_sha256'],
        embedding_definition='single modality' if runs[name]['fusion_type'] == 'none' else 'joint')))
if args.qa:
    target = ROOT / 'outputs/export_qa/patient_embeddings.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    with parquet_vector_compatibility():
        pd.concat(frames, ignore_index=True).to_parquet(target, index=False)
    result_path = target.parent / 'validation.json'
else:
    target = directory / 'patient_embeddings.parquet'
    result_path = ROOT / 'logs/frozen_vector_export_verification.json'
restored = pd.read_parquet(target)
id_column = next(k for k in ['patient_id', 'sample_id', 'index'] if k in restored)
assert set(restored.model_id) == set(sources)
for name, original in sources.items():
    current = restored[restored.model_id.eq(name)]
    np.testing.assert_array_equal(current[id_column].to_numpy(dtype=str), original['ids'])
    np.testing.assert_array_equal(current['split'].to_numpy(dtype=str), original['split'])
    np.testing.assert_array_equal(np.stack(current.embedding), original['embedding'])
    assert current.checkpoint_sha256.eq(runs[name]['checkpoint_sha256']).all()
    assert current.preprocessing_sha256.eq(runs[name]['preprocessing_sha256']).all()
selection = read_json(ROOT / 'outputs/tcga_brca/features/encoder_selection.json')
inventory = pd.read_parquet(ROOT / 'outputs/tcga_brca/preprocess/slide_inventory.parquet')
example = next(iter(sources.values()))
split_lookup = dict(zip(example['ids'], example['split']))
slides = []
for slide in inventory[inventory.primary_eligible].itertuples():
    with h5py.File(ROOT / 'outputs/tcga_brca/features' / selection['primary_encoder'] / (slide.slide_id + '.h5')) as store:
        mean = store['features'][:].mean(0)
    slides.append(dict(slide_id=slide.slide_id, patient_id=slide.patient_id, split=split_lookup[slide.patient_id],
        embedding=mean.tolist(), encoder=selection['primary_encoder'],
        encoder_receipt=selection['records'][selection['primary_encoder']]))
expected_slides = pd.DataFrame(slides)
slide_target = target.with_name('slide_embeddings.parquet')
if args.qa:
    with parquet_vector_compatibility():
        expected_slides.to_parquet(slide_target, index=False)
actual_slides = pd.read_parquet(slide_target)
for column in ['slide_id', 'patient_id', 'split', 'encoder']:
    np.testing.assert_array_equal(actual_slides[column], expected_slides[column])
np.testing.assert_array_equal(np.stack(actual_slides.embedding), np.stack(expected_slides.embedding))
assert [json.loads(v) for v in actual_slides.encoder_receipt] == expected_slides.encoder_receipt.tolist()
write_json(result_path, dict(passed=True, scope='isolated real-vector QA' if args.qa else 'frozen production export',
    model_count=len(sources), rows=len(restored), slides=len(actual_slides), parquet_sha256=checksum(target),
    slide_parquet_sha256=checksum(slide_target), exact_slide_values_and_metadata_match=True,
    exact_numeric_and_identifier_match=True, frozen_models_and_plans_unchanged=True,
    source_fingerprint_sha256=checksum(ROOT / 'logs/frozen_before_vector_export_fix.json'),
    verification_source_sha256=checksum(__file__)))
print('Verified exact vectors, identifiers, splits, frozen plans and 11 checkpoints:', result_path)
