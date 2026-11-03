"""Passing real synthetic training and end-to-end evidence generate the gate."""
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

from .config import Config
from .orchestrator import run
from .schemas import ComputeBudget, ObjectiveSpec
from .utils import ROOT, checksum, code_hash, read_json, require_compute, write_json


def smoke():
    require_compute()
    initial = code_hash()
    subprocess.run([sys.executable, '-m', 'ruff', 'check', 'src', 'tests'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '--basetemp=.cache/pytest-smoke'], cwd=ROOT, check=True)
    fixtures = ROOT / 'outputs/smoke/fixtures'
    fixtures.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(774)
    n = 150
    labels = np.asarray(['CD4 Naive', 'CD14 Mono', 'NK'])[np.arange(n) % 3]
    latent = rng.normal(size=(n, 6)).astype(np.float32)
    latent[:, :3] += np.eye(3)[np.arange(n) % 3] * 3
    ids = np.asarray([f'synthetic_{i}' for i in range(n)])
    a = latent + rng.normal(0, .05, latent.shape).astype(np.float32)
    b = latent + rng.normal(0, .05, latent.shape).astype(np.float32)
    # Complete pipeline fixture one: count modalities stored in the real 10x HDF5 layout.
    counts = rng.poisson(np.exp(np.clip(latent @ rng.normal(0, .25, (6, 180)), -1, 3))).astype(np.float32)
    peaks = rng.binomial(1, 1 / (1 + np.exp(-latent @ rng.normal(0, .5, (6, 120))))).astype(np.float32)
    from scipy import sparse
    matrix = sparse.csc_matrix(np.concatenate([counts, peaks], axis=1).T)
    path = fixtures / 'counts.h5'
    with h5py.File(path, 'w') as f:
        group = f.create_group('matrix')
        for name, value in [('data', matrix.data), ('indices', matrix.indices), ('indptr', matrix.indptr), ('shape', matrix.shape)]:
            group.create_dataset(name, data=value)
        group.create_dataset('barcodes', data=ids.astype('S'))
        feature = group.create_group('features')
        feature.create_dataset('id', data=np.asarray([f'feature_{i}' for i in range(300)], dtype='S'))
        feature.create_dataset('name', data=np.asarray([f'feature_{i}' for i in range(300)], dtype='S'))
        feature.create_dataset('feature_type', data=np.asarray(['Gene Expression'] * 180 + ['Peaks'] * 120, dtype='S'))
    label_file = fixtures / 'labels.csv'
    pd.DataFrame({'celltype': labels}, index=ids).to_csv(label_file)
    write_json(fixtures / 'label_provenance.json', dict(recommended_label_column='celltype', recommended_label_file=label_file.name,
                                                       label_origin='Known synthetic latent class'))
    budget = ComputeBudget(pilot_epochs=3, final_epochs=4, pilot_seconds=60, seeds=[774], max_shortlist=1)
    count_config = Config(dataset_id='synthetic_counts', output='outputs/smoke/counts', report_name='smoke_counts',
        smoke=True, modalities=[dict(modality_id='rna', modality_type='count_matrix_gene', path=str(path), format='10x_h5',
                                    pairing_key='barcode', parameters={'feature_type': 'Gene Expression'}),
                               dict(modality_id='atac', modality_type='count_matrix_peak', path=str(path), format='10x_h5',
                                    pairing_key='barcode', parameters={'feature_type': 'Peaks'})],
        labels=dict(path=str(label_file), column='celltype', quality='synthetic', provenance=str(fixtures / 'label_provenance.json')),
        budget=budget, bootstrap_replicates=50, preprocessing=dict(n_hvg=80, n_peaks=80, latent_dim=6))
    count_objective = ObjectiveSpec(objective_id='synthetic_joint', original_text='Synthetic joint count structure and alignment; labels evaluation only',
        objective_types=['cell_state_structure', 'cross_modal_alignment'], target_variable='celltype',
        allowed_supervision='evaluation_only', unit_of_analysis='cell', downstream_tasks=['classification', 'retrieval'],
        primary_metrics=['macro_f1', 'retrieval_recall_at_10'])
    run(count_config, count_objective, resume=False)
    # Complete pipeline fixture two: variable image-bag-like features + continuous table.
    array_path = fixtures / 'image_bags.npz'
    np.savez_compressed(array_path, ids=ids, labels=labels, view_0=a, view_1=b,
                        bags=a[:, None, :] + rng.normal(0, .03, (n, 4, 6)).astype(np.float32))
    image_config = Config(dataset_id='synthetic_image_bags', output='outputs/smoke/image_bags',
        report_name='smoke_image_bags', smoke=True, budget=budget, bootstrap_replicates=50,
        modalities=[dict(modality_id='image_bag', modality_type='dense_image', path=str(array_path), format='paired_npz', pairing_key='sample_id'),
                    dict(modality_id='tabular', modality_type='continuous_tabular', path=str(array_path), format='paired_npz', pairing_key='sample_id')],
        preprocessing=dict(latent_dim=6))
    image_objective = ObjectiveSpec(objective_id='synthetic_supervised', original_text='Predict known synthetic classes; training labels may be used',
        objective_types=['supervised_prediction'], target_variable='known_latent_class', allowed_supervision='train_labels',
        unit_of_analysis='sample', downstream_tasks=['classification', 'retrieval'], primary_metrics=['macro_f1', 'balanced_accuracy'])
    run(image_config, image_objective, resume=False)
    for cfg in [count_config, image_config]:
        rows = read_json(cfg.out / 'baseline/baseline_metrics.json') + read_json(cfg.out / 'pilot/pilot_metrics.json')
        if any(r['status'] != 'success' for r in rows):
            raise AssertionError('Synthetic eligible candidate failure: ' + str([(r['model_id'], r['status']) for r in rows]))
        run(cfg, count_objective if cfg is count_config else image_objective, resume=True)
    if code_hash() != initial:
        raise RuntimeError('Analysis code changed during smoke; rerun the current code')
    evidence = {str(cfg.out.relative_to(ROOT) / 'validation_results.json'): checksum(cfg.out / 'validation_results.json')
                for cfg in [count_config, image_config]}
    write_json(ROOT / 'logs/smoke_acceptance.json', dict(passed=True, analysis_code_hash=initial, evidence=evidence,
               tests='full pytest; two end-to-end pipelines; validated resume; unseen profile and objective-swap regression',
               slurm_required=True))
    gate = read_json(ROOT / 'logs/smoke_acceptance.json')
    gate['slurm_job_id'] = os.environ['SLURM_JOB_ID']
    write_json(ROOT / 'logs/smoke_acceptance.json', gate)
