"""Worker isolation, baseline/pilot/tuning/final stages and frozen test access."""
import os
import signal
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .selector import select
from .utils import ROOT, checksum, code_hash, digest, provenance, read_json, write_json


def parameters(config, objective, model_id, final=False):
    p = dict(latent_dim=config.preprocessing.get('latent_dim', 32), hidden=128, batch_size=128,
             lr=1e-3, epochs=config.budget.final_epochs if final else config.budget.pilot_epochs,
             seconds=config.budget.pilot_seconds * (2 if final else 1), patience=10)
    if model_id in {'fm_last_block', 'fm_image_text'}:
        p.update(lr=1e-5, batch_size=4, finetune_patients=128,
                 model_store=config.model_store,
                 clinical_columns=config.preprocessing.get('clinical_columns', []))
    if model_id in {'histology_conch', 'histology_imagenet'}:
        p['input_encoder'] = {'histology_conch': 'CONCH-Z', 'histology_imagenet': 'imagenet_resnet18'}[model_id]
    if objective.primary_metrics == ['c_index']:
        p['task'] = 'survival'
    return p


def candidate(config, objective, model_id, directory, seed, params):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    request = dict(config=config.model_dump(), objective=objective.model_dump(), model_id=model_id,
                   directory=str(directory), seed=seed, params=params, analysis_code_hash=code_hash(),
                   preprocessing_hash=checksum(config.out / 'preprocess/preprocessing_manifest.json'))
    key = digest(request)
    result_path = directory / 'result.json'
    if result_path.exists():
        previous = read_json(result_path)
        if previous.get('request_hash') == key and all(
                Path(p).is_file() and checksum(p) == sha for p, sha in previous.get('artifacts', {}).items()):
            return previous
    attempts = []
    for retry in range(2):
        attempt = directory / f'attempt_{retry}'
        attempt.mkdir(exist_ok=True)
        current = {**request, 'directory': str(attempt)}
        if retry:
            current['params'] = {**params, 'batch_size': max(2, params['batch_size'] // 2)}
        write_json(attempt / 'request.json', current)
        with (attempt / 'worker.log').open('w') as log:
            timed_out = False
            process = subprocess.Popen([sys.executable, '-m', 'agent3_replearn.worker', '--request', str(attempt / 'request.json')],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=params['seconds'] + 1800)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
                raise
        if not (attempt / 'result.json').exists():
            row = dict(status='timeout' if timed_out else 'failed', model_id=model_id,
                       seed=seed, error_type='TimeoutError' if timed_out else 'WorkerExit',
                       error_message=f'Worker exit {process.returncode}; inspect worker.log')
        else:
            row = read_json(attempt / 'result.json')
        attempts.append(dict(retry=retry, status=row['status'], params=current['params'], error=row.get('error_message')))
        if row['status'] == 'success' or 'out of memory' not in row.get('error_message', '').lower():
            break
    row.update(request_hash=key, retry_count=len(attempts) - 1, attempts=attempts,
               artifacts={str(p.resolve()): checksum(p) for p in attempt.rglob('*')
                          if p.is_file() and p.name not in {'worker.log', 'result.json'}})
    write_json(result_path, row)
    print(model_id, seed, row['status'], flush=True)
    return row


def save_rows(directory, rows, filename):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    write_json(path, rows)
    table = []
    for row in rows:
        table.append(dict(model_id=row['model_id'], seed=row['seed'], status=row['status'],
                          **row.get('validation', {}), fit_seconds=row.get('runtime_seconds'),
                          error=row.get('error_message', '')))
    # Nested classification details are kept in JSON, never dumped into report tables.
    pd.json_normalize(table).to_csv(path.with_suffix('.csv'), index=False)
    return [path, path.with_suffix('.csv')]


def execute_stage(config, objective, stage):
    out = config.out
    if stage in {'baseline', 'pilot'}:
        applicability = read_json(out / 'applicability_matrix.json')
        baseline_ids = ['view_a', 'view_b', 'concatenation']
        if any(m.modality_type == 'count_matrix_peak' for m in config.modalities):
            baseline_ids.append('wnn')
        elif objective.allowed_supervision == 'train_labels' and any(
                m.modality_type in {'wsi_image', 'dense_image'} for m in config.modalities):
            baseline_ids.append('abmil')
        if any(m.modality_type == 'wsi_image' for m in config.modalities):
            baseline_ids.extend(['histology_conch', 'histology_imagenet'])
        if stage == 'baseline':
            ids = baseline_ids
            rows = []
        else:
            ids = [r['model_id'] for r in applicability if r['status'] == 'eligible_for_pilot' and r['model_id'] not in baseline_ids]
            rows = []
            write_json(out / 'pilot/budget_manifest.json', dict(config_budget=config.budget,
                declared_policy='fit-time wall limit + family epochs + validation stopping; imports excluded',
                candidates=ids, seeds=config.budget.seeds, eligible_candidates_all_attempted=True))
        for model_id in ids:
            seeds = config.budget.seeds if model_id not in {'view_a', 'view_b', 'concatenation', 'wnn', 'histology_conch', 'histology_imagenet'} else [config.seed]
            for seed in seeds:
                rows.append(candidate(config, objective, model_id, out / stage / 'model_runs' / model_id / str(seed),
                                      seed, parameters(config, objective, model_id)))
        paths = save_rows(out / stage, rows, 'baseline_metrics.json' if stage == 'baseline' else 'pilot_metrics.json')
        if stage == 'pilot':
            write_json(out / stage / 'failures.json', [r for r in rows if r['status'] != 'success'])
            paths.append(out / stage / 'failures.json')
        elif any(r['status'] != 'success' for r in rows):
            raise RuntimeError('A required baseline failed; fix before interpreting learned candidates')
        return paths
    if stage == 'select':
        rows = read_json(out / 'baseline/baseline_metrics.json') + read_json(out / 'pilot/pilot_metrics.json')
        decision = select(rows, objective.primary_metrics, config.equivalence_margin, config.budget.max_shortlist)
        write_json(out / 'selection/decisions.json', decision)
        return [out / 'selection/decisions.json']
    if stage == 'tune':
        decision = read_json(out / 'selection/decisions.json')
        results, chosen, best_rows = [], {}, []
        original = read_json(out / 'baseline/baseline_metrics.json') + read_json(out / 'pilot/pilot_metrics.json')
        for mid in decision['shortlist']:
            defaults = [r for r in original if r['model_id'] == mid and r['status'] == 'success']
            default = defaults[0]
            best_params, retained = default['hyperparameters'], defaults
            if mid not in {'view_a', 'view_b', 'concatenation', 'wnn', 'histology_conch', 'histology_imagenet'}:
                p = {**best_params, 'lr': best_params['lr'] / 3}
                if mid == 'mofa':
                    p = {**best_params, 'latent_dim': max(2, best_params['latent_dim'] // 2)}
                alternate = [candidate(config, objective, mid, out / 'tuning/model_runs' / mid / str(seed), seed, p)
                             for seed in config.budget.seeds]
                results.extend(alternate)
                if all(row['status'] == 'success' for row in alternate):
                    def mean_metric(rows, key):
                        return sum(r['validation'][key] for r in rows) / len(rows)
                    gain = mean_metric(alternate, objective.primary_metrics[0]) - mean_metric(defaults, objective.primary_metrics[0])
                    no_loss = all(mean_metric(alternate, key) >= mean_metric(defaults, key) - config.equivalence_margin
                                  for key in objective.primary_metrics[1:])
                    if gain > config.equivalence_margin and no_loss:
                        best_params, retained = p, alternate
            chosen[mid] = best_params
            best_rows.extend(retained)
        tuned_decision = select(best_rows, objective.primary_metrics, config.equivalence_margin, config.budget.max_shortlist)
        write_json(out / 'tuning/decisions_after_tuning.json', tuned_decision)
        write_json(out / 'tuning/chosen_parameters.json', dict(selected=tuned_decision['selected'], parameters=chosen,
                   selection_rule='replicated validation gain beyond margin, no other-primary loss; rerank shortlist', used_test_data=False))
        paths = save_rows(out / 'tuning', results, 'tuning_results.json')
        return paths + [out / 'tuning/chosen_parameters.json', out / 'tuning/decisions_after_tuning.json']
    if stage == 'final':
        tuning = read_json(out / 'tuning/chosen_parameters.json')
        references = ['view_a', 'view_b', 'concatenation']
        if any(m.modality_type == 'count_matrix_peak' for m in config.modalities):
            references.append('wnn')
        elif objective.allowed_supervision == 'train_labels' and any(
                m.modality_type in {'wsi_image', 'dense_image'} for m in config.modalities):
            references.append('abmil')
        if any(m.modality_type == 'wsi_image' for m in config.modalities):
            references.extend(['histology_conch', 'histology_imagenet'])
        ids = list(dict.fromkeys(references + list(tuning['parameters'])))
        runs = []
        for mid in ids:
            base = tuning['parameters'].get(mid, parameters(config, objective, mid, final=True))
            params = {**base, 'epochs': config.budget.final_epochs, 'seconds': config.budget.pilot_seconds * 2}
            deterministic = {'view_a', 'view_b', 'concatenation', 'wnn', 'histology_conch', 'histology_imagenet'}
            seeds = [config.seed] if mid in deterministic else config.budget.seeds
            for seed in seeds:
                runs.append(candidate(config, objective, mid, out / 'final/model_runs' / mid / str(seed), seed, params))
        if any(r['status'] != 'success' for r in runs):
            raise RuntimeError('Final run failed; preserve failure and repair before freezing test access')
        for row in runs:
            write_json(out / 'final/training_curves' / f'{row["model_id"]}_{row["seed"]}.json', row.get('training_curves', []))
        path = out / 'final/final_plan.json'
        plan = dict(selected=tuning['selected'], runs=runs, frozen_before_test=True,
                    config_hash=config.config_hash,
                    split_hash=checksum(out / 'splits/split_manifest.json'),
                    preprocessing_hash=checksum(out / 'preprocess/preprocessing_manifest.json'), **provenance())
        write_json(path, plan)
        selected = next(r for r in runs if r['model_id'] == tuning['selected'])
        write_json(out / 'final/model_card.json', dict(
            model_id=selected['model_id'], family=selected['family'], encoders=selected['encoders'],
            supervision=objective.allowed_supervision, primary_metrics=objective.primary_metrics,
            checkpoint_sha256=selected['checkpoint_sha256'], preprocessing_sha256=selected['preprocessing_sha256'],
            seed=selected['seed'], embedding_dim=selected['embedding_dim'],
            permitted_use='Exploratory analysis within the declared data/objective scope',
            prohibited_uses=objective.prohibited_uses,
            limitations=read_json(out / 'dataset_profile.json')['label_caveats']))
        (out / 'final/frozen.sha256').write_text(checksum(path) + '\n')
        return [path, out / 'final/frozen.sha256', out / 'final/model_card.json']
    from .final import execute_final_stage
    return execute_final_stage(config, objective, stage)
