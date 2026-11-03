"""Reports rendered only from validated saved evidence, with per-number source maps."""

from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pypdf import PdfReader
from weasyprint import HTML

from .utils import ROOT, checksum, read_json, write_json


def json_field(value, keys):
    for key in keys:
        value = value[key]
    return value


def source_claim(path, keys, formatting='.3f'):
    data = read_json(path)
    value = json_field(data, keys)
    return dict(source=str(path), source_sha256=checksum(path), keys=keys, value=value,
                rendered=format(value, formatting) if isinstance(value, (int, float)) else str(value))


def validate_evidence(config):
    out = config.out
    evidence = read_json(out / 'evaluation/evidence.json')
    verdict = read_json(out / 'verdict.json')
    from .verdict import compute_verdict
    expected = compute_verdict(evidence, config.equivalence_margin).model_dump(mode='json')
    if expected != verdict:
        raise ValueError('Saved verdict disagrees with computed evidence')
    final = out / 'final/final_plan.json'
    if checksum(final) != (out / 'final/frozen.sha256').read_text().strip():
        raise ValueError('Final-plan freeze receipt mismatch')
    if not all(r['validity']['valid'] for r in evidence['models'].values()):
        raise ValueError('Report evidence contains invalid selected/final representations')
    return evidence, verdict


def descriptive_tables(evidence, downstream):
    """Summarize saved results without fitting, selecting or changing analysis evidence."""
    tables = []

    def add(title, columns, rows):
        if rows:
            tables.append(dict(title=title, columns=columns, rows=rows))

    add('Cross-modal alignment in the fixed held-out candidate pool',
        ['Model', 'A→B recall@1', 'A→B recall@5', 'A→B recall@10', 'B→A recall@10', 'Pool'],
        [[name, r['alignment']['a_to_b_recall_at_1'], r['alignment']['a_to_b_recall_at_5'],
          r['alignment']['a_to_b_recall_at_10'], r['alignment']['b_to_a_recall_at_10'],
          r['alignment']['candidate_pool_size']]
         for name, r in evidence['models'].items() if 'a_to_b_recall_at_1' in r['alignment']])
    add('Held-out masking with the frozen probe', ['Model', 'Masked modality index', 'Macro-F1', 'Balanced accuracy'],
        [[name, str(modality), row['macro_f1'], row['balanced_accuracy']]
         for name, masks in evidence.get('ablations', {}).items() for modality, row in masks.items()])
    sensitivity = downstream.get('sensitivity', {})
    seed = sensitivity.get('seed_stability', {})
    add('Selected representation across the predeclared seeds', ['Seed', 'Validation macro-F1', 'Test macro-F1', 'Test balanced accuracy'],
        [[row['seed'], row['validation_macro_f1'], row['test_macro_f1'], row['test_balanced_accuracy']]
         for row in seed.get('rows', [])])
    stability = [(evidence['selected'], sensitivity.get('subsample_stability', {}).get('rows', []))]
    stability += list(sensitivity.get('baseline_subsample_stability', {}).get('models', {}).items())
    add('Training-subsample stability, conditional on fixed preprocessing',
        ['Model', 'Seed', 'Training fraction', 'Validation neighbor overlap'],
        [[name, row['seed'], row['training_fraction'], row['validation_neighbor_overlap']]
         for name, rows in stability for row in rows])
    add('Refitted sensitivity analyses', ['Analysis', 'Macro-F1', 'Balanced accuracy'],
        [[name.replace('_', ' '), row.get('metrics', row)['macro_f1'], row.get('metrics', row)['balanced_accuracy']]
         for name, row in sensitivity.items()
         if name in {'coarse_labels', 'without_receptors', 'site_held_out'} and 'macro_f1' in row.get('metrics', row)])
    luminal = []
    for name, row in evidence['models'].items():
        classification = row['classification']
        classes = classification['classes']
        if not {'LumA', 'LumB'}.issubset(classes):
            continue
        ix = [classes.index(v) for v in ['LumA', 'LumB']]
        matrix = np.asarray(classification['confusion_matrix'])[ix]
        scores = []
        for local, column in enumerate(ix):
            tp = matrix[local, column]
            denominator = matrix[:, column].sum() + matrix[local].sum()
            scores.append(float(2 * tp / denominator) if denominator else 0.)
        luminal.append([name, int(matrix.sum()), float(np.mean(scores))])
    add('LumA/LumB diagnostic from the frozen four-class confusion matrices',
        ['Model', 'Luminal test patients', 'Luminal macro-F1'], luminal)
    retrieval = downstream.get('retrieval', {})
    if retrieval:
        add('Patient/cell neighbors in the joint representation', ['Candidate pool', 'Label agreement@5', 'Label agreement@10'],
            [[retrieval['candidate_pool_size'], retrieval['subtype_precision_at_5'], retrieval['subtype_precision_at_10']]])
    reference = sensitivity.get('r_wnn_crosscheck', {})
    if reference:
        add('Canonical R WNN cross-check on training cells', ['Training cells', 'Top-15 graph overlap'],
            [[reference['training_cells'], reference['top15_graph_overlap']]])
    return tables


def count_visualizations(config, plan):
    """Visualize frozen outputs; this never refits representations or changes evidence."""
    import pandas as pd

    from .utils import digest

    names = [('rna_umis', True, 'RNA UMIs (log1p)'),
             ('atac_fragments', True, 'ATAC fragments (log1p)'),
             ('tss_fragment_fraction', False, 'TSS fragment fraction proxy'),
             ('mito_fraction', False, 'RNA mitochondrial fraction')]
    obs_paths = [config.out / 'preprocess' / split / 'obs.parquet' for split in ['train', 'test']]
    obs = [pd.read_parquet(path) for path in obs_paths]
    if not all(name in obs[1] for name, _, _ in names):
        return [], []
    selected = plan['selected']
    source = config.out / 'representations' / (selected + '.npz')
    run = next(r for r in plan['runs'] if r['model_id'] == selected)
    sources = {str(path): checksum(path) for path in [source, *obs_paths]}
    if selected == 'clip_dual':
        for split in ['train', 'test']:
            for name in ['rna_pca.npy', 'atac_lsi.npy']:
                path = config.out / 'preprocess' / split / name
                sources[str(path)] = checksum(path)
    sources[run['checkpoint_path']] = checksum(run['checkpoint_path'])
    assert sources[run['checkpoint_path']] == run['checkpoint_sha256']
    signature = digest(dict(sources=sources, recipe=1, seed=config.seed))
    directory = config.out / 'evidence_bundle/report_figures'
    receipt = directory / 'visualization_manifest.json'
    if receipt.exists():
        old = read_json(receipt)
        if old['signature'] == signature and all(Path(p).is_file() and checksum(p) == sha
                                                for p, sha in old['outputs'].items()):
            paths = [Path(p) for p in old['outputs']]
            return [p for p in paths if p.suffix == '.png'], [*paths, receipt]
    import matplotlib.pyplot as plt
    import umap

    directory.mkdir(parents=True, exist_ok=True)
    with np.load(source) as saved:
        train = saved['embedding'][saved['split'] == 'train']
        test = saved['embedding'][saved['split'] == 'test']
        assert np.array_equal(saved['ids'][saved['split'] == 'train'], obs[0].index.astype(str))
        assert np.array_equal(saved['ids'][saved['split'] == 'test'], obs[1].index.astype(str))
    mapper = umap.UMAP(n_neighbors=15, min_dist=.3, random_state=config.seed, n_jobs=1).fit(train)
    coordinates = mapper.transform(test)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, (name, log, label) in zip(axes.flat, names):
        values = obs[1][name].to_numpy(dtype=float)
        colors = np.log1p(values) if log else values
        points = ax.scatter(*coordinates.T, c=colors, s=5, cmap='viridis', alpha=.8)
        ax.set(title=label, xlabel='UMAP 1', ylabel='UMAP 2')
        fig.colorbar(points, ax=ax, shrink=.8)
    fig.tight_layout()
    qc = directory / 'selected_qc_umap.png'
    fig.savefig(qc, dpi=160)
    plt.close(fig)
    arrays = directory / 'display_coordinates.npz'
    values = dict(test_ids=obs[1].index.to_numpy(dtype=str), joint_test_umap=coordinates)
    images = [qc]
    if selected == 'clip_dual':
        from .final import model_from_run
        model = model_from_run(run)
        views = []
        for split, frame in zip(['train', 'test'], obs):
            base = config.out / 'preprocess' / split
            data = dict(ids=frame.index.to_numpy(dtype=str),
                        views=[np.load(base / 'rna_pca.npy'), np.load(base / 'atac_lsi.npy')])
            views.append([model.encode(data, modality=i) for i in range(2)])
        common = umap.UMAP(n_neighbors=15, min_dist=.3, random_state=config.seed, n_jobs=1).fit(np.concatenate(views[0]))
        mapped = common.transform(np.concatenate(views[1]))
        fig, ax = plt.subplots(figsize=(6, 4))
        for index, label in enumerate(['RNA tower', 'ATAC tower']):
            xy = mapped[index * len(test):(index + 1) * len(test)]
            ax.scatter(*xy.T, s=5, alpha=.45, label=label)
        ax.set(title='Frozen paired towers by modality', xlabel='UMAP 1', ylabel='UMAP 2')
        ax.legend(frameon=False)
        fig.tight_layout()
        image = directory / 'modality_umap.png'
        fig.savefig(image, dpi=160)
        plt.close(fig)
        images.append(image)
        values['modality_test_umap'] = mapped
    np.savez_compressed(arrays, **values)
    outputs = [*images, arrays]
    write_json(receipt, dict(signature=signature, sources=sources, outputs={str(p): checksum(p) for p in outputs},
               scope='Frozen checkpoint/embeddings only; UMAP fitted on training embeddings and transforms held-out embeddings. Display only; no model selection or metric change.'))
    return images, [*outputs, receipt]


def site_visualization(config, plan):
    """Label site colors from training frequencies; never change frozen evidence."""
    import pandas as pd

    from .utils import digest

    source = config.out / 'representations' / (plan['selected'] + '.npz')
    obs_paths = [config.out / 'preprocess' / split / 'obs.parquet' for split in ['train', 'test']]
    obs = [pd.read_parquet(path) for path in obs_paths]
    sources = {str(path): checksum(path) for path in [source, *obs_paths]}
    signature = digest(dict(sources=sources, recipe=1, seed=config.seed))
    directory = config.out / 'evidence_bundle/report_figures'
    receipt = directory / 'site_visualization_manifest.json'
    if receipt.exists():
        old = read_json(receipt)
        if old['signature'] == signature and all(Path(p).is_file() and checksum(p) == sha
                                                for p, sha in old['outputs'].items()):
            paths = [Path(p) for p in old['outputs']]
            return next(p for p in paths if p.suffix == '.png'), [*paths, receipt]
    import matplotlib.pyplot as plt
    import umap

    directory.mkdir(parents=True, exist_ok=True)
    with np.load(source) as saved:
        train = saved['embedding'][saved['split'] == 'train']
        test = saved['embedding'][saved['split'] == 'test']
        for split, frame in zip(['train', 'test'], obs):
            assert np.array_equal(saved['ids'][saved['split'] == split], frame.index.astype(str))
    mapper = umap.UMAP(n_neighbors=15, min_dist=.3, random_state=config.seed, n_jobs=1).fit(train)
    coordinates = mapper.transform(test)
    frequencies = obs[0]['site'].astype(str).value_counts()
    top = sorted(frequencies.index, key=lambda site: (-frequencies[site], site))[:5]
    sites = obs[1]['site'].astype(str).to_numpy()
    grouped = np.asarray([site if site in top else ('Other training sites' if site in frequencies.index
                                                  else 'Unseen in training') for site in sites])
    labels = top + ['Other training sites', 'Unseen in training']
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#929292', '#222222']
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for label, color in zip(labels, colors):
        keep = grouped == label
        if keep.any():
            ax.scatter(*coordinates[keep].T, s=14, alpha=.8, color=color, label=label)
    ax.set(title='Held-out embedding by tissue source site', xlabel='UMAP 1', ylabel='UMAP 2')
    ax.legend(title='Site (top five by training count)', fontsize=7, title_fontsize=7,
              bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    fig.tight_layout()
    image = directory / 'site_umap.png'
    fig.savefig(image, dpi=160)
    plt.close(fig)
    arrays = directory / 'site_display_coordinates.npz'
    np.savez_compressed(arrays, test_ids=obs[1].index.to_numpy(dtype=str), joint_test_umap=coordinates,
                        site=sites, display_group=grouped)
    paths = [image, arrays]
    write_json(receipt, dict(signature=signature, sources=sources, top_training_sites=top,
               outputs={str(path): checksum(path) for path in paths},
               scope='Display only: UMAP fits frozen training embeddings, transforms test; site groups chosen by training frequencies. No model, metric or selection change.'))
    return image, [*paths, receipt]


def render_report(config, objective):
    evidence, verdict = validate_evidence(config)
    out = config.out
    profile = read_json(out / 'dataset_profile.json')
    preprocessing = read_json(out / 'preprocess/preprocessing_manifest.json')
    decisions = read_json(out / 'selection/decisions.json')
    applicability = read_json(out / 'applicability_matrix.json')
    final = read_json(out / 'final/final_plan.json')
    alignment_definitions = {}
    for row in final['runs']:
        if row.get('alignment_definition'):
            definition = row['alignment_definition']
            if row['model_id'] == 'fusion_late':
                definition = 'raw late-fusion tower coordinates; independent classifiers and no embedding-alignment loss'
            alignment_definitions.setdefault(definition, set()).add(row['model_id'])
    alignment_definitions = {key: sorted(value) for key, value in alignment_definitions.items()}
    downstream = read_json(out / 'downstream/results.json')
    run_indices = {}
    for index, row in enumerate(final['runs']):
        run_indices.setdefault(row['model_id'], index)
    display_encoders = verdict['encoders']
    if all(m['modality_type'].startswith('count_matrix') for m in profile['modalities']):
        if verdict['selected_representation'] in {'clip_dual', 'simclr'}:
            display_encoders = ['RNA MLP head (PCA input)', 'ATAC MLP head (TF-IDF/LSI input)']
        else:
            display_encoders = [name.replace('MLP_or_MIL', 'MLP heads on PCA/LSI scores') for name in display_encoders]
    elif any(m['modality_type'] == 'wsi_image' for m in profile['modalities']) and verdict['selected_representation'] in {
            'simclr', 'clip_dual', 'fusion_early', 'fusion_late', 'fusion_gated', 'abmil'}:
        display_encoders = [display_encoders[0].replace(' with patient pooling', ' tile encoder'),
                            'gated attention MIL image branch']
        if verdict['selected_representation'] != 'abmil':
            display_encoders.append('clinical MLP with train-fitted tabular preprocessing')
    claims, comparison_rows = [], []
    for name, row in evidence['models'].items():
        f1 = source_claim(out / 'evaluation/evidence.json', ['models', name, 'classification', 'macro_f1'])
        ba = source_claim(out / 'evaluation/evidence.json', ['models', name, 'classification', 'balanced_accuracy'])
        claims.extend([f1, ba])
        dimension = source_claim(out / 'final/final_plan.json', ['runs', run_indices[name], 'embedding_dim'], 'd')
        claims.append(dimension)
        ci = evidence['bootstrap']['intervals'][name]
        comparison_rows.append(dict(model=name, family=row['family'], dimension=dimension['rendered'], macro_f1=f1['rendered'],
                                    balanced_accuracy=ba['rendered'], ci=f"{ci['lower']:.3f}–{ci['upper']:.3f}"))
        claims.extend([source_claim(out / 'evaluation/evidence.json', ['bootstrap', 'intervals', name, key]) for key in ['lower', 'upper']])
    bundle = out / 'evidence_bundle'
    bundle.mkdir(exist_ok=True)
    for index, modality in enumerate(profile['modalities']):
        for key in ['n_samples', 'n_features']:
            claims.append(source_claim(out / 'dataset_profile.json', ['modalities', index, key], 'd'))
    for key, value in profile['cohort_flow'].items():
        if type(value) in {int, float}:
            claims.append(source_claim(out / 'dataset_profile.json', ['cohort_flow', key], 'd' if type(value) is int else ''))
    preprocessing_source = out / 'preprocess/preprocessing_manifest.json'
    for key, value in preprocessing.get('cohort_flow', {}).items():
        if type(value) is int:
            claims.append(source_claim(preprocessing_source, ['cohort_flow', key], 'd'))
        elif key == 'by_split':
            for split in value:
                claims.append(source_claim(preprocessing_source, ['cohort_flow', key, split], 'd'))
    for variable, bounds in preprocessing.get('thresholds', {}).items():
        for bound in ['lower', 'upper']:
            if bounds.get(bound) is not None:
                claims.append(source_claim(preprocessing_source, ['thresholds', variable, bound]))
    for index, modality in enumerate(preprocessing.get('modalities', [])):
        claims.append(source_claim(preprocessing_source, ['modalities', index, 'retained_features'], 'd'))
        if 'first_lsi_depth_correlation' in modality:
            claims.append(source_claim(preprocessing_source, ['modalities', index, 'first_lsi_depth_correlation']))
    for key in ['threshold', 'predicted_training_doublet_fraction']:
        if preprocessing.get('doublet', {}).get(key) is not None:
            claims.append(source_claim(preprocessing_source, ['doublet', key]))
    wsi, wsi_sources = None, []
    encoder_selection = out / 'features/encoder_selection.json'
    if encoder_selection.exists():
        tiling_path = out / 'features/tiling_manifest.json'
        selection, tiling = read_json(encoder_selection), read_json(tiling_path)
        coordinate_example = bundle / 'wsi_coordinate_example.json'
        write_json(coordinate_example, read_json(tiling['coordinate_manifests'][0]))
        wsi = dict(selection=selection, tiling=tiling, example=read_json(coordinate_example),
                   projected_hours=selection['projected_hours_by_cap'][str(selection['tile_cap'])])
        wsi_sources = [encoder_selection, tiling_path, coordinate_example]
        repairs = []
        for filename in tiling['coordinate_manifests']:
            metadata = read_json(filename)
            if metadata.get('coordinate_repair'):
                record = dict(slide_id=metadata['slide_id'], metadata_path=filename,
                    metadata_sha256=checksum(filename), receipt=metadata['coordinate_repair'])
                raw_check = out / 'features/qc_diagnosis' / metadata['slide_id'] / 'raw_integrity.json'
                if raw_check.is_file():
                    record['raw_integrity'] = read_json(raw_check)
                    assert record['raw_integrity']['passed']
                    wsi_sources.append(raw_check)
                repairs.append(record)
                wsi_sources.append(Path(filename))
        repair_summary = bundle / 'wsi_coordinate_repairs.json'
        write_json(repair_summary, dict(count=len(repairs), records=repairs))
        wsi['coordinate_repairs'] = read_json(repair_summary)
        wsi_sources.append(repair_summary)
        if repairs:
            claims.append(source_claim(repair_summary, ['count'], 'd'))
        for key in ['slides_total', 'slides_success']:
            claims.append(source_claim(tiling_path, [key], 'd'))
        for key in ['tile_cap', 'measured_slides', 'safety_factor']:
            claims.append(source_claim(encoder_selection, [key], 'd'))
        claims.append(source_claim(encoder_selection, ['projected_hours_by_cap', str(selection['tile_cap'])]))
        claims.append(source_claim(encoder_selection, ['extraction_hours_envelope']))
        for name in selection['throughput_tiles_per_second']:
            claims.append(source_claim(encoder_selection, ['throughput_tiles_per_second', name]))
        claims.append(source_claim(coordinate_example, ['tile_size'], 'd'))
        claims.append(source_claim(coordinate_example, ['target_mpp']))
        if 'operational_constraints' in selection:
            for key in ['original_budget_only_tile_cap', 'array_parallel', 'startup_seconds_per_task']:
                claims.append(source_claim(encoder_selection, ['operational_constraints', key], 'd'))
            claims.append(source_claim(encoder_selection, ['operational_constraints', 'wall_hours_per_task']))
        measurement = out / 'features/throughput_measurement.json'
        if measurement.exists():
            wsi_sources.append(measurement)
    compute_gpu_hours = None
    ledger_reconciled_utc = None
    if (ROOT / 'logs/gpu_ledger.json').exists():
        ledger = read_json(ROOT / 'logs/gpu_ledger.json')
        assert ledger['gpu_hours_used'] <= ledger['gpu_hours_cap']
        write_json(bundle / 'compute_ledger_snapshot.json', ledger)
        compute_gpu_hours = ledger['gpu_hours_used']
        ledger_reconciled_utc = ledger.get('reconciled_utc')
        claims.append(source_claim(bundle / 'compute_ledger_snapshot.json', ['gpu_hours_used']))
    pilot = read_json(out / 'baseline/baseline_metrics.json') + read_json(out / 'pilot/pilot_metrics.json')
    pilot_summary = []
    for name in sorted({r['model_id'] for r in pilot}):
        attempts = [r for r in pilot if r['model_id'] == name]
        successful = [r for r in attempts if r['status'] == 'success']
        values = [r['validation']['macro_f1'] for r in successful]
        retrieval = [r['validation']['retrieval_recall_at_10'] for r in successful if 'retrieval_recall_at_10' in r['validation']]
        pilot_summary.append(dict(model=name, successful=len(successful), attempted=len(attempts),
            validation_macro_f1_mean=float(np.mean(values)) if values else None,
            validation_macro_f1_sd=float(np.std(values)) if len(values) > 1 else None,
            validation_recall_at_10=float(np.mean(retrieval)) if retrieval else None,
            fit_seconds=float(np.mean([r['runtime_seconds'] for r in successful])) if successful else None,
            failure='; '.join(r.get('error_message', '')[:200] for r in attempts if r['status'] != 'success')))
    write_json(bundle / 'pilot_summary.json', dict(rows=pilot_summary,
        derived_from={str(out / p): checksum(out / p) for p in ['baseline/baseline_metrics.json', 'pilot/pilot_metrics.json']},
        formulas='arithmetic mean and population SD across successful declared seeds; failures remain separate'))
    for i, row in enumerate(pilot_summary):
        for field in ['successful', 'attempted']:
            claims.append(source_claim(bundle / 'pilot_summary.json', ['rows', i, field], 'd'))
        for field in ['validation_macro_f1_mean', 'validation_macro_f1_sd', 'validation_recall_at_10']:
            if row[field] is not None:
                claims.append(source_claim(bundle / 'pilot_summary.json', ['rows', i, field]))
    details = descriptive_tables(evidence, downstream)
    write_json(bundle / 'descriptive_tables.json', dict(tables=details,
        sources={str(out / p): checksum(out / p) for p in ['evaluation/evidence.json', 'downstream/results.json']},
        interpretation='Saved metrics or arithmetic summaries of frozen confusion matrices; no new fit or selection'))
    for table_index, table in enumerate(details):
        for row_index, row in enumerate(table['rows']):
            for column_index, value in enumerate(row):
                if isinstance(value, (int, float)):
                    claims.append(source_claim(bundle / 'descriptive_tables.json',
                        ['tables', table_index, 'rows', row_index, column_index], 'd' if isinstance(value, int) else '.3f'))
    for metric in verdict['comparison_to_best_single_modality'].get('primary_metric_differences', {}):
        for side in ['comparison_to_best_single_modality', 'comparison_to_simple_fusion_baseline']:
            for field in ['estimate', 'lower', 'upper']:
                if verdict[side]['primary_metric_differences'][metric][field] is not None:
                    claims.append(source_claim(out / 'verdict.json', [side, 'primary_metric_differences', metric, field]))
    selected_details = evidence['models'][evidence['selected']]
    if selected_details.get('end_to_end_head'):
        for key in ['macro_f1', 'balanced_accuracy']:
            claims.append(source_claim(out / 'evaluation/evidence.json', ['models', evidence['selected'], 'end_to_end_head', key]))
    for index, run in enumerate(final['runs']):
        for key, formatting in [('seed', 'd'), ('runtime_seconds', '.1f'), ('epochs_or_steps_completed', 'd')]:
            claims.append(source_claim(out / 'final/final_plan.json', ['runs', index, key], formatting))
    for name, row in evidence['models'].items():
        for section, key, formatting in [('validity', 'effective_rank', '.2f'), ('structure', 'ari', '.3f'),
                                         ('structure', 'nmi', '.3f'), ('alignment', 'foscttm', '.3f'),
                                         ('alignment', 'candidate_pool_size', 'd')]:
            if key in row[section] and (section != 'alignment' or 'foscttm' in row[section]):
                claims.append(source_claim(out / 'evaluation/evidence.json', ['models', name, section, key], formatting))
    for name, row in selected_details['confounding'].items():
        if row.get('applicable', True):
            for key in ['test_r_squared', 'balanced_accuracy', 'chance_reference']:
                if key in row:
                    claims.append(source_claim(out / 'evaluation/evidence.json', ['models', evidence['selected'], 'confounding', name, key]))
    for name, row in downstream.get('cross_modal_prediction', {}).get('models', {}).items():
        for key in ['per_cell_auroc', 'per_cell_auprc', 'per_peak_auprc']:
            if row.get(key) is not None:
                claims.append(source_claim(out / 'downstream/results.json', ['cross_modal_prediction', 'models', name, key]))
    if downstream.get('survival', {}).get('status') == 'success':
        for key in ['training_events', 'test_events']:
            claims.append(source_claim(out / 'downstream/results.json', ['survival', key], 'd'))
    if downstream.get('survival', {}).get('status') == 'success':
        for key in ['c_index', 'lower', 'upper']:
            if downstream['survival'].get(key) is not None:
                claims.append(source_claim(out / 'downstream/results.json', ['survival', key]))
    for label in selected_details['classification']['per_class_recall']:
        claims.append(source_claim(out / 'evaluation/evidence.json', ['models', evidence['selected'], 'classification', 'per_class_recall', label]))
        claims.append(source_claim(out / 'evaluation/evidence.json', ['models', evidence['selected'], 'classification', 'per_class_support', label], 'd'))
        if label in selected_details['structure']['per_class_graph_connectivity']:
            claims.append(source_claim(out / 'evaluation/evidence.json', ['models', evidence['selected'], 'structure', 'per_class_graph_connectivity', label]))
        for field in ['per_class_auroc']:
            if selected_details['classification'].get(field, {}).get(label) is not None:
                claims.append(source_claim(out / 'evaluation/evidence.json', ['models', evidence['selected'], 'classification', field, label]))
        for metric in ['balanced_accuracy', 'ovr_auroc']:
            ci = evidence['bootstrap'].get('secondary', {}).get(metric, {}).get('per_class_intervals', {}).get(evidence['selected'], {}).get(label)
            if ci:
                for key in ['lower', 'upper']:
                    if ci[key] is not None:
                        claims.append(source_claim(out / 'evaluation/evidence.json', ['bootstrap', 'secondary', metric, 'per_class_intervals', evidence['selected'], label, key]))
    source_files = [out / p for p in ['dataset_profile.json', 'objective_spec.json', 'applicability_matrix.json',
                    'preprocessing_plan.json', 'selection/decisions.json', 'final/final_plan.json',
                    'evaluation/evidence.json', 'downstream/results.json', 'verdict.json']]
    source_files += [preprocessing_source, *wsi_sources]
    write_json(bundle / 'manifest.json', dict(validated=True, sources={str(p): checksum(p) for p in source_files},
                                            numbers='saved artifact fields only', llm_used=False))
    write_json(bundle / 'claim_sources.json', claims)
    figures = sorted((out / 'evaluation/figures').glob('*.png'))
    report_figure_artifacts = []
    if (out / 'preprocess/train/rna_pca.npy').is_file():
        supplemental, report_figure_artifacts = count_visualizations(config, final)
        figures.extend(supplemental)
    survival_figure = out / 'downstream/survival/survival_km.png'
    if survival_figure.exists():
        figures.append(survival_figure)
    main_figures = [p for name in ['model_comparison.png', 'selected_umap.png', 'confusion_matrix.png', 'site_umap.png']
                    if (p := out / 'evaluation/figures' / name).exists()]
    if any(path.name == 'site_umap.png' for path in main_figures):
        site_figure, site_artifacts = site_visualization(config, final)
        main_figures = [site_figure if path.name == 'site_umap.png' else path for path in main_figures]
        figures = [site_figure if path.name == 'site_umap.png' else path for path in figures]
        report_figure_artifacts.extend(site_artifacts)
    env = Environment(loader=FileSystemLoader(ROOT / 'templates'), autoescape=select_autoescape(['html']))
    template = env.get_template('generated_report.html.j2')
    context = dict(config=config, objective=objective, evidence=evidence, verdict=verdict,
                   profile=profile, preprocessing=preprocessing, wsi=wsi, decisions=decisions,
                   applicability=applicability, final=final, downstream=downstream,
                   comparison_rows=comparison_rows, pilot_rows=pilot_summary, selected_details=selected_details,
                   descriptive_tables=details,
                   alignment_definitions=alignment_definitions,
                   display_encoders=display_encoders,
                   compute_gpu_hours=compute_gpu_hours, ledger_reconciled_utc=ledger_reconciled_utc, figures=main_figures,
                   extra_figures=[p for p in figures if p not in main_figures],
                   css=(ROOT / 'templates/report.css').read_text(),
                   provenance=read_json(out / 'representations/provenance.json'))
    html = template.render(**context)
    html_path = bundle / 'report.html'
    html_path.write_text(html)
    destination = ROOT / 'reports' / (config.report_name + '.pdf')
    destination.parent.mkdir(exist_ok=True)
    HTML(string=html, base_url=str(ROOT)).write_pdf(destination)
    reader = PdfReader(destination)
    text = '\n'.join(p.extract_text() for p in reader.pages)
    missing = [claim['rendered'] for claim in claims if claim['rendered'] not in text]
    if missing:
        raise ValueError('PDF claim-to-source verification failed: ' + str(missing))
    write_json(bundle / 'pdf_validation.json', dict(passed=True, pages=len(reader.pages),
                pdf_sha256=checksum(destination), claims_verified=len(claims), text_characters=len(text)))
    paths = [destination, html_path, bundle / 'manifest.json', bundle / 'claim_sources.json', bundle / 'pdf_validation.json', bundle / 'pilot_summary.json', bundle / 'descriptive_tables.json']
    if compute_gpu_hours is not None:
        paths.append(bundle / 'compute_ledger_snapshot.json')
    if wsi is not None:
        paths.append(bundle / 'wsi_coordinate_example.json')
        paths.append(bundle / 'wsi_coordinate_repairs.json')
    return paths + report_figure_artifacts

