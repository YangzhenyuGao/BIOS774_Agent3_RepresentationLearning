"""Reproducible figures chosen by declared rules, from frozen results."""
import numpy as np
import pandas as pd

from .utils import read_json, write_json


def make_figures(config, objective, plan, train, test, train_z, test_z):
    import matplotlib.pyplot as plt
    import umap
    out = config.out / 'evaluation/figures'
    out.mkdir(parents=True, exist_ok=True)
    for previous in out.iterdir():
        if previous.is_file() and previous.suffix in {'.png', '.json'}:
            previous.unlink()
    evidence = read_json(config.out / 'evaluation/evidence.json')
    # Fit the visualization map on training embeddings and transform held-out samples.
    mapper = umap.UMAP(n_neighbors=15, min_dist=.3, random_state=config.seed, n_jobs=1)
    mapper.fit(train_z)
    coordinates = mapper.transform(test_z)
    labels = np.asarray(test['labels'])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label in sorted(set(labels)):
        keep = labels == label
        ax.scatter(coordinates[keep, 0], coordinates[keep, 1], s=7, alpha=.7, label=label)
    ax.set(title='Held-out representation: ' + plan['selected'], xlabel='UMAP 1', ylabel='UMAP 2')
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'selected_umap.png', dpi=160)
    plt.close(fig)
    rows = evidence['models']
    names = list(rows)
    value = [rows[m]['classification']['macro_f1'] for m in names]
    interval = evidence['bootstrap']['intervals']
    lower = [max(0, v - interval[m]['lower']) for m, v in zip(names, value)]
    upper = [max(0, interval[m]['upper'] - v) for m, v in zip(names, value)]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(names, value, xerr=[lower, upper], color=['#287f8e' if m == plan['selected'] else '#8997a0' for m in names])
    ax.set(xlabel='Held-out macro-F1 (95% paired bootstrap CI)', xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(out / 'model_comparison.png', dpi=160)
    plt.close(fig)
    matrix = np.asarray(rows[plan['selected']]['classification']['confusion_matrix'])
    classes = rows[plan['selected']]['classification']['classes']
    fig, ax = plt.subplots(figsize=(max(4, len(classes) * .35), max(3.5, len(classes) * .32)))
    ax.imshow(matrix / np.maximum(matrix.sum(1, keepdims=True), 1), vmin=0, vmax=1, cmap='Blues')
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel='Predicted', ylabel='Curated/reference label')
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    manifest = dict(figure_selection='all frozen finalists; selected by validation only',
                    umap='fit train, transform test; visualization only', figures=[p.name for p in out.glob('*.png')])
    if 'site' in test['obs']:
        fig, ax = plt.subplots(figsize=(6, 4))
        sites = test['obs'].site.to_numpy()
        for site in sorted(set(sites)):
            keep = sites == site
            ax.scatter(coordinates[keep, 0], coordinates[keep, 1], s=10, alpha=.65)
        ax.set(title='Held-out embedding by tissue source site', xlabel='UMAP 1', ylabel='UMAP 2')
        fig.tight_layout()
        fig.savefig(out / 'site_umap.png', dpi=160)
        plt.close(fig)
        attention_gallery(config, plan, test)
    if 'counts' in test:
        marker_panels(config, test)
        for i, name in enumerate(['rna_pca', 'atac_lsi']):
            projection = umap.UMAP(n_neighbors=15, min_dist=.3, random_state=config.seed, n_jobs=1).fit(train['views'][i])
            xy = projection.transform(test['views'][i])
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for label in sorted(set(labels)):
                mask = labels == label
                ax.scatter(xy[mask, 0], xy[mask, 1], s=7, alpha=.7, label=label)
            ax.set(title='Held-out ' + name.replace('_', ' '), xlabel='UMAP 1', ylabel='UMAP 2')
            ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
            fig.tight_layout()
            fig.savefig(out / f'{name}_umap.png', dpi=160)
            plt.close(fig)
    selected_run = next(r for r in plan['runs'] if r['model_id'] == plan['selected'])
    curves = selected_run.get('training_curves', [])
    if curves:
        keys = ['train_loss', 'validation_loss', 'elbo_train', 'elbo_validation']
        fig, ax = plt.subplots(figsize=(6, 3))
        plotted = False
        for key in keys:
            points = [(i + 1, r[key]) for i, r in enumerate(curves) if key in r]
            if points:
                ax.plot(*np.asarray(points).T, label=key)
                plotted = True
        if plotted:
            ax.set(xlabel='Epoch/recorded iteration', ylabel='Family-specific training objective', title='Selected-model training')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / 'selected_training_curves.png', dpi=160)
        plt.close(fig)
    manifest['figures'] = [p.name for p in out.glob('*.png')]
    write_json(out / 'figure_manifest.json', manifest)


def attention_gallery(config, plan, test):
    import matplotlib.pyplot as plt
    import openslide

    from .preprocessing.wsi import read_tile
    selected = plan['selected']
    directory = config.out / 'representations/attention'
    predictions = pd.read_csv(config.out / 'evaluation' / f'{selected}_predictions.csv')
    if not directory.exists():
        write_json(config.out / 'evaluation/figures/attention_status.json', dict(
            applicable=False, reason='Selected representation has no learned attention; no fabricated attention map'))
        return
    chosen = []
    for label in sorted(predictions.truth.unique()):
        frame = predictions[predictions.truth.eq(label)].copy()
        probability_cols = [c for c in frame if c.startswith('probability_')]
        frame['confidence'] = frame[probability_cols].max(axis=1)
        for correct in [True, False]:
            part = frame[frame.prediction.eq(frame.truth) == correct].sort_values('confidence', ascending=False)
            if len(part):
                chosen.append((part.iloc[0].sample_id, label, correct))
    records = []
    for patient, label, correct in chosen:
        p = directory / f'{selected}_{patient}.parquet'
        if not p.exists():
            continue
        weights = pd.read_parquet(p).sort_values('attention', ascending=False)
        fig, axes = plt.subplots(1, min(4, len(weights)), figsize=(8, 2.5), squeeze=False)
        for ax, row in zip(axes[0], weights.head(4).itertuples()):
            metadata = read_json(config.out / 'features/coordinates' / f'{row.slide_id}.json')
            with openslide.OpenSlide(row.source_path) as slide:
                ax.imshow(read_tile(slide, row.coordinate, metadata))
            ax.set_title(f'Attention {row.attention:.3f}', fontsize=8)
            ax.axis('off')
        fig.suptitle(f'{label}; {"correct" if correct else "incorrect"}; {patient}', fontsize=10)
        fig.tight_layout()
        path = config.out / 'evaluation/figures' / f'attention_{patient}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        # Spatial attention is descriptive, not a validated localization label.
        slide_id = weights.groupby('slide_id').attention.sum().idxmax()
        spatial = weights[weights.slide_id.eq(slide_id)]
        metadata = read_json(config.out / 'features/coordinates' / f'{slide_id}.json')
        with openslide.OpenSlide(spatial.iloc[0].source_path) as slide:
            thumb = slide.get_thumbnail((1024, 1024)).convert('RGB')
            width, height = slide.dimensions
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(thumb)
        maximum = float(spatial.attention.max())
        from matplotlib.patches import Rectangle
        for row in spatial.itertuples():
            x, y = row.coordinate
            scale_x, scale_y = thumb.width / width, thumb.height / height
            ax.add_patch(Rectangle((x * scale_x, y * scale_y),
                metadata['level0_tile_size'] * scale_x, metadata['level0_tile_size'] * scale_y,
                facecolor=plt.cm.inferno(row.attention / max(maximum, 1e-12)), alpha=.55, edgecolor='none'))
        ax.set_title(f'{label}; spatial attention (descriptive)', fontsize=10)
        ax.axis('off')
        fig.tight_layout()
        heatmap = config.out / 'evaluation/figures' / f'attention_heatmap_{patient}.png'
        fig.savefig(heatmap, dpi=140)
        plt.close(fig)
        records.append(dict(patient_id=patient, truth=label, correct=correct, figure=str(path)))
    write_json(config.out / 'evaluation/figures/attention_selection.json', dict(
        rule='highest-confidence correct and incorrect patient per class, then top-four attention tiles', records=records))


def marker_panels(config, test):
    import io
    from pathlib import Path

    import matplotlib.pyplot as plt

    from .sensitivity import coarse
    path = config.preprocessing.get('peak_annotation')
    if not path:
        return
    genes = pd.read_csv(config.out / 'preprocess/selected_genes.csv')
    peaks = pd.read_csv(config.out / 'preprocess/selected_peaks.csv')
    markers = [g for g in ['CD3D', 'MS4A1', 'NKG7', 'LYZ', 'TCL1A', 'CD27', 'PPBP'] if g in set(genes['name'])]
    if not markers:
        return
    annotation = pd.read_csv(io.StringIO(Path(path).read_text().replace('\\t', '\t')), sep='\t')
    peak_names = peaks['name'].str.replace(':', '_', regex=False).str.replace('-', '_', regex=False)
    lookup = {v: i for i, v in enumerate(peak_names)}
    gene_to_peak = {}
    for row in annotation.itertuples():
        if row.peak in lookup:
            for gene in str(row.gene).replace(';', ',').split(','):
                gene_to_peak.setdefault(gene.strip(), []).append(lookup[row.peak])
    rna, activity = [], []
    for marker in markers:
        j = int(np.flatnonzero(genes['name'].eq(marker))[0])
        r = np.asarray(test['counts'][0][:, j].toarray()).ravel()
        rna.append(np.log1p(1e4 * r / np.maximum(test['obs'].rna_umis.to_numpy(), 1)))
        indices = sorted(set(gene_to_peak.get(marker, [])))
        score = np.asarray(test['counts'][1][:, indices].sum(1)).ravel() if indices else np.zeros(len(r))
        activity.append(np.log1p(1e4 * score / np.maximum(test['obs'].atac_counts.to_numpy(), 1)))
    labels = coarse(test['labels'])
    groups = sorted(set(labels) - {'__unlabeled__'})
    matrices = [np.stack([np.mean(np.asarray(values)[:, labels == group], axis=1) for group in groups]) for values in [rna, activity]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, matrix, title in zip(axes, matrices, ['RNA marker expression', 'Peak-linked gene activity proxy']):
        standardized = (matrix - matrix.mean(0)) / np.maximum(matrix.std(0), 1e-6)
        ax.imshow(standardized, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
        ax.set(xticks=np.arange(len(markers)), xticklabels=markers, yticks=np.arange(len(groups)), yticklabels=groups, title=title)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    out = config.out / 'evaluation/figures'
    fig.savefig(out / 'marker_gene_activity.png', dpi=160)
    plt.close(fig)
    write_json(out / 'marker_gene_activity.json', dict(markers=markers, groups=groups,
        activity_definition='Sum selected-peak counts linked to each gene by vendor peak annotation; log-normalized by total peak counts',
        caveat='Approximate gene-activity proxy, not measured expression or independent annotation evidence',
        linked_peaks_per_marker={g: len(set(gene_to_peak.get(g, []))) for g in markers}))
