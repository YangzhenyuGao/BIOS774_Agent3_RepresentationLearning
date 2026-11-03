"""Critical four-page paper; quantitative results come from frozen artifacts."""
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pypdf import PdfReader
from weasyprint import HTML

from .config import load_config
from .reporting import source_claim, validate_evidence
from .utils import ROOT, checksum, code_hash, read_json, write_json

NAMES = {
    'view_a': 'RNA PCA', 'view_b': 'ATAC LSI', 'concatenation': 'Scaled concatenation',
    'wnn': 'Inductive WNN', 'multivi': 'MultiVI', 'mofa': 'Gaussian MOFA+',
    'simclr': 'SimCLR-style', 'clip_dual': 'Paired contrastive', 'abmil': 'Attention MIL',
    'fusion_early': 'Early fusion', 'fusion_late': 'Late fusion', 'fusion_gated': 'Gated fusion',
    'histology_conch': 'Frozen CONCH-Z mean', 'histology_imagenet': 'ImageNet ResNet-18 mean',
    'fm_last_block': 'CONCH-Z last-block tuning', 'fm_image_text': 'CONCH-Z image–text tuning',
}


def architecture_figure(path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(8, 1.22))
    ax.set(xlim=(-.2, 10.2), ylim=(0, 1.5))
    ax.axis('off')
    labels = ['Profile data\n+ parse objective', 'Eligibility rules\n+ preprocessing',
              'Bounded pilots\n+ validation selection', 'Freeze models\n+ held-out evidence',
              'Conditional verdict\n+ source-linked report']
    for x, label in zip([0, 2.1, 4.2, 6.3, 8.4], labels):
        ax.add_patch(FancyBboxPatch((x, .35), 1.6, .85, boxstyle='round,pad=.05',
                                    facecolor='#f5f5f5', edgecolor='#333333', linewidth=.8))
        ax.text(x + .8, .77, label, ha='center', va='center', fontsize=7.5)
        if x:
            ax.add_patch(FancyArrowPatch((x - .43, .77), (x - .08, .77), arrowstyle='->', mutation_scale=10))
    fig.tight_layout(pad=.1)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def audit_context():
    """Read code and receipts; do not load, train or evaluate any model."""
    files = [ROOT / 'src/agent3_replearn' / name for name in
             ['models/multivi_adapter.py', 'models/baselines.py', 'models/wnn.py',
              'models/neural.py', 'models/foundation_finetune.py', 'evaluation/downstream.py',
              'evaluation/cross_modal.py', 'selector.py']]
    logs = [ROOT / f'outputs/pbmc_multiome/pilot/model_runs/multivi/{seed}/attempt_0/worker.log'
            for seed in [774, 775, 776]]
    runs = [row for row in read_json(ROOT / 'outputs/pbmc_multiome/pilot/pilot_metrics.json')
            if row['model_id'] == 'multivi']
    assert len(runs) == 3 and all(row['hyperparameters']['epochs'] == 50 for row in runs)
    assert all('`max_epochs=50` reached' in path.read_text() for path in logs)
    assert all(not row['training_curves'] for row in runs)
    return dict(sources={str(path): checksum(path) for path in [*files, *logs]},
                multivi_max_epochs=50, multivi_kl_warmup_epochs=25, multivi_curve_rows=0,
                custom_validation_callback=True, native_loss_class_weighted=False,
                probe_class_weighted=False, scope='Existing code/log interpretation only; no new experiment')


def build_project_report():
    directory = ROOT / 'reports'
    directory.mkdir(exist_ok=True)
    audit = directory / 'report_audit_context.json'
    write_json(audit, audit_context())
    claims, sources = [], set()

    def value(path, keys, fmt='.3f'):
        record = source_claim(path, keys, fmt)
        claims.append(record)
        sources.add(str(path))
        return record['rendered']

    roots = {name: ROOT / 'outputs' / name for name in ['pbmc_multiome', 'tcga_brca']}
    evidence, verdicts = {}, {}
    for ds in roots:
        config = load_config(ROOT / 'config' / (ds + '.yaml'))
        assert read_json(config.out / 'validation_results.json')['passed']
        evidence[ds], verdicts[ds] = validate_evidence(config)

    def model(ds, name, category, metric):
        return value(roots[ds] / 'evaluation/evidence.json', ['models', name, category, metric])

    def pilot_table(ds, order):
        path = roots[ds] / 'evidence_bundle/pilot_summary.json'
        rows = read_json(path)['rows']
        table = []
        for name in order:
            index = next(i for i, row in enumerate(rows) if row['model'] == name)
            row = rows[index]
            f1 = value(path, ['rows', index, 'validation_macro_f1_mean'])
            if row['validation_macro_f1_sd'] is not None:
                f1 += ' ± ' + value(path, ['rows', index, 'validation_macro_f1_sd'])
            title = NAMES[name]
            if ds == 'tcga_brca':
                title = {'view_a': 'Frozen UNI2-h mean', 'view_b': 'Clinical only',
                         'clip_dual': 'Feature-pair contrastive'}.get(name, title)
            table.append(dict(name=title, selected=name == evidence[ds]['selected'], validation=f1,
                retrieval=value(path, ['rows', index, 'validation_recall_at_10'])
                          if ds == 'pbmc_multiome' and row['validation_recall_at_10'] is not None else '—',
                test=model(ds, name, 'classification', 'macro_f1') if name in evidence[ds]['models'] else '—',
                ari=model(ds, name, 'structure', 'ari') if ds == 'pbmc_multiome' and name in evidence[ds]['models'] else '—',
                nmi=model(ds, name, 'structure', 'nmi') if ds == 'pbmc_multiome' and name in evidence[ds]['models'] else '—'))
        return table

    p, t = 'pbmc_multiome', 'tcga_brca'
    pbmc = pilot_table(p, ['view_a', 'view_b', 'concatenation', 'wnn', 'multivi', 'mofa', 'simclr', 'clip_dual'])
    tcga = pilot_table(t, ['view_a', 'histology_conch', 'histology_imagenet', 'view_b', 'concatenation',
                         'abmil', 'simclr', 'clip_dual', 'fusion_early', 'fusion_gated', 'fusion_late',
                         'fm_last_block', 'fm_image_text'])
    c = {}
    c['pbmc_n'] = value(roots[p] / 'preprocess/preprocessing_manifest.json', ['cohort_flow', 'after'], 'd')
    c['pbmc_split'] = [value(roots[p] / 'preprocess/preprocessing_manifest.json', ['cohort_flow', 'by_split', split], 'd')
                       for split in ['train', 'validation', 'test']]
    for key in ['primary_slides', 'primary_slide_patients']:
        c[key] = value(roots[t] / 'dataset_profile.json', ['cohort_flow', key], 'd')
    for ds, prefix in [(p, 'p'), (t, 't')]:
        for label, field in [('single', 'comparison_to_best_single_modality'), ('concat', 'comparison_to_simple_fusion_baseline')]:
            c[prefix + '_' + label + '_delta'] = [value(roots[ds] / 'verdict.json',
                [field, 'primary_metric_differences', 'macro_f1', bound]) for bound in ['estimate', 'lower', 'upper']]
    for label, name in [('clip', 'clip_dual'), ('rna', 'view_a'), ('concat', 'concatenation')]:
        c[label + '_f1'] = model(p, name, 'classification', 'macro_f1')
        c[label + '_ari'] = model(p, name, 'structure', 'ari')
    c['clip_r10'] = model(p, 'clip_dual', 'alignment', 'retrieval_recall_at_10')
    c['ridge_r10'] = model(p, 'view_a', 'alignment', 'retrieval_recall_at_10')
    c['retrieval_delta'] = [value(roots[p] / 'verdict.json', ['comparison_to_best_single_modality',
                           'primary_metric_differences', 'retrieval_recall_at_10', key])
                           for key in ['estimate', 'lower', 'upper']]
    assert evidence[p]['models']['view_a']['alignment'] == evidence[p]['models']['view_b']['alignment']
    assert evidence[p]['models']['view_a']['alignment'] == evidence[p]['models']['concatenation']['alignment']
    downstream = {ds: read_json(roots[ds] / 'downstream/results.json') for ds in roots}
    prediction = downstream[p]['cross_modal_prediction']['models']
    for name in ['view_a_knn_decoder', 'view_b_knn_decoder', 'concatenation_knn_decoder']:
        assert prediction[name] == prediction['knn_rna_pca']
    cross_rows = []
    for name, title in [('peak_frequency', 'Training peak frequency'), ('knn_rna_pca', 'RNA PCA + kNN decoder'),
                        ('clip_dual_knn_decoder', 'Paired contrastive + RNA decoder')]:
        cross_rows.append(dict(name=title, values=[value(roots[p] / 'downstream/results.json',
            ['cross_modal_prediction', 'models', name, key]) for key in ['per_cell_auroc', 'per_cell_auprc', 'per_peak_auprc']]))
    c['wnn_overlap'] = value(roots[p] / 'downstream/results.json', ['sensitivity', 'r_wnn_crosscheck', 'top15_graph_overlap'])
    for name, label in [('without_receptors', 'without_receptors'), ('site_held_out', 'site_heldout')]:
        c[label] = value(roots[t] / 'downstream/results.json', ['sensitivity', name, 'metrics', 'macro_f1'])
    for name in ['fusion_late', 'abmil', 'view_b', 'concatenation']:
        c[name + '_f1'] = model(t, name, 'classification', 'macro_f1')
    for label, keys, fmt in [
        ('head_f1', ['end_to_end_head', 'macro_f1'], '.3f'),
        ('head_her2_recall', ['end_to_end_head', 'per_class_recall', 'Her2'], '.3f'),
        ('probe_her2_recall', ['classification', 'per_class_recall', 'Her2'], '.3f'),
        ('her2_test_n', ['end_to_end_head', 'per_class_support', 'Her2'], 'd')]:
        c[label] = value(roots[t] / 'evaluation/evidence.json', ['models', 'fusion_late', *keys], fmt)
    c['her2_n'] = value(roots[t] / 'dataset_profile.json', ['pairing', 'label_availability_by_split_candidate', 'Her2'], 'd')
    c['labeled_n'] = value(roots[t] / 'dataset_profile.json', ['cohort_flow', 'labeled_four_class'], 'd')
    for key in ['c_index', 'lower', 'upper', 'test_events']:
        c['survival_' + key] = value(roots[t] / 'downstream/results.json', ['survival', key], 'd' if key == 'test_events' else '.3f')
    for key in ['gpu_hours_used', 'gpu_hours_cap']:
        c[key] = value(ROOT / 'logs/gpu_ledger.json', [key], 'd' if key.endswith('cap') else '.3f')
    c['tile_cap'] = value(roots[t] / 'features/encoder_selection.json', ['tile_cap'], 'd')
    c['projection_256'] = value(roots[t] / 'features/encoder_selection.json', ['projected_hours_by_cap', '256'])
    for key in ['multivi_max_epochs', 'multivi_kl_warmup_epochs']:
        c[key] = value(audit, [key], 'd')
    modality_figure = roots[p] / 'evidence_bundle/report_figures/modality_umap.png'
    assert modality_figure.is_file()
    sources.add(str(modality_figure))
    architecture_figure(directory / 'architecture.png')
    publication_path = ROOT / 'docs/PUBLICATION_LINKS.json'
    publication = read_json(publication_path)
    sources.add(str(publication_path))
    env = Environment(loader=FileSystemLoader(ROOT / 'templates'), autoescape=select_autoescape(['html']))
    swaps = read_json(ROOT / 'outputs/robustness/objective_swap_summary.json')
    for demo in swaps:
        for row in demo['changes']:
            row['display_name'] = NAMES.get(row['model_id'], row['model_id'])
    html = env.get_template('project_report.html.j2').render(c=c, pbmc=pbmc, tcga=tcga, cross_rows=cross_rows,
        architecture=str(directory / 'architecture.png'), modality_figure=str(modality_figure),
        references=read_json(ROOT / 'docs/REFERENCES.json'), objective_swaps=swaps, publication=publication,
        css=(ROOT / 'templates/project_paper.css').read_text())
    (directory / 'report.html').write_text(html)
    path = directory / 'report.pdf'
    HTML(string=html, base_url=str(ROOT)).write_pdf(path)
    reader = PdfReader(path)
    texts = [page.extract_text() for page in reader.pages]
    ref = [i for i, text in enumerate(texts) if text.strip().startswith('References')]
    assert len(ref) == 1 and ref[0] == 4, f'Expected four body pages, got reference starts {ref}'
    appendix = [i for i, text in enumerate(texts) if text.strip().startswith('Appendix A')]
    assert len(appendix) == 1 and appendix[0] > ref[0]
    all_text = '\n'.join(texts)
    expected_links = [publication['repository'], *[study['url'] for study in publication['studies']]]
    first_page_links = [str(action['/URI']) for item in reader.pages[0].get('/Annots', [])
                        if (action := item.get_object().get('/A')) and '/URI' in action]
    assert all(url in first_page_links for url in expected_links), 'Missing clickable GitHub links on first page'
    missing = [claim for claim in claims if claim['rendered'] not in all_text]
    assert not missing, f'Numeric source claims missing: {[(x["keys"], x["rendered"]) for x in missing]}'
    assert not any(term in '\n'.join(texts[:4]) for term in ['clip_dual', 'view_a', 'pbmc_multiome', 'fusion_late'])
    write_json(directory / 'report_claim_sources.json', claims)
    write_json(directory / 'project_report_validation.json', dict(passed=True, body_pages=4,
        reference_pages=appendix[0] - ref[0], appendix_pages=len(texts) - appendix[0], total_pages=len(texts),
        claims_verified=len(claims), sources={name: checksum(name) for name in sorted(sources)},
        references_sha256=checksum(ROOT / 'docs/REFERENCES.json'),
        objective_swap_source_sha256=checksum(ROOT / 'outputs/robustness/objective_swap_summary.json'),
        analysis_code_hash=code_hash(), pdf_sha256=checksum(path), llm_used=False,
        publication_links_verified=expected_links,
        revision_scope='Critical report-only reinterpretation; original models, metrics and machine verdicts unchanged'))
    return path
