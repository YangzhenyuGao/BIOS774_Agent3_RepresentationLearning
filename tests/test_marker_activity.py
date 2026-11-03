from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent3_replearn.figures import marker_panels
from agent3_replearn.utils import ROOT, read_json


@pytest.mark.parametrize('separator', ['\t', '\\t'])
def test_vendor_peak_links_accept_tabs_or_escaped_tabs_without_changing_inputs(separator):
    directory = ROOT / '.cache' / ('marker_fixture_escaped' if separator == '\\t' else 'marker_fixture_tabs')
    (directory / 'preprocess').mkdir(parents=True, exist_ok=True)
    (directory / 'evaluation/figures').mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': ['gene1', 'gene2'], 'name': ['CD3D', 'MS4A1']}).to_csv(directory / 'preprocess/selected_genes.csv', index=False)
    pd.DataFrame({'id': ['chr1:1-2', 'chr1:3-4'], 'name': ['chr1:1-2', 'chr1:3-4']}).to_csv(directory / 'preprocess/selected_peaks.csv', index=False)
    annotation = directory / 'peak_annotation.tsv'
    source = '\n'.join([separator.join(['peak', 'gene', 'distance', 'peak_type']),
                        separator.join(['chr1_1_2', 'CD3D', '0', 'promoter']),
                        separator.join(['chr1_3_4', 'MS4A1', '0', 'promoter'])]) + '\n'
    annotation.write_text(source)
    counts = sparse.csr_matrix(np.asarray([[9, 1], [8, 2], [1, 9], [2, 8]], dtype=np.float32))
    data = dict(counts=[counts, counts], labels=np.asarray(['CD4 Naive', 'CD4 Naive', 'Naive B', 'Naive B']),
                obs=pd.DataFrame({'rna_umis': [10] * 4, 'atac_counts': [10] * 4}))
    marker_panels(SimpleNamespace(out=directory, preprocessing={'peak_annotation': str(annotation)}), data)
    result = read_json(directory / 'evaluation/figures/marker_gene_activity.json')
    assert result['linked_peaks_per_marker'] == {'CD3D': 1, 'MS4A1': 1}
    assert annotation.read_text() == source
