"""Prepare train-only files and invoke the canonical R WNN cross-check."""
import subprocess

import pandas as pd
from scipy import io, sparse

from agent3_replearn.config import load_config
from agent3_replearn.orchestrator import require_smoke
from agent3_replearn.utils import checksum, require_compute, write_json

require_compute()
require_smoke()
cfg = load_config('config/pbmc_multiome.yaml')
prepared = cfg.out / 'preprocess'
destination = cfg.out / 'baseline/r_wnn_crosscheck'
destination.mkdir(parents=True, exist_ok=True)
rna = sparse.load_npz(prepared / 'train/rna_counts.npz')
atac = (sparse.load_npz(prepared / 'train/atac_counts.npz') > 0).astype('float32')
io.mmwrite(destination / 'rna.mtx', rna.T)
io.mmwrite(destination / 'atac.mtx', atac.T)
ids = pd.read_parquet(prepared / 'train/obs.parquet').index
genes = pd.read_csv(prepared / 'selected_genes.csv')['id']
peaks = pd.read_csv(prepared / 'selected_peaks.csv')['name']
for name, values in [('barcodes', ids), ('genes', genes), ('peaks', peaks)]:
    (destination / (name + '.txt')).write_text('\n'.join(map(str, values)) + '\n')
command = ['bash', '-c', 'source /etc/profile.d/modules.sh\nmodule --ignore_cache load r/4.6.0\nRscript --vanilla scripts/wnn_crosscheck.R "$1"',
           'r-wnn-crosscheck', str(destination)]
with (destination / 'R.log').open('w') as log:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
write_json(destination / 'manifest.json', dict(command=command, train_only=True, labels_used=False,
    source_script_sha256=checksum('scripts/wnn_crosscheck.R'),
    input_hashes={str(p): checksum(p) for p in [destination / 'rna.mtx', destination / 'atac.mtx', destination / 'barcodes.txt']},
    output_hashes={str(p): checksum(p) for p in [destination / 'seurat_wsnn.mtx', destination / 'sessionInfo.txt', destination / 'reference_metadata.csv']}))
