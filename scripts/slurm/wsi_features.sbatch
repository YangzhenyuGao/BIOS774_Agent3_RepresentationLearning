#!/usr/bin/env bash
#SBATCH --job-name=agent3-wsi-features
#SBATCH --output=logs/wsi-features-%A_%a.log
set -euo pipefail
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/scripts/slurm/common.sh
python - <<'PY'
import os
import shutil
import subprocess
import sys
from agent3_replearn.config import load_config
from agent3_replearn.preprocessing.wsi import feature_stage
from agent3_replearn.utils import read_json
config = load_config('config/tcga_brca.yaml')
shard = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
nshards = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
try:
    feature_stage(config, shard, nshards)
except RuntimeError:
    receipt = config.out / 'features' / f'extraction_shard_{shard}.json'
    failures = read_json(receipt)['failures']
    if not failures:
        raise
    archive = config.out / 'features/qc_diagnosis'
    archive.mkdir(exist_ok=True)
    shutil.copy2(receipt, archive / f'failed_shard_{shard}_{os.environ["SLURM_JOB_ID"]}.json')
    # Only the independently verified uniform edge-artifact repair is allowed.
    # Other failures still stop the pipeline; region QC and the tile cap stay fixed.
    subprocess.run([sys.executable, 'scripts/repair_wsi_coordinates.py', '--failures', str(receipt)], check=True)
    feature_stage(config, shard, nshards)
PY
