#!/usr/bin/env bash
#SBATCH --job-name=agent3-wsi-throughput
#SBATCH --output=logs/throughput-%j.log
set -euo pipefail
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/scripts/slurm/common.sh
python - <<'PY'
from agent3_replearn.config import load_config
from agent3_replearn.preprocessing.wsi import throughput_pilot
throughput_pilot(load_config('config/tcga_brca.yaml'))
PY
python scripts/adjust_wsi_resources.py
