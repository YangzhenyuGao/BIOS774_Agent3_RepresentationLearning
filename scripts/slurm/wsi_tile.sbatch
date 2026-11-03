#!/usr/bin/env bash
#SBATCH --job-name=agent3-wsi-tile
#SBATCH --output=logs/wsi-tile-%j.log
set -euo pipefail
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/scripts/slurm/common.sh
python -m agent3_replearn run --config config/tcga_brca.yaml --objective objectives/tcga_brca_pam50.yaml --through tile --resume
python scripts/check_wsi_index.py
