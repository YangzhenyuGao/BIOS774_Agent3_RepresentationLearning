#!/usr/bin/env bash
#SBATCH --job-name=agent3-preprocess
#SBATCH --account=rc_yunmli_pi
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess-%j.log
set -euo pipefail
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/scripts/slurm/common.sh
python -m agent3_replearn run --config config/pbmc_multiome.yaml --objective objectives/pbmc_multiome_joint.yaml --through preprocess --resume
python -m agent3_replearn run --config config/tcga_brca.yaml --objective objectives/tcga_brca_pam50.yaml --through preprocess --resume
python scripts/wnn_crosscheck.py
python scripts/plan_demonstrations.py
