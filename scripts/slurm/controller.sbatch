#!/usr/bin/env bash
#SBATCH --job-name=agent3-controller
#SBATCH --account=rc_yunmli_pi
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/controller-%j.log
set -euo pipefail
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/scripts/slurm/common.sh
python scripts/workflow_controller.py
