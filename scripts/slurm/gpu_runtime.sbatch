#!/usr/bin/env bash
#SBATCH --job-name=agent3-runtime
#SBATCH --output=logs/gpu-runtime-%j.log
set -euo pipefail
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/scripts/slurm/common.sh
python scripts/gpu_runtime_check.py
