#!/usr/bin/env bash
#SBATCH --job-name=agent3-run
#SBATCH --output=logs/run-%j.log
set -euo pipefail
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/scripts/slurm/common.sh
python scripts/agent3_cli.py run --config "$1" --objective "$2" --through "${3:-verdict}" --resume
