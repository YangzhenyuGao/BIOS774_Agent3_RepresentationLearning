#!/usr/bin/env bash
#SBATCH --job-name=agent3-smoke
#SBATCH --account=rc_yunmli_pi
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke-%j.log
set -euo pipefail
bash /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/scripts/03_smoke_test.sh
