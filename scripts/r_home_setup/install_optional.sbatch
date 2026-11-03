#!/usr/bin/env bash
#SBATCH --job-name=r-home-appendix-b
#SBATCH --account=rc_yunmli_pi
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r_home_setup/optional-%j.log
set -euo pipefail
source /etc/profile.d/modules.sh
module --ignore_cache load r/4.6.0
cd /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning
export R_LIBS_USER=/nas/sycamore/home/gaoyang/R/x86_64-pc-linux-gnu-library/4.6
export TMPDIR="$PWD/logs/r_home_setup/tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export MAKEFLAGS=-j1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
hostname
date -Is
Rscript --vanilla scripts/r_home_setup/install_optional.R
