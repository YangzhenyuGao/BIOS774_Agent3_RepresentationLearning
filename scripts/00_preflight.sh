#!/usr/bin/env bash
set -euo pipefail
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/scripts/slurm/common.sh
hostname
test -n "${SLURM_JOB_ID:-}" || { echo 'Run integrity scanning in a CPU Slurm allocation'; exit 1; }
python scripts/preflight_integrity.py
