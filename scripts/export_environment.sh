#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/slurm/common.sh"
AGENT3_CONDA=/nas/sycamore/apps/anaconda/2025.12-2/bin/conda
"$AGENT3_CONDA" env export -p "$AGENT3_ENV" > environment.yml
"$AGENT3_CONDA" list -p "$AGENT3_ENV" --explicit > environment.conda-explicit.txt
python -m pip freeze > environment.lock.txt
python -m pip check
