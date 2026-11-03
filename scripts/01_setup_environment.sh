#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/slurm/common.sh"
AGENT3_CONDA=/nas/sycamore/apps/anaconda/2025.12-2/bin/conda
"$AGENT3_CONDA" install -p "$AGENT3_ENV" --override-channels -c conda-forge -y \
  fontconfig pango harfbuzz fonts-conda-ecosystem
bash scripts/export_environment.sh
python -m pip install 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
bash scripts/export_environment.sh
python -m pip install -r requirements-analysis.txt
bash scripts/export_environment.sh
