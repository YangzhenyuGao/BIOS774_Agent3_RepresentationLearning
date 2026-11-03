#!/usr/bin/env bash
set -euo pipefail
AGENT3_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AGENT3_ENV=/vast/som/yunligrp/users/ygao/conda/envs/agent3
export AGENT3_ROOT AGENT3_ENV
export PATH="$AGENT3_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$AGENT3_ROOT/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface" HF_MODULES_CACHE="$XDG_CACHE_HOME/hf_modules"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$XDG_CACHE_HOME/torch" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba" PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda_pkgs"
export TMPDIR="$XDG_CACHE_HOME/tmp"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export LD_LIBRARY_PATH="$AGENT3_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$AGENT3_ENV/etc/fonts/fonts.conf" ]]; then
  export FONTCONFIG_FILE="$AGENT3_ENV/etc/fonts/fonts.conf" FONTCONFIG_PATH="$AGENT3_ENV/etc/fonts"
fi
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"
cd "$AGENT3_ROOT"
python -c 'import sys; assert not any("/.local/" in p for p in sys.path), sys.path'
