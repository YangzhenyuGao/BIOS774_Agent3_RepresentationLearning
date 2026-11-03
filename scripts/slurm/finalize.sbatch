#!/usr/bin/env bash
#SBATCH --job-name=agent3-finalize
#SBATCH --output=logs/finalize-%j.log
set -euo pipefail
source /vast/som/yunligrp/users/ygao/BIOS774Agent/Agent3_RepresentationLearning/scripts/slurm/common.sh
if [[ -f logs/frozen_before_vector_export_fix.json ]]; then
  python scripts/verify_frozen_export.py
fi
for agent3_data in pbmc_multiome tcga_brca; do
  if [[ "$agent3_data" == pbmc_multiome ]]; then
    agent3_objective=objectives/pbmc_multiome_joint.yaml
  else
    agent3_objective=objectives/tcga_brca_pam50.yaml
  fi
  python -m agent3_replearn report --config "config/$agent3_data.yaml" --objective "$agent3_objective" --resume
  python -m agent3_replearn validate --config "config/$agent3_data.yaml" --objective "$agent3_objective" --resume
done
python scripts/build_project_report.py
python scripts/render_report_previews.py
python scripts/package_submission.py
