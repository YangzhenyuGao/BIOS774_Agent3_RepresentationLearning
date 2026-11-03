SHELL := /bin/bash
.ONESHELL:
export PYTHONNOUSERSITE := 1

.PHONY: check smoke workflow environment
check:
	source scripts/slurm/common.sh
	python -m ruff check src tests
	python -m pytest -q tests/test_policy.py tests/test_alignment.py tests/test_wsi.py

smoke:
	sbatch scripts/slurm/smoke.sbatch

workflow:
	bash scripts/submit_workflow.sh

environment:
	bash scripts/export_environment.sh
