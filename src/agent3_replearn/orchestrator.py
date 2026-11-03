"""Stage DAG, content-validated resume, and explicit held-out test access gate."""

import fcntl
import json
from pathlib import Path

from .inspector import inspect_dataset, read_profile
from .objective_parser import interpret
from .planner import applicability, preprocessing_plan
from .splits import make_splits
from .utils import ROOT, checksum, code_hash, digest, provenance, read_json, require_compute, write_json

STAGES = [
    "inspect",
    "interpret-objective",
    "plan",
    "split",
    "preprocess",
    "tile",
    "extract-features",
    "baseline",
    "pilot",
    "select",
    "tune",
    "final",
    "embed",
    "evaluate",
    "downstream",
    "verdict",
    "report",
    "validate",
]
DEPENDENCIES = {
    "inspect": [],
    "interpret-objective": [],
    "plan": ["inspect", "interpret-objective"],
    "split": ["inspect"],
    "preprocess": ["plan", "split"],
    "tile": ["preprocess"],
    "extract-features": ["tile"],
    "baseline": ["preprocess"],
    "pilot": ["baseline", "plan"],
    "select": ["pilot"],
    "tune": ["select"],
    "final": ["tune"],
    "embed": ["final"],
    "evaluate": ["embed"],
    "downstream": ["evaluate"],
    "verdict": ["downstream"],
    "report": ["verdict"],
    "validate": ["report"],
}


def applicable_stages(config):
    if any(m.modality_type == "wsi_image" for m in config.modalities):
        return STAGES
    return [s for s in STAGES if s not in {"tile", "extract-features"}]


def manifest_path(config, stage):
    return config.out / "stages" / f"{stage}.json"


def dependencies(config, stage):
    deps = list(DEPENDENCIES[stage])
    if stage == "baseline" and any(m.modality_type == "wsi_image" for m in config.modalities):
        deps.append("extract-features")
    return deps


def require_smoke():
    gate = ROOT / "logs/smoke_acceptance.json"
    if not gate.exists():
        raise RuntimeError("No automatically generated smoke gate; run scripts/03_smoke_test.sh in Slurm")
    data = read_json(gate)
    if not data.get("passed") or data["analysis_code_hash"] != code_hash():
        raise RuntimeError("Smoke gate is stale or failed for the current analysis code")
    if not all((ROOT / p).is_file() and checksum(ROOT / p) == sha for p, sha in data["evidence"].items()):
        raise RuntimeError("Smoke gate evidence is missing or changed")


def stage_signature(config, objective, stage):
    deps = dependencies(config, stage)
    dep_hashes = {}
    for dep in deps:
        p = manifest_path(config, dep)
        if not p.exists():
            raise RuntimeError(f"{stage} requires validated stage {dep}")
        manifest = read_json(p)
        if manifest["status"] != "complete":
            raise RuntimeError(f"Incomplete dependency {dep}")
        for name, sha in manifest["artifacts"].items():
            if not Path(name).is_file() or checksum(name) != sha:
                raise RuntimeError(f"Changed dependency artifact {name}; rerun {dep}")
        dep_hashes[dep] = checksum(p)
    # Resolve original file contents, not just paths, for acquisition changes.
    sources = {}
    for m in config.modalities:
        paths = [m.path] + [v for v in m.parameters.values() if isinstance(v, str) and Path(v).is_file()]
        sources.update({p: checksum(p) for p in paths if Path(p).is_file()})
    if config.labels.get("path"):
        sources[config.labels["path"]] = checksum(config.labels["path"])
    for mapping in [config.preprocessing, config.labels]:
        for value in mapping.values():
            if isinstance(value, str) and Path(value).is_file():
                sources[value] = checksum(value)
    return digest(
        dict(
            config=config.model_dump(),
            objective=objective.model_dump(),
            code_hash=code_hash(report=stage in {"report", "validate"}),
            dependencies=dep_hashes,
            source_checksums=sources,
        )
    )


def dispatch(config, objective, stage):
    out = config.out
    if stage == "inspect":
        inspect_dataset(config)
        return [out / "dataset_profile.json", out / "sample_index.json"] + list(
            (out / "preprocess").glob("clinical*.json")
        )
    if stage == "interpret-objective":
        write_json(out / "objective_spec.json", objective)
        return [out / "objective_spec.json"]
    if stage == "plan":
        profile = read_profile(config)
        rows = [r.model_dump() for r in applicability(profile, objective)]
        write_json(out / "applicability_matrix.json", rows)
        write_json(out / "preprocessing_plan.json", preprocessing_plan(profile, objective))
        log = out / "decision_log.jsonl"
        with log.open("a") as f:
            for row in rows:
                f.write(
                    json.dumps(
                        dict(
                            stage=stage,
                            rule_id=row["rule_id"],
                            alternatives=row["model_id"],
                            evidence=row["evidence"],
                            reason=row["reason"],
                            status=row["status"],
                            **provenance(),
                        )
                    )
                    + "\n"
                )
        return [out / "applicability_matrix.json", out / "preprocessing_plan.json"]
    if stage == "split":
        index = read_json(out / "sample_index.json")
        manifest = make_splits(index["ids"], index["labels"], config.split_fractions, config.seed)
        manifest.update(
            unit_of_analysis=objective.unit_of_analysis,
            all_eligible_samples_assigned=True,
            created_before_modeling=True,
        )
        write_json(out / "splits/split_manifest.json", manifest)
        return [out / "splits/split_manifest.json"]
    if stage == "preprocess":
        if all(m.format == 'paired_npz' for m in config.modalities):
            from .data.synthetic import preprocess_arrays
            preprocess_arrays(config)
        elif any(m.modality_type == "count_matrix_gene" for m in config.modalities):
            from .preprocessing.counts import preprocess_counts

            preprocess_counts(config)
        else:
            from .data.prepared import preprocess_clinical

            preprocess_clinical(config)
        return [p for p in (out / "preprocess").rglob("*") if p.is_file()]
    if stage in {"tile", "extract-features"}:
        from .preprocessing.wsi import run_wsi_stage

        return run_wsi_stage(config, objective, stage)
    if stage in {
        "baseline",
        "pilot",
        "select",
        "tune",
        "final",
        "embed",
        "evaluate",
        "downstream",
        "verdict",
    }:
        from .execution import execute_stage

        return execute_stage(config, objective, stage)
    if stage == "report":
        from .reporting import render_report

        return render_report(config, objective)
    if stage == "validate":
        from .validation import validate_run

        return validate_run(config, objective)
    raise ValueError(f"Unknown stage: {stage}")


def run_stage(config, objective, stage, resume=False, force=False):
    require_compute()
    objective = interpret(objective)
    if stage not in applicable_stages(config):
        raise ValueError(f"{stage} is inapplicable for this modality profile")
    if stage in STAGES[7:] and not config.smoke:
        require_smoke()
    config.out.mkdir(parents=True, exist_ok=True)
    path = manifest_path(config, stage)
    signature = stage_signature(config, objective, stage)
    if resume and not force and path.exists():
        old = read_json(path)
        if (
            old["signature"] == signature
            and old["status"] == "complete"
            and all(Path(p).is_file() and checksum(p) == sha for p, sha in old["artifacts"].items())
        ):
            print("Validated resume:", stage, flush=True)
            return old
    if force:
        order = applicable_stages(config)
        for later in order[order.index(stage) :]:
            marker = manifest_path(config, later)
            if marker.exists():
                marker.unlink()
    write_json(path, dict(stage=stage, signature=signature, status="running", artifacts={}, **provenance()))
    artifacts = dispatch(config, objective, stage)
    if signature != stage_signature(config, objective, stage):
        raise RuntimeError("Source, configuration or analysis code changed while stage was running")
    result = dict(
        stage=stage,
        signature=signature,
        status="complete",
        artifacts={str(p.resolve()): checksum(p) for p in artifacts},
        **provenance(),
    )
    write_json(path, result)
    print("Completed:", config.dataset_id, stage, flush=True)
    return result


def run(config, objective, resume=False, force_stage=None, through=None):
    config.out.mkdir(parents=True, exist_ok=True)
    with (config.out / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another workflow is active for this dataset/objective output") from exc
        objective = interpret(objective)
        stages = applicable_stages(config)
        if through:
            stages = stages[: stages.index(through) + 1]
        for stage in stages:
            run_stage(config, objective, stage, resume=resume, force=stage == force_stage)
        write_json(
            config.out / "run_manifest.json",
            dict(
                dataset_id=config.dataset_id,
                objective_id=objective.objective_id,
                config_hash=config.config_hash,
                through=stages[-1],
                stages=stages,
                **provenance(),
            ),
        )
