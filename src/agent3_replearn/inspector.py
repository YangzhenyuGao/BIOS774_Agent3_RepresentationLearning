"""Data profiling by storage and modality properties, with source receipts."""

from pathlib import Path

import numpy as np

from .data.multiome import load_multiome
from .data.tcga_brca import INPUTS, build_tables
from .schemas import DataProfile, ModalityProfile, PairingProfile
from .utils import checksum, read_json, require_compute, write_json


def inspect_dataset(config):
    require_compute()
    if all(m.format == 'paired_npz' for m in config.modalities):
        from .data.synthetic import inspect_arrays
        return inspect_arrays(config)
    kinds = {m.modality_type for m in config.modalities}
    source_paths = {Path(m.path) for m in config.modalities if Path(m.path).is_file()}
    modalities, caveats = [], []
    if "wsi_image" in kinds:
        cohort, slides, flow = build_tables(config)
        for m in config.modalities:
            if m.modality_type == "wsi_image":
                source_paths.add(Path(m.parameters["manifest"]))
                modalities.append(
                    ModalityProfile(
                        modality_id=m.modality_id,
                        modality_type=m.modality_type,
                        n_samples=len(cohort),
                        n_features=0,
                        storage_format="pyramidal_svs",
                        wsi={
                            "n_slides": int(slides.primary_eligible.sum()),
                            "metadata_qc": "per-slide tiling stage",
                            "objective_power": None,
                            "mpp": None,
                            "levels": None,
                            "estimated_tissue_area": None,
                            "estimated_tiles": None,
                        },
                        qc_metrics_available=["manifest size"],
                        estimated_memory=0,
                        estimated_preprocessing_cost={
                            "bytes_read_policy": "pyramid/regions only; bounded readers"
                        },
                    )
                )
            else:
                source_paths.add(Path(m.parameters["xena"]))
                missing = {c: float((cohort[c].isna() | cohort[c].eq("missing")).mean()) for c in INPUTS}
                modalities.append(
                    ModalityProfile(
                        modality_id=m.modality_id,
                        modality_type=m.modality_type,
                        n_samples=len(cohort),
                        n_features=len(INPUTS),
                        storage_format="gdc_json+xena_tsv",
                        feature_types=["numeric", "categorical"],
                        missingness_by_feature=missing,
                    )
                )
        pairing = PairingProfile(
            pairing_key="patient_id",
            pairing_level="patient",
            n_paired=len(cohort),
            multiplicity={
                str(k): int(v)
                for k, v in slides.loc[slides.primary_eligible]
                .groupby("patient_id")
                .size()
                .value_counts()
                .items()
            },
            label_availability_by_split_candidate=flow["labeled_classes"],
            group_or_batch_variables=["patient", "slide", "tissue_source_site"],
            missing_modality_rates={"histology": 0, "clinical": 0},
            licensing_and_release_constraints=[
                "TCGA public diagnostic data; no record-level external API use"
            ],
        )
        caveats = [
            "Retrospective clinical data; missing covariates",
            "Histology may encode tissue source site",
            "PAM50 is an RNA-derived target; no RNA-derived input fields allowed",
        ]
        ids, labels = cohort.index.to_numpy(), cohort.label.to_numpy()
    else:
        matrices, obs, features = load_multiome(config)
        if config.labels.get("path"):
            source_paths.add(Path(config.labels["path"]))
        for value in list(config.preprocessing.values()) + list(config.labels.values()):
            if isinstance(value, str) and Path(value).is_file():
                source_paths.add(Path(value))
        for m in config.modalities:
            x = matrices[m.modality_id]
            depth = np.asarray(x.sum(1)).ravel()
            modalities.append(
                ModalityProfile(
                    modality_id=m.modality_id,
                    modality_type=m.modality_type,
                    n_samples=x.shape[0],
                    n_features=x.shape[1],
                    feature_types=[m.parameters["feature_type"]],
                    storage_format=m.format,
                    shape_and_channels=list(x.shape),
                    value_range=[0, float(x.data.max())],
                    dtype=str(x.dtype),
                    sparsity=1 - x.nnz / np.prod(x.shape),
                    library_size_distribution={
                        str(q): float(np.quantile(depth, q)) for q in [0, 0.25, 0.5, 0.75, 1]
                    },
                    binary_or_count_indicators={
                        "nonnegative": bool((x.data >= 0).all()),
                        "integer": bool(np.equal(x.data, np.floor(x.data)).all()),
                    },
                    qc_metrics_available=list(obs.columns),
                    estimated_memory=int(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes),
                )
            )
        pairing = PairingProfile(
            pairing_key="barcode",
            pairing_level="cell",
            n_paired=len(obs),
            multiplicity={"one_to_one": True},
            label_availability_by_split_candidate=obs.label.value_counts().to_dict(),
            group_or_batch_variables=["donor"],
            missing_modality_rates={m.modality_id: 0 for m in config.modalities},
            licensing_and_release_constraints=["CC BY 4.0"],
        )
        caveats = []
        if config.preprocessing.get('donor_count') == 1:
            caveats.append('Single donor: no donor generalization')
        if config.labels.get('quality') == 'curated':
            caveats.append('Curated FixFlip labels derive from WNN clusters; WNN structural advantage; evaluation-only labels')
        elif config.smoke:
            caveats.append('Synthetic fixture labels; not real biomedical evidence')
        flow = dict(cells_before_agent_qc=len(obs), curated_labeled=int(obs.label.ne("filtered").sum()))
        ids, labels = obs.index.to_numpy(), obs.label.to_numpy()
    profile = DataProfile(
        dataset_id=config.dataset_id,
        modalities=modalities,
        pairing=pairing,
        source_checksums={str(p): checksum(p) for p in sorted(source_paths)},
        label_caveats=caveats,
        cohort_flow=flow,
    )
    write_json(config.out / "dataset_profile.json", profile)
    write_json(config.out / "sample_index.json", dict(ids=ids.tolist(), labels=labels.tolist()))
    return profile


def read_profile(config):
    return DataProfile.model_validate(read_json(config.out / "dataset_profile.json"))
