import numpy as np
import pandas as pd
import pytest

from agent3_replearn.objective_parser import ClarificationRequired, interpret
from agent3_replearn.planner import applicability
from agent3_replearn.preprocessing.clinical import ClinicalTransform, clinical_text
from agent3_replearn.schemas import DataProfile, ModalityProfile, ObjectiveSpec, PairingProfile
from agent3_replearn.splits import assert_train_scope, make_splits


def objective(supervision="evaluation_only", unit="cell"):
    return ObjectiveSpec(
        objective_id="fixture",
        original_text="fixture",
        objective_types=["cell_state_structure"],
        target_variable="label",
        allowed_supervision=supervision,
        unit_of_analysis=unit,
        downstream_tasks=["classification"],
        primary_metrics=["macro_f1"],
    )


def profile(types):
    return DataProfile(
        dataset_id="unseen_random_name",
        source_checksums={},
        modalities=[
            ModalityProfile(
                modality_id=str(i),
                modality_type=t,
                n_samples=100,
                n_features=100 if "count" in t else 5,
                storage_format="fixture",
            )
            for i, t in enumerate(types)
        ],
        pairing=PairingProfile(pairing_key="id", pairing_level="sample", n_paired=100),
    )


def test_disjoint_all_samples_and_no_fit_leakage():
    ids = [f"id_{i}" for i in range(100)]
    labels = ["a"] * 43 + ["b"] * 43 + ["__unlabeled__"] * 14
    first = make_splits(ids, labels)
    assert first == make_splits(ids, labels)
    assert set(sum(first["groups"].values(), [])) == set(ids)
    for group in first["groups"].values():
        assert any(labels[ids.index(x)] == "__unlabeled__" for x in group)
    with pytest.raises(ValueError, match="non-training"):
        assert_train_scope(first["groups"]["test"], first)


def test_planner_changes_with_supervision_and_modality_not_name():
    counts = profile(["count_matrix_gene", "count_matrix_peak"])
    a = {r.model_id: r for r in applicability(counts, objective())}
    assert a["multivi"].status == "eligible_for_pilot"
    assert a["fusion_gated"].status == "reference_only"
    b = {r.model_id: r for r in applicability(counts, objective("train_labels"))}
    assert b["fusion_gated"].status == "eligible_for_pilot"
    image = profile(["wsi_image", "tabular_clinical"])
    c = {r.model_id: r for r in applicability(image, objective("train_labels", "patient"))}
    assert c["multivi"].status == "not_applicable"
    assert c["fm_image_text"].status == "eligible_for_pilot"
    # Third modality combination produces a distinct valid plan without code changes.
    unseen = profile(["continuous_tabular", "continuous_tabular"])
    d = {r.model_id: r for r in applicability(unseen, objective("none", "sample"))}
    assert d["mofa"].status == "eligible_for_pilot"
    assert d["wnn"].status == "not_applicable"
    counts.dataset_id = "renamed"
    assert [r.model_dump() for r in applicability(counts, objective())] == [
        r.model_dump() for r in a.values()
    ]


def test_ambiguous_text_does_not_guess_supervision():
    with pytest.raises(ClarificationRequired):
        interpret("Learn RNA ATAC cell representations")
    parsed = interpret("Learn RNA ATAC cell representations; labels are evaluation only")
    assert parsed.allowed_supervision == "evaluation_only"


def test_clinical_target_exclusion_missingness_and_train_vocabulary():
    train = pd.DataFrame(
        {
            "age": [40.0, np.nan, 60.0],
            "stage": ["I", "II", None],
            "PAM50Call_RNAseq": ["Basal", "LumA", "LumB"],
        },
        index=["a", "b", "c"],
    )
    with pytest.raises(ValueError, match="allowlist"):
        ClinicalTransform(["age", "PAM50Call_RNAseq"])
    transform = ClinicalTransform(["age", "stage"]).fit(train)
    test = pd.DataFrame({"age": [np.nan], "stage": ["IV"]}, index=["test"])
    encoded = transform.encode(test)
    assert np.isfinite(encoded).all()
    assert encoded[0, -2:].tolist() == [1, 0]
    assert transform.fit_ids == ["a", "b", "c"]
    assert "PAM50" not in clinical_text(train, ["age", "stage"])[0]
