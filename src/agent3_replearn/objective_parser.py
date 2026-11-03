"""Structured objectives and a conservative deterministic text interpreter."""

from pathlib import Path

import yaml

from .schemas import ObjectiveSpec
from .utils import digest


class ClarificationRequired(ValueError):
    pass


def interpret(value):
    if isinstance(value, ObjectiveSpec):
        return value
    if isinstance(value, Path):
        text = value.read_text()
        value = yaml.safe_load(text) if value.suffix in {".yaml", ".yml", ".json"} else text
    if isinstance(value, dict):
        result = ObjectiveSpec.model_validate(value)
        if result.clarification_needed:
            raise ClarificationRequired("; ".join(result.ambiguities))
        return result
    if not isinstance(value, str):
        raise TypeError("Objective must be a mapping, file or supported plain-language string")
    text = value.lower()
    if "evaluation only" in text or "evaluation-only" in text or "not for fitting" in text:
        supervision = "evaluation_only"
    elif "training labels may be used" in text or "labels may train" in text:
        supervision = "train_labels"
    elif "no labels" in text or "unlabeled" in text:
        supervision = "none"
    else:
        raise ClarificationRequired("May labels train the representation, or are they evaluation-only?")
    survival = "survival" in text and "primary" in text
    pam50 = "pam50" in text
    cell = "cell" in text or ("rna" in text and "atac" in text)
    if survival:
        types, tasks, metrics, target = (
            ["survival_risk"],
            ["survival", "retrieval"],
            ["c_index"],
            "overall_survival",
        )
    elif pam50:
        types, tasks, metrics, target = (
            ["supervised_prediction", "patient_similarity"],
            ["classification", "retrieval", "survival"],
            ["macro_f1", "balanced_accuracy"],
            "PAM50Call_RNAseq",
        )
    elif cell:
        types, tasks, metrics, target = (
            ["cell_state_structure", "cross_modal_alignment", "cross_modal_prediction"],
            ["classification", "clustering", "retrieval", "cross_modal_prediction"],
            ["macro_f1", "retrieval_recall_at_10"],
            "celltype",
        )
    else:
        raise ClarificationRequired(
            "Specify cell structure, supervised prediction or survival as the objective"
        )
    return ObjectiveSpec(
        objective_id="text_" + digest(value)[:12],
        original_text=value,
        objective_types=types,
        downstream_tasks=tasks,
        primary_metrics=metrics,
        target_variable=target,
        allowed_supervision=supervision,
        unit_of_analysis="cell" if cell else "patient",
        parser_mode="deterministic_text",
        parser_confidence=0.9,
    )
