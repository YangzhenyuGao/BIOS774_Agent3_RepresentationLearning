"""Testable profile/objective rules; dataset_id never participates in a branch."""

from .registry import METHODS
from .schemas import Applicability


def applicability(profile, objective, encoder_evidence=None):
    kinds = {m.modality_type for m in profile.modalities}
    counts = {"count_matrix_gene", "count_matrix_peak"} <= kinds
    wsi = "wsi_image" in kinds
    paired = profile.pairing.n_paired >= 30
    labels_train = objective.allowed_supervision == "train_labels"
    survival = objective.primary_metrics == ["c_index"]
    rows = []
    for m in METHODS:
        status, rule, reason = (
            "eligible_for_pilot",
            "ELIGIBLE_PROFILE_OBJECTIVE",
            "Compatible modalities and objective",
        )
        if m.data_requirement == "counts" and not counts:
            status, rule, reason = (
                "not_applicable",
                "COUNT_LIKELIHOOD_OR_NEIGHBOR_INPUT",
                "Requires paired RNA/ATAC count modalities",
            )
        elif m.data_requirement == "wsi" and not wsi and not (
                m.model_id == 'abmil' and 'dense_image' in kinds and profile.pairing.multiplicity.get('tiles_per_sample', 0) > 0):
            status, rule, reason = "not_applicable", "WSI_ENCODER_REQUIRED", "No whole-slide image modality"
        elif m.data_requirement == "numeric" and wsi:
            status, rule, reason = (
                "not_applicable",
                "RAW_VIEW_FACTOR_MODEL",
                "Gigapixel images require a patch/MIL encoder; direct numeric factor model is not a raw-image encoder",
            )
        elif m.data_requirement == "paired" and not paired:
            status, rule, reason = (
                "infeasible_under_budget",
                "MINIMUM_PAIRED_SAMPLES",
                "Fewer than 30 paired samples for a validation pilot",
            )
        elif m.supervised and not labels_train:
            status, rule, reason = (
                "reference_only",
                "LABELS_EVALUATION_ONLY",
                "Objective does not allow representation fitting on labels",
            )
        if survival and m.supervised and status == "eligible_for_pilot":
            reason = "Labels allowed; replace classification loss with Cox partial likelihood"
        if m.model_id in {"clip_dual", "fm_image_text"} and paired:
            reason += (
                "; small-pair regime: contrastive gains must be established empirically"
                if profile.pairing.n_paired < 2000
                else ""
            )
        missing = max(profile.pairing.missing_modality_rates.values(), default=0)
        if (
            missing > 0.2
            and status == "eligible_for_pilot"
            and m.model_id not in {"multivi", "fusion_gated", "view_a", "view_b"}
        ):
            status, rule, reason = (
                "not_applicable",
                "MISSING_MODALITY_SUPPORT",
                "More than 20% modality missingness; adapter does not support this regime",
            )
        rows.append(
            Applicability(
                model_id=m.model_id,
                family=m.family,
                status=status,
                rule_id=rule,
                reason=reason,
                encoders=list(m.encoders),
                fusion_type=m.fusion,
                training_objective="Cox_partial_likelihood" if survival and m.supervised else m.loss,
                evidence={
                    "modality_types": sorted(kinds),
                    "paired_samples": profile.pairing.n_paired,
                    "allowed_supervision": objective.allowed_supervision,
                    "missing_modality_rate": missing,
                },
            )
        )
    return rows


def preprocessing_plan(profile, objective):
    actions = []
    for modality in profile.modalities:
        kind = modality.modality_type
        if kind == "count_matrix_gene":
            steps = [
                "MAD count/mitochondrial QC",
                "training-only doublet diagnostic",
                "raw counts retained for count likelihood",
                "train HVGs",
                "library normalization/log1p",
                "train PCA",
            ]
        elif kind == "count_matrix_peak":
            steps = [
                "ATAC count/fragment QC",
                "train peak-frequency filter",
                "declared binarization",
                "train IDF weights",
                "train LSI; diagnose depth-correlated component",
            ]
        elif kind == "wsi_image":
            steps = [
                "per-slide metadata and physical resolution",
                "thumbnail tissue segmentation",
                "seeded tissue tile sampling with QC",
                "frozen local encoder features",
                "slide/patient pooling",
            ]
        elif kind == "tabular_clinical":
            steps = [
                "explicit clinical allowlist",
                "category harmonization",
                "train imputation",
                "train vocabulary and scaling",
                "per-variable missingness indicators",
            ]
        else:
            steps = ["finite-value inspection", "train imputation/scaling"]
        for step in steps:
            actions.append(
                dict(
                    action=step,
                    modality_id=modality.modality_id,
                    parameters={},
                    reason=f"profile modality_type={kind}",
                    fit_scope="train"
                    if any(s in step for s in ["train", "MAD", "QC", "imputation"])
                    else "fixed",
                    data_leakage_check="enforced by fitted-ID receipts",
                    rows_and_features_before_after={"before": [modality.n_samples, modality.n_features]},
                )
            )
    return dict(
        actions=actions,
        labels_may_fit=objective.allowed_supervision == "train_labels",
        downstream_tasks=objective.downstream_tasks,
        primary_metrics=objective.primary_metrics,
        stages_inserted=["tile", "extract-features"]
        if any(m.modality_type == "wsi_image" for m in profile.modalities)
        else [],
    )
