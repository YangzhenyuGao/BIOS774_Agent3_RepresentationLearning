"""Validated inputs and evidence contracts; unknown fields fail explicitly."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ComputeBudget(StrictModel):
    gpu_hours_cap: float = Field(default=48, gt=0, le=48)
    concurrent_gpus: int = Field(default=2, ge=0, le=2)
    pilot_seconds: int = Field(default=600, gt=0)
    pilot_epochs: int = Field(default=50, gt=0)
    final_epochs: int = Field(default=150, gt=0)
    seeds: list[int] = Field(default_factory=lambda: [774, 775, 776])
    max_shortlist: int = Field(default=2, ge=1, le=2)


class ObjectiveSpec(StrictModel):
    objective_id: str
    original_text: str
    objective_types: list[str]
    target_variable: str | None = None
    label_source_and_quality: dict[str, Any] = Field(default_factory=lambda: {"quality": "none"})
    allowed_supervision: Literal["none", "evaluation_only", "train_labels"]
    modalities_required: list[str] = Field(default_factory=list)
    unit_of_analysis: Literal["cell", "slide", "patient", "sample"]
    downstream_tasks: list[str]
    primary_metrics: list[str]
    secondary_metrics: list[str] = Field(default_factory=list)
    inapplicable_metrics: dict[str, str] = Field(default_factory=dict)
    clinical_inputs_and_exclusions: dict[str, Any] = Field(default_factory=dict)
    sensitivity_analyses: list[str] = Field(default_factory=list)
    compute_budget: ComputeBudget = Field(default_factory=ComputeBudget)
    prohibited_uses: list[str] = Field(default_factory=list)
    parser_mode: Literal["structured", "deterministic_text", "llm"] = "structured"
    parser_confidence: float = Field(default=1, ge=0, le=1)
    ambiguities: list[str] = Field(default_factory=list)
    clarification_needed: bool = False

    @model_validator(mode="after")
    def supervision_consistency(self):
        if self.allowed_supervision == "train_labels" and not self.target_variable:
            raise ValueError("Train-label objective requires an explicit target")
        if not self.primary_metrics or not self.downstream_tasks:
            raise ValueError("Objective requires declared metrics and downstream tasks")
        return self


class ModalityProfile(StrictModel):
    modality_id: str
    modality_type: Literal[
        "count_matrix_gene",
        "count_matrix_peak",
        "wsi_image",
        "tabular_clinical",
        "dense_image",
        "continuous_tabular",
    ]
    n_samples: int = Field(ge=0)
    n_features: int = Field(ge=0)
    feature_types: list[str] = Field(default_factory=list)
    storage_format: str
    shape_and_channels: list[int] = Field(default_factory=list)
    value_range: list[float] = Field(default_factory=list)
    dtype: str = ""
    sparsity: float | None = None
    library_size_distribution: dict[str, float] = Field(default_factory=dict)
    binary_or_count_indicators: dict[str, Any] = Field(default_factory=dict)
    missingness_by_feature: dict[str, float] = Field(default_factory=dict)
    wsi: dict[str, Any] = Field(default_factory=dict)
    qc_metrics_available: list[str] = Field(default_factory=list)
    estimated_memory: int = 0
    estimated_preprocessing_cost: dict[str, Any] = Field(default_factory=dict)


class PairingProfile(StrictModel):
    pairing_key: str
    pairing_level: str
    n_paired: int = Field(ge=0)
    n_unpaired_by_modality: dict[str, int] = Field(default_factory=dict)
    multiplicity: dict[str, Any] = Field(default_factory=dict)
    label_availability_by_split_candidate: dict[str, Any] = Field(default_factory=dict)
    group_or_batch_variables: list[str] = Field(default_factory=list)
    missing_modality_rates: dict[str, float] = Field(default_factory=dict)
    licensing_and_release_constraints: list[str] = Field(default_factory=list)


class DataProfile(StrictModel):
    dataset_id: str
    modalities: list[ModalityProfile]
    pairing: PairingProfile
    source_checksums: dict[str, str]
    label_caveats: list[str] = Field(default_factory=list)
    cohort_flow: dict[str, Any] = Field(default_factory=dict)


class Applicability(StrictModel):
    model_id: str
    family: str
    status: Literal[
        "eligible_for_pilot",
        "reference_only",
        "not_applicable",
        "infeasible_under_budget",
        "failed_validation",
    ]
    rule_id: str
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    encoders: list[str] = Field(default_factory=list)
    fusion_type: str
    training_objective: str


class RunResult(StrictModel):
    dataset_id: str
    objective_id: str
    family: str
    model_id: str
    encoders: list[str]
    fusion_type: str
    training_objective: str
    status: Literal["success", "failed", "timeout", "invalid"]
    eligibility_reason: str
    hyperparameters: dict[str, Any]
    embedding_dim: int = 0
    seed: int
    parameter_count: int = 0
    trainable_parameter_count: int = 0
    epochs_or_steps_completed: int = 0
    early_stopping_reason: str = ""
    training_curves: list[dict[str, Any]] = Field(default_factory=list)
    checkpoint_path: str = ""
    embedding_paths: dict[str, str] = Field(default_factory=dict)
    runtime_seconds: float = 0
    peak_memory_mb: float = 0
    peak_gpu_memory_mb: float = 0
    hardware: dict[str, Any] = Field(default_factory=dict)
    slurm_job_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error_type: str = ""
    error_message: str = ""
    retry_count: int = 0
    package_versions: dict[str, str] = Field(default_factory=dict)
    git_commit: str = ""
    config_hash: str = ""
    analysis_code_hash: str = ""


class Verdict(StrictModel):
    integration_verdict: Literal["integrate", "integrate_with_caution", "single_modality_sufficient"]
    selected_representation: str
    selected_strategy: str
    encoders: list[str]
    fusion_type: str
    training_objective: str
    comparison_to_best_single_modality: dict[str, Any]
    comparison_to_simple_fusion_baseline: dict[str, Any]
    structure_evidence: dict[str, Any]
    alignment_evidence: dict[str, Any]
    utility_evidence: dict[str, Any]
    modality_contribution_evidence: dict[str, Any]
    confounding_and_validity_flags: list[str]
    approved_downstream_uses: list[str]
    discouraged_or_prohibited_uses: list[str]
    uncertainty_and_limitations: list[str]
    conditions_for_reconsideration: list[str]
