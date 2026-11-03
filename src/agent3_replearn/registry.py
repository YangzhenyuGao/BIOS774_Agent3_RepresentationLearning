"""Method metadata separate from eligibility and concrete adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    model_id: str
    family: str
    encoders: tuple[str, ...]
    fusion: str
    loss: str
    data_requirement: str
    supervised: bool = False


METHODS = [
    MethodSpec("view_a", "baseline_unimodal", ("PCA_or_frozen_mean",), "none", "none", "any"),
    MethodSpec("view_b", "baseline_unimodal", ("LSI_or_clinical",), "none", "none", "any"),
    MethodSpec("concatenation", "baseline_fusion", ("block_scaled_views",), "early", "none", "paired"),
    MethodSpec("wnn", "baseline_fusion", ("muon_WNN",), "neighborhood", "weighted_neighbor_graph", "counts"),
    MethodSpec(
        "multivi", "self_supervised_generative", ("scvi_MULTIVI",), "latent", "ELBO_NB_Bernoulli", "counts"
    ),
    MethodSpec(
        "mofa", "self_supervised_generative", ("mofapy2",), "factor", "variational_factor_model", "numeric"
    ),
    MethodSpec("simclr", "contrastive_unimodal", ("MLP_or_MIL",), "late_concatenation", "NT_Xent", "paired"),
    MethodSpec(
        "clip_dual",
        "clip_cross_modal",
        ("MLP_or_MIL", "MLP"),
        "projected_mean",
        "symmetric_InfoNCE",
        "paired",
    ),
    MethodSpec(
        "fusion_early", "supervised_fusion", ("MLP_or_MIL", "MLP"), "early", "cross_entropy", "paired", True
    ),
    MethodSpec(
        "fusion_late", "supervised_fusion", ("MLP_or_MIL", "MLP"), "late", "cross_entropy", "paired", True
    ),
    MethodSpec(
        "fusion_gated",
        "supervised_fusion",
        ("gated_ABMIL", "MLP"),
        "intermediate",
        "cross_entropy",
        "paired",
        True,
    ),
    MethodSpec(
        "abmil",
        "pretrained_foundation",
        ("local_pathology_FM", "gated_ABMIL"),
        "none",
        "cross_entropy",
        "wsi",
        True,
    ),
    MethodSpec(
        "fm_last_block",
        "pretrained_foundation",
        ("local_pathology_FM_last_block",),
        "early",
        "cross_entropy",
        "wsi",
        True,
    ),
    MethodSpec(
        "fm_image_text",
        "pretrained_multimodal_finetune",
        ("CONCH-Z_image", "CONCH-Z_text"),
        "projected_mean",
        "symmetric_InfoNCE_partial_unfreezing",
        "wsi",
    ),
]
REGISTRY = {m.model_id: m for m in METHODS}
for mid, encoder in [('histology_conch', 'CONCH-Z'), ('histology_imagenet', 'imagenet_resnet18')]:
    spec = MethodSpec(mid, 'baseline_unimodal', (encoder, 'mean_pool'), 'none', 'none', 'wsi')
    METHODS.append(spec)
    REGISTRY[mid] = spec


def build_adapter(model_id, parameters=None):
    from .models.baselines import BaselineAdapter
    if model_id in {'histology_conch', 'histology_imagenet'}:
        return BaselineAdapter('view_a', parameters or {})

    if model_id in {"view_a", "view_b", "concatenation"}:
        return BaselineAdapter(model_id, parameters or {})
    if model_id == "wnn":
        from .models.wnn import WNNAdapter

        return WNNAdapter(parameters or {})
    if model_id == "multivi":
        from .models.multivi_adapter import MultiVIAdapter

        return MultiVIAdapter(parameters or {})
    if model_id == "mofa":
        from .models.mofa_adapter import MOFAAdapter

        return MOFAAdapter(parameters or {})
    if model_id in {"fm_last_block", "fm_image_text"}:
        from .models.foundation_finetune import FoundationFineTuneAdapter

        return FoundationFineTuneAdapter(model_id, parameters or {})
    from .models.neural import NeuralAdapter

    return NeuralAdapter(model_id, parameters or {})
