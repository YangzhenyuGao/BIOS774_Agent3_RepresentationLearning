import os

import numpy as np
import pytest

from agent3_replearn.registry import build_adapter


@pytest.mark.parametrize(
    "model_id",
    [
        "view_a",
        "view_b",
        "concatenation",
        "wnn",
        "multivi",
        "mofa",
        "simclr",
        "clip_dual",
        "fusion_early",
        "fusion_late",
        "fusion_gated",
        "abmil",
    ],
)
def test_real_adapter_training_inductive_encode_save_load(paired_data, model_id, tmp_path):
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.skip("Adapter training belongs in Slurm")
    train, val, test = paired_data
    if model_id == "abmil":
        for data in [train, val, test]:
            data["bags"] = [np.repeat(x[None], 4, axis=0) for x in data["views"][0]]
    model = build_adapter(model_id, dict(epochs=3, latent_dim=4, hidden=16, seconds=60, batch_size=16))
    model.fit(train, val, seed=774)
    z = model.encode(test)
    assert z.shape[0] == len(test["ids"])
    assert z.ndim == 2 and z.shape[1] > 0 and np.isfinite(z).all()
    if hasattr(model, "fit_ids"):
        assert not set(model.fit_ids) & set(test["ids"])
    path = tmp_path / f"{model_id}.joblib"
    model.save(path)
    loaded = type(model).load(path)
    restored = loaded.encode(test)
    np.testing.assert_allclose(z, restored, atol=1e-4, rtol=1e-4)
    if model_id == "abmil":
        for weights in loaded.attention(test):
            assert np.isclose(weights.sum(), 1)
