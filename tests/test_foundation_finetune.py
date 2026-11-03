import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from agent3_replearn.models.foundation_finetune import FoundationFineTuneAdapter


@pytest.mark.parametrize('model_id', ['fm_last_block', 'fm_image_text'])
def test_actual_pretrained_tower_update_save_load_and_encode(model_id, tmp_path):
    if not os.environ.get('SLURM_JOB_ID'):
        pytest.skip('Actual pretrained transformer training belongs in Slurm')
    rng = np.random.default_rng(774)
    images = []
    for i in range(4):
        path = tmp_path / f'tile_{i}.jpg'
        Image.fromarray(rng.integers(20, 220, (224, 224, 3), dtype=np.uint8)).save(path)
        images.append(str(path))
    blocks = []
    for split in ['train', 'validation', 'test']:
        ids = np.asarray([f'{split}_0', f'{split}_1'])
        obs = pd.DataFrame(dict(label=['a', 'b'], age=[40., 70.], stage=['I', 'III'],
                                histological_type=['ductal', 'lobular'], er=['negative', 'positive'],
                                pr=['negative', 'positive'], her2=['negative', 'positive']), index=ids)
        blocks.append(dict(ids=ids, labels=obs.label.to_numpy(), obs=obs,
                           views=[rng.normal(size=(2, 8)).astype(np.float32), rng.normal(size=(2, 4)).astype(np.float32)],
                           tile_images=[[images[0]], [images[1]]]))
    train, val, test = blocks
    adapter = FoundationFineTuneAdapter(model_id, dict(epochs=1, seconds=120, latent_dim=4, lr=1e-5))
    adapter.fit(train, val, seed=774)
    assert adapter.pretrained_parameter_squared_change > 0
    before = adapter.encode(test)
    path = tmp_path / f'{model_id}.joblib'
    adapter.save(path)
    loaded = FoundationFineTuneAdapter.load(path)
    after = loaded.encode(test)
    np.testing.assert_allclose(before, after, atol=1e-4, rtol=1e-4)
