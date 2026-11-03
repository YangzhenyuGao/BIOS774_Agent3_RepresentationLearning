"""CPU loading/forward diagnostic while H100 acceptance waits in the queue."""
import gc
import traceback

import torch

from agent3_replearn.models.pathology_fm import MODEL_IDS, PathologyEncoder
from agent3_replearn.utils import ROOT, write_json

torch.set_num_threads(8)
rows = []
for mid in MODEL_IDS:
    try:
        model = PathologyEncoder(mid, '/vast/som/yunligrp/users/ygao/models', verify=False).eval()
        with torch.inference_mode():
            z = model(torch.randn(1, 3, model.image_size, model.image_size))
            assert z.shape == (1, model.output_dim) and torch.isfinite(z).all()
            if mid in {'CONCH-Z', 'MUSK'}:
                im = model(torch.randn(1, 3, model.image_size, model.image_size), projected=True)
                tx = model.encode_text(['Age 50 years. Stage II. ER positive.'])
                assert im.shape == tx.shape and torch.isfinite(tx).all()
        rows.append(dict(status='success', **model.record()))
        del model, z
    except Exception as exc:
        traceback.print_exc()
        rows.append(dict(model_id=mid, status='failed', error=str(exc)))
    gc.collect()
    print(rows[-1], flush=True)
write_json(ROOT / 'logs/preflight/cpu_model_diagnostic.json', dict(models=rows, passed=all(r['status'] == 'success' for r in rows)))
assert all(r['status'] == 'success' for r in rows)
