"""Hardware/model-loading acceptance; no real data, training or pilot selection."""
import gc
import json
import subprocess
import time
from importlib.metadata import version

import torch

from agent3_replearn.models.pathology_fm import MODEL_IDS, PathologyEncoder
from agent3_replearn.utils import ROOT, provenance, write_json

assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.get_device_capability() == (9, 0), 'Expected H100 sm_90'
assert 'sm_90' in torch.cuda.get_arch_list(), 'PyTorch lacks native H100 support'
x = torch.randn(32, 16, device='cuda', requires_grad=True)
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(x.grad).all()
rows = []
for mid in MODEL_IDS:
    started = time.monotonic()
    try:
        model = PathologyEncoder(mid, '/vast/som/yunligrp/users/ygao/models', verify=False).to('cuda').eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            z = model(torch.randn(1, 3, model.image_size, model.image_size, device='cuda'))
            assert z.shape == (1, model.output_dim), (mid, z.shape)
            assert torch.isfinite(z).all()
            if mid in {'CONCH-Z', 'MUSK'}:
                im = model(torch.randn(1, 3, model.image_size, model.image_size, device='cuda'), projected=True)
                tx = model.encode_text(['Age 50 years. Stage II. Histology ductal carcinoma. ER positive.'])
                assert im.shape == tx.shape and torch.isfinite(tx).all()
        rows.append(dict(status='success', **model.record(), elapsed_seconds=time.monotonic() - started,
                         peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2))
        del model, z
    except Exception as exc:
        import traceback
        traceback.print_exc()
        rows.append(dict(model_id=mid, status='failed', error_type=type(exc).__name__, error=str(exc)))
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(rows[-1]), flush=True)
write_json(ROOT / 'logs/preflight/gpu_runtime.json', dict(passed=all(r['status'] == 'success' for r in rows),
           models=rows, torch=version('torch'), timm=version('timm'), transformers=version('transformers'),
           cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
           nvidia_smi=subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout, **provenance()))
assert all(r['status'] == 'success' for r in rows), 'At least one foundation model failed runtime acceptance'
