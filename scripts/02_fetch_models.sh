#!/usr/bin/env bash
# The four pathology FMs are already local. Fetch only the ImageNet reference.
set -euo pipefail
source "$(dirname "$0")/slurm/common.sh"
python - <<'PY'
from pathlib import Path
import torch
from torchvision.models import ResNet18_Weights
from agent3_replearn.utils import checksum, write_json
weights = ResNet18_Weights.IMAGENET1K_V1
path = Path(__import__('os').environ['TORCH_HOME']) / 'hub/checkpoints' / Path(weights.url).name
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    torch.hub.download_url_to_file(weights.url, str(path), hash_prefix='f37072fd')
assert checksum(path).startswith('f37072fd')
write_json('logs/preflight/imagenet_weights.json', dict(source=weights.url, path=str(path), sha256=checksum(path),
          model='torchvision ResNet18 ImageNet1K V1', pathology_foundation_model=False))
PY
