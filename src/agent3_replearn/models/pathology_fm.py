"""Strict local-weight loaders for exactly the four user-approved pathology FMs."""
import json
from pathlib import Path

import torch
from torch import nn

from ..utils import checksum, read_json

MODEL_IDS = ('UNI2-h', 'Virchow2', 'CONCH-Z', 'MUSK')


def verify_manifest(folder):
    manifest = read_json(folder / 'DOWNLOAD_MANIFEST.json')
    if len(manifest['revision']) != 40:
        raise ValueError('Unpinned local model revision')
    for row in manifest['files']:
        path = folder / row['path']
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Model manifest escapes local folder')
        if path.stat().st_size != row['bytes'] or checksum(path) != row['sha256']:
            raise ValueError(f'Model integrity failure: {path}')
    return manifest


class PathologyEncoder(nn.Module):
    def __init__(self, model_id, store, verify=True):
        super().__init__()
        import timm
        from safetensors.torch import load_file
        from torchvision import transforms as T
        if model_id not in MODEL_IDS:
            raise ValueError(f'Unapproved pathology model: {model_id}')
        self.model_id = model_id
        self.folder = Path(store) / model_id
        self.manifest = verify_manifest(self.folder) if verify else read_json(self.folder / 'DOWNLOAD_MANIFEST.json')
        self.remote_code_sha256 = {}
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.image_size = 224
        self.output_dim = 1536
        if model_id == 'UNI2-h':
            self.model = timm.create_model('vit_giant_patch14_224', pretrained=False,
                                           img_size=224, patch_size=14, depth=24, num_heads=24,
                                           init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
                                           num_classes=0, no_embed_class=True,
                                           mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU,
                                           reg_tokens=8, dynamic_img_size=True)
        elif model_id == 'Virchow2':
            spec = json.loads((self.folder / 'config.json').read_text())
            self.model = timm.create_model(spec['architecture'], pretrained=False,
                                           mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU,
                                           **spec['model_args'])
            self.output_dim = 2560  # CLS plus mean patch token, excluding four register tokens.
        elif model_id == 'CONCH-Z':
            from transformers import AutoConfig, AutoModel
            files = ['modeling_conch_z.py', 'configuration_conch_z.py', 'vision_tower.py',
                     'text_tower.py', 'conch_tokenizer.py']
            self.remote_code_sha256 = {p: checksum(self.folder / p) for p in files}
            cfg = AutoConfig.from_pretrained(str(self.folder), trust_remote_code=True, local_files_only=True)
            cfg.name_or_path = str(self.folder)
            self.model = AutoModel.from_config(cfg, trust_remote_code=True)
            self.image_size, self.output_dim = 448, 768
        else:
            from musk import modeling  # noqa: F401 - registers the official timm architecture.
            self.model = timm.create_model('musk_large_patch16_384', pretrained=False)
            self.image_size, self.output_dim = 384, 1024
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        weight = self.folder / 'model.safetensors'
        state = load_file(str(weight)) if weight.exists() else torch.load(
            self.folder / 'pytorch_model.bin', map_location='cpu', weights_only=True)
        # No missing/unexpected key suppression, interpolation or relaxed matching.
        self.model.load_state_dict(state, strict=True)
        self.weight_sha256 = next(r['sha256'] for r in self.manifest['files']
                                  if r['path'] == (weight.name if weight.exists() else 'pytorch_model.bin'))
        self.transform = T.Compose([T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC),
                                    T.CenterCrop(self.image_size), T.ToTensor(), T.Normalize(mean, std)])
        self.normalization = dict(mean=mean, std=std, interpolation='bicubic', crop='center')
        self.eval()

    def forward(self, images, projected=False):
        if self.model_id == 'CONCH-Z':
            return self.model.encode_image(images) if projected else self.model.encode_image_features(images)
        if self.model_id == 'MUSK':
            return self.model(image=images, with_head=projected, out_norm=projected,
                              ms_aug=False, return_global=True)[0]
        values = self.model(images)
        if self.model_id == 'Virchow2':
            values = torch.cat((values[:, 0], values[:, 5:].mean(1)), dim=1)
        return values

    def encode_text(self, texts):
        if self.model_id == 'CONCH-Z':
            return self.model.encode_text(self.model.tokenizer(texts).to(next(self.parameters()).device))
        if self.model_id == 'MUSK':
            from musk.utils import xlm_tokenizer
            from transformers import XLMRobertaTokenizer
            tokenizer = XLMRobertaTokenizer(str(self.folder / 'tokenizer.spm'))
            pairs = [xlm_tokenizer(t, tokenizer, max_len=100) for t in texts]
            device = next(self.parameters()).device
            tokens = torch.tensor([p[0] for p in pairs], device=device)
            padding = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.bool)
            return self.model(text_description=tokens, padding_mask=padding, with_head=True,
                              out_norm=True, ms_aug=False, return_global=True)[1]
        raise NotImplementedError(f'{self.model_id} is vision-only')

    def record(self):
        return dict(model_id=self.model_id, revision=self.manifest['revision'],
                    license=self.manifest['license'], image_size=self.image_size,
                    embedding_dim=self.output_dim, weight_sha256=self.weight_sha256,
                    strict_state_dict=True, remote_code_sha256=self.remote_code_sha256,
                    normalization=self.normalization,
                    embedding_definition='unprojected, single scale; Virchow CLS + mean patch tokens')
