"""Actual local CONCH-Z tower updates on a bounded patient-level tile subset."""
import copy
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from ..preprocessing.clinical import clinical_text
from ..utils import seed_all
from .base import CapabilityError, RepresentationAdapter
from .neural import infonce
from .pathology_fm import PathologyEncoder


class FoundationFineTuneAdapter(RepresentationAdapter):
    def __init__(self, model_id, params):
        self.model_id, self.params = model_id, params
        self.history = []

    def capabilities(self):
        return dict(new_sample_encoding=True, cross_modal_prediction=False,
                    attention=False, missing_modality=False,
                    supervised=self.model_id == 'fm_last_block', updates_pretrained_towers=True)

    def _initialize(self, dimensions):
        self.encoder = PathologyEncoder('CONCH-Z', self.params.get('model_store', '/vast/som/yunligrp/users/ygao/models'))
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        # Last transformer blocks are actual pretrained tower parameters.
        for p in self.encoder.model.visual.trunk.blocks[-1].parameters():
            p.requires_grad_(True)
        if self.model_id == 'fm_image_text':
            for p in self.encoder.model.text.transformer.resblocks[-1].parameters():
                p.requires_grad_(True)
        latent = self.params.get('latent_dim', 32)
        self.image_projection = nn.Linear(768, latent)
        self.table_projection = nn.Linear(dimensions[1], latent)
        self.head = nn.Linear(2 * latent, max(2, len(self.classes)))
        if self.model_id == 'fm_image_text':
            for m in [self.image_projection, self.table_projection, self.head]:
                m.requires_grad_(False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for module in self.modules():
            module.to(self.device)

    def modules(self):
        return [self.encoder, self.image_projection, self.table_projection, self.head]

    def batch(self, data, indices, random_tile=False):
        images = []
        for i in indices:
            paths = data['tile_images'][i]
            if not paths:
                raise ValueError('Fine-tuning requires the declared bounded tile cache')
            path = paths[np.random.randint(len(paths)) if random_tile else 0]
            with Image.open(path) as image:
                images.append(self.encoder.transform(image.convert('RGB')))
        x = torch.stack(images).to(self.device)
        table = torch.as_tensor(data['views'][1][indices], device=self.device, dtype=torch.float32)
        # Most layers are frozen; last-block gradients remain enabled.
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == 'cuda'):
            image_features = self.encoder(x, projected=self.model_id == 'fm_image_text').float()
            if self.model_id == 'fm_image_text':
                frame = data['obs'].iloc[indices]
                columns = self.params.get('clinical_columns', ['age', 'stage', 'histological_type', 'er', 'pr', 'her2'])
                text = clinical_text(frame, columns)
                text_features = self.encoder.encode_text(text).float()
                return image_features, text_features
        return self.image_projection(image_features), self.table_projection(table)

    def fit(self, train_data, val_data, params=None, seed=774):
        self.validate_input(train_data, val_data)
        seed_all(seed)
        self.params.update(params or {})
        self.params['dimensions'] = [v.shape[1] for v in train_data['views']]
        self.classes = (sorted(set(train_data['labels']) - {'__unlabeled__', 'filtered'})
                        if self.model_id == 'fm_last_block' else ['unused_0', 'unused_1'])
        self._initialize([v.shape[1] for v in train_data['views']])
        # BN/dropout remain in evaluation mode, as standard in partial ViT fine-tuning.
        self.encoder.eval()
        trainable = [p for m in self.modules() for p in m.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.params.get('lr', 1e-5), weight_decay=.01)
        cap = min(self.params.get('finetune_patients', 128), len(train_data['ids']))
        candidates = np.arange(len(train_data['ids']))
        if self.model_id == 'fm_last_block':
            candidates = candidates[np.isin(train_data['labels'], self.classes)]
        chosen = np.random.permutation(candidates)[:cap]
        self.fit_ids = np.asarray(train_data['ids'])[chosen].tolist()
        self.validation_ids = list(val_data['ids'])[:min(32, len(val_data['ids']))]
        self.parameter_count = sum(p.numel() for m in self.modules() for p in m.parameters())
        self.trainable_parameter_count = sum(p.numel() for p in trainable)
        initial = {k: p.detach().cpu().clone() for k, p in self.encoder.named_parameters() if p.requires_grad}
        best, state, stale = float('inf'), None, 0
        started = time.monotonic()
        batch_size = max(2, min(4, self.params.get('batch_size', 4)))
        for epoch in range(self.params.get('epochs', 50)):
            losses = []
            for indices in np.array_split(np.random.permutation(chosen), max(1, int(np.ceil(len(chosen) / batch_size)))):
                if len(indices) < 2:
                    continue
                optimizer.zero_grad()
                a, b = self.batch(train_data, indices, random_tile=True)
                if self.model_id == 'fm_image_text':
                    loss = infonce(a, b, self.params.get('temperature', .1))
                else:
                    target = torch.tensor([self.classes.index(train_data['labels'][i]) for i in indices], device=self.device)
                    loss = F.cross_entropy(self.head(torch.cat([a, b], 1)), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite actual tower fine-tuning loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
                if time.monotonic() - started > self.params.get('seconds', 600):
                    break
            values = []
            with torch.no_grad():
                nv = min(32, len(val_data['ids']))
                for ix in np.array_split(np.arange(nv), max(1, int(np.ceil(nv / batch_size)))):
                    if len(ix) < 2:
                        continue
                    a, b = self.batch(val_data, ix)
                    if self.model_id == 'fm_image_text':
                        loss = infonce(a, b, self.params.get('temperature', .1))
                    else:
                        y = torch.tensor([self.classes.index(val_data['labels'][i]) if val_data['labels'][i] in self.classes else -1 for i in ix], device=self.device)
                        valid = y >= 0
                        if not valid.any():
                            continue
                        loss = F.cross_entropy(self.head(torch.cat([a, b], 1))[valid], y[valid])
                    values.append(float(loss))
            validation = float(np.mean(values))
            self.history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_loss=validation))
            if validation < best:
                best, state, stale = validation, self.trainable_state(), 0
            else:
                stale += 1
            if stale >= 5 or time.monotonic() - started > self.params.get('seconds', 600):
                break
        if state is None:
            raise RuntimeError('No validated fine-tuning checkpoint')
        self.restore_trainable_state(state)
        delta = sum((p.detach().cpu() - initial[k]).square().sum().item()
                    for k, p in self.encoder.named_parameters() if k in initial)
        if delta <= 0:
            raise ValueError('Pretrained tower parameters did not change; cannot claim fine-tuning')
        self.pretrained_parameter_squared_change = delta
        for m in self.modules():
            m.cpu()
        return self

    def trainable_state(self):
        return {str(i): {k: v.detach().cpu().clone() for k, v in m.named_parameters() if v.requires_grad}
                for i, m in enumerate(self.modules())}

    def restore_trainable_state(self, state):
        for i, module in enumerate(self.modules()):
            expected = {k: p for k, p in module.named_parameters() if p.requires_grad}
            if set(expected) != set(state[str(i)]):
                raise ValueError('Fine-tuned parameter-name mismatch')
            with torch.no_grad():
                for name, parameter in expected.items():
                    parameter.copy_(state[str(i)][name].to(parameter.device))

    def encode(self, data, modality=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for m in self.modules():
            m.to(self.device).eval()
        rows = []
        batch_size = max(2, min(4, self.params.get('batch_size', 4)))
        with torch.no_grad():
            for i in range(0, len(data['ids']), batch_size):
                ix = np.arange(i, min(i + batch_size, len(data['ids'])))
                a, b = self.batch(data, ix)
                z = ((F.normalize(a, dim=1) + F.normalize(b, dim=1)) / 2
                     if self.model_id == 'fm_image_text' else torch.cat([a, b], 1))
                if modality is not None:
                    z = [a, b][int(modality)]
                rows.append(z.float().cpu().numpy())
        for m in self.modules():
            m.cpu()
        return np.concatenate(rows)

    def predict(self, data):
        if self.model_id != 'fm_last_block':
            raise CapabilityError('Image-text fine-tuning arm has no supervised head')
        z = self.encode(data)
        with torch.no_grad():
            return self.head(torch.tensor(z)).softmax(1).numpy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainable_state(), path.with_suffix('.trainable.pt'))
        modules = self.modules()
        del self.encoder, self.image_projection, self.table_projection, self.head
        try:
            super().save(path)
        finally:
            self.encoder, self.image_projection, self.table_projection, self.head = modules

    @classmethod
    def load(cls, path):
        instance = super().load(path)
        instance._initialize(instance.params['dimensions'])
        instance.restore_trainable_state(torch.load(Path(path).with_suffix('.trainable.pt'), map_location='cpu', weights_only=True))
        return instance
