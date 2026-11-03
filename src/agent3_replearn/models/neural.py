"""Genuine NT-Xent, symmetric InfoNCE, supervised fusion and gated MIL."""
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..utils import seed_all
from .base import CapabilityError, RepresentationAdapter
from .mil import GatedMIL, pad_bags


def infonce(a, b, temperature):
    logits = F.normalize(a, dim=1) @ F.normalize(b, dim=1).T / temperature
    target = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def ntxent(a, b, temperature=0.2):
    z = F.normalize(torch.cat([a, b]), dim=1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    n = len(a)
    targets = (torch.arange(2 * n, device=a.device) + n) % (2 * n)
    return F.cross_entropy(logits, targets)


def cox_loss(risk, times, events):
    # Full risk sets; Breslow denominator for ties, rather than order-dependent ties.
    at_risk = times[None, :] >= times[:, None]
    denominator = torch.logsumexp(risk[None, :].expand(len(risk), -1).masked_fill(~at_risk, -torch.inf), dim=1)
    return -((risk - denominator) * events).sum() / events.sum().clamp_min(1)


class FusionNetwork(nn.Module):
    def __init__(self, dimensions, latent, hidden, n_classes, fusion, use_bags=False):
        super().__init__()
        self.use_bags, self.fusion = use_bags, fusion
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(.1),
                                                     nn.Linear(hidden, latent)) for d in dimensions])
        if use_bags:
            self.mil = GatedMIL(dimensions[0], hidden, latent)
            self.encoders[0] = nn.Identity()
        self.gate = nn.Linear(2 * latent, latent)
        self.early = nn.Sequential(nn.Linear(2 * latent, latent), nn.ReLU())
        self.head = nn.Linear(latent, n_classes)
        self.head_b = nn.Linear(latent, n_classes)
        self.log_temperature = nn.Parameter(torch.tensor(np.log(.1), dtype=torch.float32))

    def views(self, xs, bags=None, bagmask=None):
        a = self.mil(bags, bagmask)[0] if self.use_bags else self.encoders[0](xs[0])
        return a, self.encoders[1](xs[1])

    def joint(self, a, b):
        if self.fusion == 'abmil':
            return a
        if self.fusion == 'fusion_early':
            return self.early(torch.cat([a, b], dim=1))
        if self.fusion == 'fusion_gated':
            weight = torch.sigmoid(self.gate(torch.cat([a, b], dim=1)))
            return weight * a + (1 - weight) * b
        if self.fusion == 'fusion_late':
            # Representation is the concatenation; prediction is decision-level fusion.
            return torch.cat([a, b], dim=1)
        if self.fusion == 'simclr':
            return torch.cat([F.normalize(a, dim=1), F.normalize(b, dim=1)], dim=1)
        return (F.normalize(a, dim=1) + F.normalize(b, dim=1)) / 2

    def predict_head(self, a, b):
        if self.fusion == 'fusion_late':
            return (self.head(a) + self.head_b(b)) / 2
        return self.head(self.joint(a, b))


class NeuralAdapter(RepresentationAdapter):
    def __init__(self, model_id, params):
        self.model_id, self.params = model_id, params
        self.supervised = model_id.startswith('fusion_') or model_id == 'abmil'
        self.history = []

    def capabilities(self):
        return dict(new_sample_encoding=True, cross_modal_prediction=False,
                    attention=getattr(self, 'use_bags', False), missing_modality=self.supervised,
                    supervised=self.supervised)

    def fit(self, train_data, val_data, params=None, seed=774):
        self.validate_input(train_data, val_data)
        seed_all(seed)
        self.params = {**self.params, **(params or {})}
        p = self.params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_bags = bool(train_data.get('bags'))
        if self.model_id == 'abmil' and not self.use_bags:
            raise ValueError('ABMIL requires actual tile bags')
        self.classes = (sorted(set(train_data['labels']) - {'filtered', '__unlabeled__'})
                        if self.supervised else ['unused_0', 'unused_1'])
        self.survival = p.get('task') == 'survival'
        self.model = FusionNetwork([x.shape[1] for x in train_data['views']], p.get('latent_dim', 32),
                                    p.get('hidden', 128), 1 if self.survival else max(2, len(self.classes)),
                                    self.model_id, self.use_bags).to(self.device)
        inactive = []
        if self.model_id != 'fusion_gated':
            inactive.append(self.model.gate)
        if self.model_id != 'fusion_early':
            inactive.append(self.model.early)
        if self.model_id != 'fusion_late':
            inactive.append(self.model.head_b)
        if not self.supervised:
            inactive.append(self.model.head)
        if self.model_id == 'abmil':
            inactive.append(self.model.encoders[1])
        for module in inactive:
            module.requires_grad_(False)
        if self.model_id != 'clip_dual':
            self.model.log_temperature.requires_grad_(False)
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                      lr=p.get('lr', 1e-3), weight_decay=1e-4)
        n = len(train_data['ids'])
        batch = n if self.survival and self.supervised else min(p.get('batch_size', 128), n)
        best, best_state, stale = float('inf'), None, 0
        started = time.monotonic()
        for epoch in range(p.get('epochs', 50)):
            self.model.train()
            losses = []
            for indices in np.array_split(np.random.permutation(n), max(1, int(np.ceil(n / batch)))):
                if len(indices) < 2:
                    continue
                optimizer.zero_grad()
                loss = self.loss(train_data, indices, augment=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()
                losses.append(loss.item())
            self.model.eval()
            with torch.no_grad():
                validation = float(self.loss(val_data, np.arange(min(len(val_data['ids']), 512)), augment=False))
            self.history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_loss=validation))
            if validation < best - 1e-4:
                best, best_state, stale = validation, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
            if stale >= p.get('patience', 10):
                self.stopping_reason = 'validation early stopping'
                break
            if time.monotonic() - started >= p.get('seconds', 600):
                self.stopping_reason = 'declared fit-time budget'
                break
        else:
            self.stopping_reason = 'maximum epochs'
        if best_state is None:
            raise RuntimeError('No finite validation checkpoint')
        self.model.load_state_dict(best_state, strict=True)
        self.model.eval().cpu()
        self.fit_ids = list(train_data['ids'])
        self.parameter_count = sum(v.numel() for v in self.model.parameters())
        self.trainable_parameter_count = sum(v.numel() for v in self.model.parameters() if v.requires_grad)
        return self

    def batch(self, data, indices, augment=False):
        xs = [torch.as_tensor(x[indices], dtype=torch.float32, device=self.device) for x in data['views']]
        bags, mask = (pad_bags(data['bags'], self.device, indices, .2 if augment else 0)
                      if self.use_bags else (None, None))
        if augment:
            # Feature dropout in the declared train-fitted PCA/LSI or frozen-feature space.
            xs = [F.dropout(x, p=.15, training=True) for x in xs]
            if self.supervised:
                for i in range(2):
                    drop = torch.rand(len(indices), 1, device=self.device) < .1
                    xs[i] = xs[i] * (~drop)
                    if i == 0 and bags is not None:
                        bags = bags * (~drop).unsqueeze(-1)
        return xs, bags, mask

    def loss(self, data, indices, augment=False):
        xs, bags, mask = self.batch(data, indices, augment)
        a, b = self.model.views(xs, bags, mask)
        if self.supervised:
            logits = self.model.predict_head(a, b)
            if self.survival:
                times = torch.as_tensor(data['os_days'][indices], device=self.device)
                events = torch.as_tensor(data['os_event'][indices], device=self.device)
                valid = torch.isfinite(times) & torch.isfinite(events) & (times > 0)
                return cox_loss(logits[valid, 0], times[valid], events[valid])
            y = torch.tensor([self.classes.index(v) if v in self.classes else -1 for v in np.asarray(data['labels'])[indices]], device=self.device)
            if (y >= 0).sum() == 0:
                return logits.sum() * 0
            return F.cross_entropy(logits, y, ignore_index=-1)
        if self.model_id == 'simclr':
            second, bags2, mask2 = self.batch(data, indices, augment=augment)
            a2, b2 = self.model.views(second, bags2, mask2)
            return (ntxent(a, a2) + ntxent(b, b2)) / 2
        temperature = self.model.log_temperature.exp().clamp(.02, .5)
        return infonce(a, b, temperature)

    def _outputs(self, data):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()
        blocks = [[], [], [], []]
        with torch.no_grad():
            batch = min(64, self.params.get('batch_size', 128))
            for start in range(0, len(data['ids']), batch):
                ix = np.arange(start, min(start + batch, len(data['ids'])))
                xs, bags, mask = self.batch(data, ix)
                a, b = self.model.views(xs, bags, mask)
                values = [self.model.joint(a, b), a, b, self.model.predict_head(a, b)
                          if self.supervised else torch.zeros(len(a), 1, device=self.device)]
                for dest, value in zip(blocks, values):
                    dest.append(value.float().cpu().numpy())
        self.model.cpu()
        return [np.concatenate(v) for v in blocks]

    def encode(self, data, modality=None):
        if self.model_id == 'abmil' and modality is not None and int(modality) == 1:
            raise CapabilityError('Histology-only MIL has no clinical representation encoder')
        outputs = self._outputs(data)
        return outputs[0 if modality is None else int(modality) + 1]

    def predict(self, data):
        if not self.supervised:
            raise CapabilityError('Contrastive adapter has no supervised head')
        logits = self._outputs(data)[3]
        if self.survival:
            return logits[:, 0]
        return torch.softmax(torch.tensor(logits), dim=1).numpy()

    def attention(self, data):
        if not self.use_bags:
            raise CapabilityError('This run has no tile-bag encoder')
        weights = []
        self.model.eval()
        with torch.no_grad():
            for bag in data['bags']:
                _, attention = self.model.mil(torch.as_tensor(bag, dtype=torch.float32)[None])
                weights.append(attention[0].numpy())
        return weights

    def save(self, path):
        self.model.cpu()
        super().save(path)
