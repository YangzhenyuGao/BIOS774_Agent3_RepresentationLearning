"""Gated attention MIL with exported patient-bag attention weights."""
import torch
from torch import nn


class GatedMIL(nn.Module):
    def __init__(self, feature_dim, hidden=128, out_dim=32):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.attention_v = nn.Linear(hidden, 64)
        self.attention_u = nn.Linear(hidden, 64)
        self.attention_w = nn.Linear(64, 1)
        self.output = nn.Linear(hidden, out_dim)

    def forward(self, bags, mask=None):
        x = self.project(bags)
        scores = self.attention_w(torch.tanh(self.attention_v(x)) * torch.sigmoid(self.attention_u(x))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -torch.inf)
        attention = scores.softmax(1)
        return self.output((x * attention.unsqueeze(-1)).sum(1)), attention


def pad_bags(bags, device, indices=None, tile_dropout=0):
    if indices is not None:
        bags = [bags[i] for i in indices]
    maximum = max(len(b) for b in bags)
    dim = bags[0].shape[1]
    x = torch.zeros(len(bags), maximum, dim, device=device)
    mask = torch.zeros(len(bags), maximum, dtype=torch.bool, device=device)
    for i, bag in enumerate(bags):
        z = torch.as_tensor(bag, device=device, dtype=torch.float32)
        if tile_dropout and len(z) > 1:
            keep = torch.rand(len(z), device=device) > tile_dropout
            keep[0] = True
            z = z[keep]
        x[i, :len(z)], mask[i, :len(z)] = z, True
    return x, mask

