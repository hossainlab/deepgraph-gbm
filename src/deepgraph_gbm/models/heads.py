"""Task heads: niche classifier, MES regressor, slide-level survival risk."""

from __future__ import annotations

import torch
from torch import nn


class NicheClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 4):
        super().__init__()
        self.lin = nn.Linear(in_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.lin(z)  # logits (n_spots, n_classes)


class MESRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(z)).squeeze(-1)  # (n_spots,)


class SurvivalRiskHead(nn.Module):
    """Attention-pooled slide-level risk score from spot embeddings."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (n_spots, in_dim) for ONE slide -> scalar risk."""
        w = torch.softmax(self.att(z), dim=0)  # (n_spots, 1)
        pooled = (w * z).sum(dim=0, keepdim=True)  # (1, in_dim)
        return self.mlp(pooled).squeeze(-1).squeeze(-1)  # scalar
