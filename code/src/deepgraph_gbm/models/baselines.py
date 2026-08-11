"""Baselines: MLP (no graph) and multinomial logistic regression."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn


class MLPBaseline(nn.Module):
    """Same capacity ballpark as the GNN encoder, but no neighborhood info."""

    def __init__(self, in_dim: int, hidden_dims=(256, 256, 128), dropout: float = 0.3, n_classes: int = 4):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.niche_head = nn.Linear(hidden_dims[-1], n_classes)
        self.mes_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, edge_index=None, with_risk: bool = False):
        z = self.trunk(x)
        return {
            "z": z,
            "niche_logits": self.niche_head(z),
            "mes_prob": torch.sigmoid(self.mes_head(z)).squeeze(-1),
        }


class LogisticBaseline:
    """Multinomial logistic regression on HVG expression (niche) + ridge (MES)."""

    def __init__(self, max_iter: int = 500):
        self.clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", n_jobs=1)
        self.mes = LogisticRegression(max_iter=max_iter, n_jobs=1)

    def fit(self, X: np.ndarray, niche_y: np.ndarray, mes_y: np.ndarray):
        mask = niche_y >= 0
        self.clf.fit(X[mask], niche_y[mask])
        mmask = ~np.isnan(mes_y)
        self.mes.fit(X[mmask], (mes_y[mmask] > 0.5).astype(int))
        return self

    def predict_proba(self, X: np.ndarray):
        return self.clf.predict_proba(X)

    def mes_proba(self, X: np.ndarray):
        return self.mes.predict_proba(X)[:, 1]
