"""GraphSAGE encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """Stack of SAGEConv layers with BatchNorm, ReLU and dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] = (256, 256, 128),
        dropout: float = 0.3,
        aggr: str = "mean",
    ):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1], aggr=aggr) for i in range(len(dims) - 1)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(d) for d in hidden_dims])
        self.dropout = dropout
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
