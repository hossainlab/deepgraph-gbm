"""Multi-task GNN: shared GraphSAGE encoder + niche/MES/survival heads."""

from __future__ import annotations

import torch
from torch import nn

from .encoder import GraphSAGEEncoder
from .heads import MESRegressor, NicheClassifier, SurvivalRiskHead


class DeepGraphGBM(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] = (256, 256, 128),
        dropout: float = 0.3,
        n_niche_classes: int = 4,
        aggr: str = "mean",
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_dim, hidden_dims, dropout, aggr)
        d = self.encoder.out_dim
        self.niche_head = NicheClassifier(d, n_niche_classes)
        self.mes_head = MESRegressor(d)
        self.surv_head = SurvivalRiskHead(d)

    def forward(self, x, edge_index, with_risk: bool = True):
        """Returns dict with niche_logits, mes_prob, (optional) slide risk."""
        z = self.encoder(x, edge_index)
        out = {
            "z": z,
            "niche_logits": self.niche_head(z),
            "mes_prob": self.mes_head(z),
        }
        if with_risk:
            out["risk"] = self.surv_head(z)
        return out


def multitask_loss(
    out: dict,
    niche_y: torch.Tensor,
    mes_y: torch.Tensor,
    risk_y: torch.Tensor | None,
    class_weights: torch.Tensor | None = None,
    weights: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Combined loss. Spots with niche_y == -1 or NaN mes_y are masked out.

    risk_y: scalar pseudo-risk target for the slide (optional).
    """
    weights = weights or {"niche": 1.0, "mes": 0.5, "survival": 0.5}
    device = out["niche_logits"].device
    total = torch.zeros((), device=device)
    parts = {}

    mask = niche_y >= 0
    if mask.any():
        l_niche = nn.functional.cross_entropy(
            out["niche_logits"][mask], niche_y[mask], weight=class_weights
        )
        total = total + weights["niche"] * l_niche
        parts["niche"] = l_niche.item()

    mes_mask = ~torch.isnan(mes_y)
    if mes_mask.any():
        l_mes = nn.functional.binary_cross_entropy(
            out["mes_prob"][mes_mask], mes_y[mes_mask]
        )
        total = total + weights["mes"] * l_mes
        parts["mes"] = l_mes.item()

    if risk_y is not None and "risk" in out:
        l_risk = nn.functional.mse_loss(out["risk"], risk_y)
        total = total + weights["survival"] * l_risk
        parts["survival"] = l_risk.item()

    return total, parts
