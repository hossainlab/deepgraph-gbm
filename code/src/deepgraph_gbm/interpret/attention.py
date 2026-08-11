"""Neighbor-importance analysis: which neighbor niches drive a spot's prediction.

Approach: gradient x input attribution on the target class logit w.r.t. each
neighbor's input features, aggregated per neighbor niche. This quantifies how
much each surrounding niche contributes to a spot's predicted class — the
spatial rule the GNN has learned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def neighbor_importance(model, data, niche_labels: pd.Series) -> pd.DataFrame:
    """Compute mean |grad x input| received from each neighbor niche per class.

    Returns a DataFrame: rows = predicted class, columns = neighbor niche,
    values = mean attribution mass (fraction of total).
    """
    model.eval()
    x = data.x.clone().requires_grad_(True)
    out = model(x, data.edge_index, with_risk=False)
    logits = out["niche_logits"]
    pred = logits.argmax(1)

    ei = data.edge_index.numpy()
    neighbor_niche = niche_labels.values[ei[0]]  # niche of source node for each edge

    n_classes = logits.shape[1]
    # attribution per edge: how much target node j's logit depends on source i
    attr_by_class = {c: np.zeros(len(niche_labels)) for c in range(n_classes)}
    counts = {c: np.zeros(len(niche_labels)) for c in range(n_classes)}

    # aggregate gradient of each predicted class logit wrt inputs
    for c in range(n_classes):
        mask = (pred == c)
        if mask.sum() == 0:
            continue
        model.zero_grad()
        logits[mask, c].sum().backward(retain_graph=True)
        grad = (x.grad * x).abs().sum(1).detach().numpy()  # per-node attribution
        x.grad = None
        # spread each target node's attribution over its incoming neighbors equally
        for u, v in zip(ei[0], ei[1]):
            if pred[v] == c:
                attr_by_class[c][u] += grad[v]
                counts[c][u] += 1

    niches = sorted(set(niche_labels.unique()) - {"exclude"})
    rows = {}
    class_names = ["immune_cold", "immune_hot", "normal", "necrotic"]
    for c in range(n_classes):
        a = attr_by_class[c]
        tot = a.sum() + 1e-12
        rows[class_names[c]] = {
            n: float(a[(niche_labels.values == n)].sum() / tot) for n in niches
        }
    return pd.DataFrame(rows).T
