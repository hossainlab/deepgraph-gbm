"""Evaluation: niche classification, MES regression, slide risk."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

NICHE_CLASSES = ["immune_cold", "immune_hot", "normal", "necrotic"]


@torch.no_grad()
def predict_graphs(model, data_list, device="cpu"):
    """Run inference; returns per-graph dicts of predictions and labels."""
    model.eval()
    outs = []
    for d in data_list:
        d = d.to(device)
        o = model(d.x, d.edge_index)
        outs.append(
            {
                "section": d.section,
                "patient": d.patient,
                "niche_logits": o["niche_logits"].cpu(),
                "mes_prob": o["mes_prob"].cpu(),
                "risk": o["risk"].item(),
                "niche_y": d.niche_y.cpu(),
                "mes_y": d.mes_y.cpu(),
                "z": o["z"].cpu(),
            }
        )
    return outs


def tune_thresholds(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Per-class decision threshold tuned to maximize per-class F1 (one-vs-rest).

    Returns thresholds per class; prediction = class with max (prob / threshold).
    Falls back to 0.5 when a class has no positive examples.
    """
    thr = np.full(prob.shape[1], 0.5, dtype=np.float32)
    for i in range(prob.shape[1]):
        yi = (y == i).astype(int)
        if yi.sum() == 0:
            continue
        best_t, best_f = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 37):
            f = f1_score(yi, (prob[:, i] >= t).astype(int))
            if f > best_f:
                best_f, best_t = f, t
        thr[i] = best_t
    return thr


def apply_thresholds(prob: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Pick the class whose probability most exceeds its tuned threshold."""
    return (prob / thr).argmax(1)


def classification_metrics(preds: list, thr: np.ndarray | None = None) -> dict:
    """Aggregate niche classification metrics over graphs (labeled spots only).

    If thr is None, thresholds are tuned on these predictions (val); otherwise the
    supplied thresholds are applied (test) so test metrics never see test labels.
    """
    y = torch.cat([p["niche_y"] for p in preds]).numpy()
    logits = torch.cat([p["niche_logits"] for p in preds]).numpy()
    mask = y >= 0
    y, logits = y[mask], logits[mask]
    if len(y) == 0:
        return {"macro_f1": 0.0}
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    if thr is None:
        thr = tune_thresholds(y, prob)
    yhat = apply_thresholds(prob, thr)

    m = {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "weighted_f1": f1_score(y, yhat, average="weighted"),
        "accuracy": float((yhat == y).mean()),
        "confusion": confusion_matrix(y, yhat, labels=list(range(4))).tolist(),
        "thresholds": thr.tolist() if hasattr(thr, "tolist") else list(thr),
    }
    for i, c in enumerate(NICHE_CLASSES):
        yi = (y == i).astype(int)
        if 0 < yi.sum() < len(yi):
            m[f"auroc_{c}"] = roc_auc_score(yi, prob[:, i])
            m[f"auprc_{c}"] = average_precision_score(yi, prob[:, i])
        m[f"f1_{c}"] = f1_score(y == i, yhat == i)
    return m


def mes_metrics(preds: list) -> dict:
    y = torch.cat([p["mes_y"] for p in preds]).numpy()
    p = torch.cat([p["mes_prob"] for p in preds]).numpy()
    mask = ~np.isnan(y)
    y, p = y[mask], p[mask]
    out = {}
    if len(np.unique(y)) > 1:
        out["mes_auroc"] = roc_auc_score(y, p)
        out["mes_auprc"] = average_precision_score(y, p)
    out["mes_pearson"] = float(np.corrcoef(y, p)[0, 1])
    return out


def evaluate_graphs(model, data_list, device="cpu", thr: np.ndarray | None = None) -> dict:
    preds = predict_graphs(model, data_list, device)
    m = classification_metrics(preds, thr=thr)
    m.update(mes_metrics(preds))
    return m
