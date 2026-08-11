"""Train + evaluate baselines (logistic regression, MLP) against the GNN.

Usage: python scripts/run_baselines.py --config configs/default.yaml --data <processed_dir> --out results/baselines
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepgraph_gbm.models.baselines import LogisticBaseline, MLPBaseline
from deepgraph_gbm.models.multitask import multitask_loss
from deepgraph_gbm.training.evaluate import NICHE_CLASSES, classification_metrics, mes_metrics
from deepgraph_gbm.training.train import make_class_weights
from deepgraph_gbm.utils import load_config, load_json, save_json, set_seed, setup_logging

log = logging.getLogger("deepgraph_gbm")


def _stack(data_list):
    X = torch.cat([d.x for d in data_list]).numpy()
    niche = torch.cat([d.niche_y for d in data_list]).numpy()
    mes = torch.cat([d.mes_y for d in data_list]).numpy()
    return X, niche, mes


def _metrics_from_prob(y, prob, mes_y, mes_p):
    yhat = prob.argmax(1)
    mask = y >= 0
    y, yhat, prob = y[mask], yhat[mask], prob[mask]
    m = {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "weighted_f1": f1_score(y, yhat, average="weighted"),
        "accuracy": float((yhat == y).mean()),
    }
    for i, c in enumerate(NICHE_CLASSES):
        yi = (y == i).astype(int)
        if 0 < yi.sum() < len(yi):
            m[f"auroc_{c}"] = roc_auc_score(yi, prob[:, i])
            m[f"auprc_{c}"] = average_precision_score(yi, prob[:, i])
    mm = ~np.isnan(mes_y)
    if len(np.unique(mes_y[mm])) > 1:
        m["mes_auroc"] = roc_auc_score((mes_y[mm] > 0.5).astype(int), mes_p[mm])
        m["mes_pearson"] = float(np.corrcoef(mes_y[mm], mes_p[mm])[0, 1])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/baselines")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    data_list = torch.load(Path(args.data) / "graphs.pt", weights_only=False)
    splits = load_json(cfg["data"]["splits_file"])
    train_data = [d for d in data_list if d.patient in splits["train"]]
    test_data = [d for d in data_list if d.patient in splits["test"]]

    Xtr, ytr, mtr = _stack(train_data)
    Xte, yte, mte = _stack(test_data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    # --- logistic regression ---
    log.info("training logistic regression baseline")
    lr = LogisticBaseline().fit(Xtr, ytr, mtr)
    prob = lr.predict_proba(Xte)
    mes_p = lr.mes_proba(Xte)
    results["logistic"] = _metrics_from_prob(yte, prob, mte, mes_p)
    log.info("logistic: %s", {k: round(v, 3) for k, v in results["logistic"].items()})

    # --- MLP (no graph) ---
    log.info("training MLP baseline (no graph)")
    mlp = MLPBaseline(in_dim=Xtr.shape[1], hidden_dims=cfg["model"]["hidden_dims"], dropout=cfg["model"]["dropout"])
    opt = torch.optim.AdamW(mlp.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    class_w = make_class_weights(train_data)
    lw = cfg["training"]["loss_weights"]
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr)
    mtr_t = torch.from_numpy(mtr).float()
    for epoch in range(1, 121):
        mlp.train()
        out_d = mlp(Xtr_t)
        loss, _ = multitask_loss(out_d, ytr_t, mtr_t, None, class_w, lw)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 30 == 0:
            log.info("MLP epoch %d loss %.3f", epoch, loss.item())
    mlp.eval()
    with torch.no_grad():
        o = mlp(torch.from_numpy(Xte).float())
    prob = torch.softmax(o["niche_logits"], 1).numpy()
    results["mlp"] = _metrics_from_prob(yte, prob, mte, o["mes_prob"].numpy())
    log.info("MLP: %s", {k: round(v, 3) for k, v in results["mlp"].items()})

    save_json(results, out / "baseline_metrics.json")
    log.info("saved -> %s", out / "baseline_metrics.json")


if __name__ == "__main__":
    main()
