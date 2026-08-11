"""Evaluate a trained checkpoint and produce metrics + prediction tables.

Usage: python scripts/04_evaluate.py --config configs/default.yaml --data <processed_dir> --ckpt models/seed42/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepgraph_gbm.models.multitask import DeepGraphGBM
from deepgraph_gbm.training.evaluate import NICHE_CLASSES, classification_metrics, mes_metrics, predict_graphs
from deepgraph_gbm.utils import load_config, load_json, save_json, set_seed, setup_logging

log = logging.getLogger("deepgraph_gbm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    data_list = torch.load(Path(args.data) / "graphs.pt", weights_only=False)
    splits = load_json(cfg["data"]["splits_file"])
    test_data = [d for d in data_list if d.patient in splits["test"]]

    model = DeepGraphGBM(
        in_dim=data_list[0].x.shape[1],
        hidden_dims=cfg["model"]["hidden_dims"],
        dropout=cfg["model"]["dropout"],
    )
    ckpt = torch.load(args.ckpt, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    preds = predict_graphs(model, test_data)
    m = classification_metrics(preds)
    m.update(mes_metrics(preds))
    log.info("test metrics: %s", {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(m, out / "test_metrics.json")

    # per-spot prediction table
    rows = []
    for p in preds:
        prob = torch.softmax(p["niche_logits"], 1).numpy()
        yhat = prob.argmax(1)
        for i in range(len(yhat)):
            rows.append(
                {
                    "section": p["section"],
                    "patient": p["patient"],
                    "spot_idx": i,
                    "true_niche": NICHE_CLASSES[p["niche_y"][i]] if p["niche_y"][i] >= 0 else "exclude",
                    "pred_niche": NICHE_CLASSES[yhat[i]],
                    "pred_prob": prob[i, yhat[i]],
                    "mes_prob": p["mes_prob"][i].item(),
                    "risk": p["risk"],
                }
            )
    pd.DataFrame(rows).to_csv(out / "test_predictions.csv", index=False)
    log.info("saved predictions -> %s", out / "test_predictions.csv")


if __name__ == "__main__":
    main()
