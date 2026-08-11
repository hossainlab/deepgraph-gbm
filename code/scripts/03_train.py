"""Train DeepGraph-GBM.

Usage: python scripts/03_train.py --config configs/default.yaml --data <processed_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepgraph_gbm.models.multitask import DeepGraphGBM
from deepgraph_gbm.training.evaluate import evaluate_graphs
from deepgraph_gbm.training.train import train_model
from deepgraph_gbm.utils import load_config, load_json, save_json, set_seed, setup_logging

log = logging.getLogger("deepgraph_gbm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True, help="processed dir with graphs.pt")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)

    data_dir = Path(args.data)
    data_list = torch.load(data_dir / "graphs.pt", weights_only=False)
    sections_meta = pd.read_csv(data_dir / "sections_meta.csv")
    splits = load_json(cfg["data"]["splits_file"])

    train_data = [d for d in data_list if d.patient in splits["train"]]
    val_data = [d for d in data_list if d.patient in splits["val"]]
    test_data = [d for d in data_list if d.patient in splits["test"]]
    log.info(
        "graphs: train %d (%d spots) | val %d (%d) | test %d (%d)",
        len(train_data), sum(d.n_spots for d in train_data),
        len(val_data), sum(d.n_spots for d in val_data),
        len(test_data), sum(d.n_spots for d in test_data),
    )

    in_dim = data_list[0].x.shape[1]
    model = DeepGraphGBM(
        in_dim=in_dim,
        hidden_dims=cfg["model"]["hidden_dims"],
        dropout=cfg["model"]["dropout"],
        aggr=cfg["model"]["aggr"],
    )
    log.info("model params: %d", sum(p.numel() for p in model.parameters()))

    out_dir = Path(args.out) / f"seed{seed}"
    result = train_model(model, train_data, val_data, cfg, out_dir)

    # evaluate best checkpoint: tune thresholds on val, apply to test (no test leakage)
    ckpt = torch.load(out_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_metrics = evaluate_graphs(model, val_data)  # tunes thresholds on val
    thr = val_metrics.get("thresholds")
    test_metrics = evaluate_graphs(model, test_data, thr=thr)
    log.info("TEST metrics: %s", {k: round(v, 4) for k, v in test_metrics.items() if isinstance(v, float)})

    save_json(
        {
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "thresholds": thr if isinstance(thr, list) else (thr.tolist() if thr is not None else None),
            "val": {k: v for k, v in val_metrics.items()},
            "test": {k: v for k, v in test_metrics.items()},
            "splits": splits,
        },
        out_dir / "metrics.json",
    )
    # save training history
    pd.DataFrame(result["history"]).to_csv(out_dir / "history.csv", index=False)


if __name__ == "__main__":
    main()
