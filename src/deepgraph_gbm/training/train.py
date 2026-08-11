"""Training loop with early stopping and checkpointing."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from ..models.multitask import DeepGraphGBM, multitask_loss

log = logging.getLogger("deepgraph_gbm")


def make_class_weights(train_data: list, n_classes: int = 4, boost: dict | None = None) -> torch.Tensor:
    """Balanced weights, optionally boosted for rare-but-critical classes.

    boost: {class_index: multiplier}. immune_hot (1) is rare and clinically key,
    so we up-weight it beyond balanced to avoid collapse to the majority class.
    """
    y = torch.cat([d.niche_y for d in train_data]).numpy()
    y = y[y >= 0]
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=y).astype(np.float32)
    for idx, mult in (boost or {}).items():
        w[idx] *= mult
    return torch.from_numpy(w).float()


def train_model(
    model: DeepGraphGBM,
    train_data: list,
    val_data: list,
    cfg: dict,
    out_dir: str | Path,
    device: str = "cpu",
) -> dict:
    """Full-graph training, one section per optimizer step."""
    from .evaluate import evaluate_graphs

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["training"]
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["max_epochs"])
    class_w = (
        make_class_weights(train_data, boost=tcfg.get("class_boost")).to(device)
        if tcfg.get("class_weighted_ce")
        else None
    )
    lw = tcfg["loss_weights"]

    best_val = -np.inf
    best_epoch = -1
    patience_left = tcfg["patience"]
    history = []

    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        order = np.random.permutation(len(train_data))
        ep_loss, ep_parts = 0.0, {}
        for i in order:
            d = train_data[i].to(device)
            out = model(d.x, d.edge_index)
            loss, parts = multitask_loss(
                out, d.niche_y, d.mes_y, d.risk_y, class_w, lw
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            for k, v in parts.items():
                ep_parts[k] = ep_parts.get(k, 0.0) + v
        sched.step()

        val_metrics = evaluate_graphs(model, val_data, device)
        history.append({"epoch": epoch, "train_loss": ep_loss, **val_metrics})
        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "epoch %3d | loss %.3f (%s) | val macroF1 %.3f | %.1fs",
                epoch, ep_loss,
                " ".join(f"{k}={v:.2f}" for k, v in ep_parts.items()),
                val_metrics["macro_f1"], time.time() - t0,
            )

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_left = tcfg["patience"]
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val": val_metrics},
                out_dir / "best_model.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("early stop at epoch %d (best %d, macroF1 %.3f)", epoch, best_epoch, best_val)
                break

    return {"best_epoch": best_epoch, "best_val_macro_f1": best_val, "history": history}
