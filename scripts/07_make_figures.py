"""Render the full DeepGraph-GBM figure set into results/figures.

Figures (PNG + SVG, Liberation Sans, colorblind-safe):
  1. Niche spatial maps (true vs predicted) per test section
  2. MES probability spatial map per test section
  3. UMAP of spot embeddings colored by niche
  4. ROC curves (4 niche classes, one-vs-rest)
  5. Confusion matrix (normalized)
  6. Kaplan-Meier survival curves (TCGA high vs low risk)

Usage:
  python scripts/07_make_figures.py --config configs/default.yaml \
      --data <processed_dir> --ckpt models/seed7/best_model.pt \
      --survival results/survival/tcga_scores_clinical.csv --out results/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepgraph_gbm.interpret.spatial_maps import (
    NICHE_CLASSES,
    NICHE_COLORS,
    plot_embedding,
    plot_mes_map,
    plot_niche_map,
)
from deepgraph_gbm.models.multitask import DeepGraphGBM
from deepgraph_gbm.training.evaluate import predict_graphs
from deepgraph_gbm.utils import load_config, load_json, set_seed, setup_logging

log = logging.getLogger("deepgraph_gbm")

# publication style: no embedded titles, 300 dpi, editable SVG text, Liberation Sans
matplotlib.rcParams.update({
    "font.family": ["Liberation Sans", "Arimo", "DejaVu Sans"],
    "svg.fonttype": "none",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.dpi": 300,
})


def _save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s.png/.svg", name)


def fig_spatial_maps(preds, data_list, out: Path, data_dir: Path):
    """True vs predicted niche maps + MES map for each test section."""
    coords = {}
    for d in data_list:
        cs = None
        if getattr(d, "pos", None) is not None:
            cs = d.pos.cpu().numpy()
        else:
            npy = data_dir / f"{d.section}_coords.npy"
            if npy.exists():
                cs = np.load(npy)
        coords[d.section] = cs
    for p in preds:
        sec = p["section"]
        cs = coords.get(sec)
        if cs is None:
            continue
        true = pd.Series(p["niche_y"].numpy()).map(
            lambda i: NICHE_CLASSES[i] if i >= 0 else "exclude"
        )
        prob = torch.softmax(p["niche_logits"], 1).numpy()
        pred = pd.Series(prob.argmax(1)).map(lambda i: NICHE_CLASSES[i])
        plot_niche_map(cs, true, f"{sec} — true niche", str(out / f"{sec}_niche_true.png"))
        plot_niche_map(cs, pred, f"{sec} — predicted niche", str(out / f"{sec}_niche_pred.png"))
        plot_mes_map(cs, p["mes_prob"].numpy(), f"{sec} — MES probability", str(out / f"{sec}_mes.png"))
        plt.close("all")


def fig_umap(preds, out: Path):
    import umap

    z = torch.cat([p["z"] for p in preds]).numpy()
    y = torch.cat([p["niche_y"] for p in preds]).numpy()
    labels = pd.Series(y).map(lambda i: NICHE_CLASSES[i] if i >= 0 else "exclude")
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(z)
    plot_embedding(emb, labels, "Spot embeddings (test) by niche", str(out / "umap_niche.png"))
    plt.close("all")


def fig_roc_confusion(preds, out: Path):
    y = torch.cat([p["niche_y"] for p in preds]).numpy()
    logits = torch.cat([p["niche_logits"] for p in preds]).numpy()
    mask = y >= 0
    y, logits = y[mask], logits[mask]
    prob = torch.softmax(torch.from_numpy(logits), 1).numpy()

    # ROC
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for i, c in enumerate(NICHE_CLASSES):
        yi = (y == i).astype(int)
        if 0 < yi.sum() < len(yi):
            fpr, tpr, _ = roc_curve(yi, prob[:, i])
            ax.plot(fpr, tpr, label=f"{c.replace('_',' ')} (AUC {auc(fpr,tpr):.2f})",
                    color=NICHE_COLORS[c], lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, out, "roc_curves")

    # Confusion (normalized)
    yhat = prob.argmax(1)
    cm = confusion_matrix(y, yhat, labels=list(range(4)), normalize="true")
    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), [c.replace("_", " ") for c in NICHE_CLASSES], rotation=45, ha="right")
    ax.set_yticks(range(4), [c.replace("_", " ") for c in NICHE_CLASSES])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Proportion")
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.8)
    fig.tight_layout()
    _save(fig, out, "confusion_matrix")


def fig_km(survival_csv: Path, out: Path):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    df = pd.read_csv(survival_csv).dropna(subset=["os_days", "risk"])
    df = df[df["os_days"] > 0]
    med = df["risk"].median()
    hi, lo = df[df["risk"] > med], df[df["risk"] <= med]
    lr = logrank_test(hi["os_days"], lo["os_days"], hi["os_event"], lo["os_event"])

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for grp, color, lbl in [(hi, "#D55E00", f"High risk (n={len(hi)})"),
                            (lo, "#0072B2", f"Low risk (n={len(lo)})")]:
        kmf = KaplanMeierFitter().fit(grp["os_days"], grp["os_event"], label=lbl)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True, linewidth=1.8)
    ax.set_xlabel("Overall survival (days)"); ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    # log-rank p as in-plot annotation (standard for KM), not a title
    ax.text(0.03, 0.06, f"log-rank p = {lr.p_value:.3f}", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, out, "km_survival")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--survival", default=None, help="tcga_scores_clinical.csv")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data_list = torch.load(Path(args.data) / "graphs.pt", weights_only=False)
    splits = load_json(cfg["data"]["splits_file"])
    test_data = [d for d in data_list if d.patient in splits["test"]]
    log.info("test sections: %s", [d.section for d in test_data])

    model = DeepGraphGBM(
        in_dim=data_list[0].x.shape[1],
        hidden_dims=cfg["model"]["hidden_dims"],
        dropout=cfg["model"]["dropout"],
    )
    ckpt = torch.load(args.ckpt, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds = predict_graphs(model, test_data)

    fig_spatial_maps(preds, test_data, out, Path(args.data))
    fig_umap(preds, out)
    fig_roc_confusion(preds, out)
    if args.survival and Path(args.survival).exists():
        fig_km(Path(args.survival), out)
    log.info("all figures -> %s", out)


if __name__ == "__main__":
    main()
