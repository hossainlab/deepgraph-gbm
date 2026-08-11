"""External validation on Greenwald et al. (Cell 2024) GBM Visium samples.

Applies the trained model to 13 held-out GBM samples from a different cohort,
platform batch, and institution. Evaluates:
  - niche-map plausibility vs. sample region labels (infiltrating/T1/necrotic/bulk)
  - MES probability vs. Neftel MES signature score

Usage: python scripts/05_external_validation.py --config configs/default.yaml \
    --greenwald-dir <dir> --ckpt models/seed42/best_model.pt --hvg <processed/hvg_genes.json>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepgraph_gbm.data.graph import build_knn_graph
from deepgraph_gbm.models.multitask import DeepGraphGBM
from deepgraph_gbm.training.evaluate import NICHE_CLASSES
from deepgraph_gbm.utils import load_config, save_json, set_seed, setup_logging

log = logging.getLogger("deepgraph_gbm")

# Neftel et al. 2019 MES1 + MES2 meta-module markers (subset robust in Visium)
NEFTEL_MES = [
    "CHI3L1", "ANXA2", "ANXA1", "CD44", "VIM", "MT2A", "C1S", "C1R", "SERPINE1",
    "TIMP1", "FN1", "LGALS1", "LGALS3", "S100A11", "S100A10", "S100A4", "CTSB",
    "CTSD", "NAMPT", "FSTL1", "B2M", "HLA-A", "HLA-B", "HLA-C",
]

REGION_MAP = {
    "inf": "infiltrating", "T1": "contrast_enhancing", "nec": "necrotic", "bulk": "bulk",
}


def _positions_for_sample(sample_dir: Path, pos_files: list[Path]) -> pd.DataFrame | None:
    """Return barcode-indexed positions; fall back to barcode lookup from a
    reference slide layout when a sample lacks its own positions file."""
    if pos_files:
        pf = pos_files[0]
        pos_df = pd.read_csv(pf, header=None if "list" in pf.name else 0)
        pos_df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"][: len(pos_df.columns)]
        return pos_df.set_index("barcode")
    # fallback: map barcodes via a reference slide's layout (Visium v1 is fixed)
    ref = _real_files(sample_dir.parent.glob("GBM_ZH916inf/**/*tissue_positions_list.csv"))
    if not ref:
        return None
    ref_pos = pd.read_csv(ref[0], header=None)
    ref_pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    bc = _real_files(sample_dir.rglob("*_barcodes.tsv"))
    if not bc:
        return None
    barcodes = pd.read_csv(bc[0], header=None, names=["barcode"])
    pos_df = barcodes.merge(ref_pos, on="barcode", how="left")
    if pos_df["array_row"].isna().any():
        return None
    pos_df["in_tissue"] = 1  # filtered matrix = in-tissue spots only
    log.info("%s: positions recovered via barcode lookup (%d spots)", sample_dir.name, len(pos_df))
    return pos_df.set_index("barcode")


def _real_files(paths):
    """Drop macOS AppleDouble stubs (._*) and zero-byte files from a glob."""
    return [p for p in paths if not p.name.startswith("._") and p.stat().st_size > 4096]


def load_greenwald_sample(sample_dir: Path, hvg_genes: list[str]):
    """Load one Greenwald sample -> (X spots x HVGs, coords, sample_name)."""
    h5 = _real_files(sample_dir.rglob("*filtered_feature_bc_matrix.h5"))
    pos = _real_files(sample_dir.rglob("*tissue_positions_list.csv")) or _real_files(
        sample_dir.rglob("*tissue_positions.csv")
    )
    if not h5:
        log.warning("missing h5 in %s, skipping", sample_dir)
        return None
    pos_df = _positions_for_sample(sample_dir, pos)
    if pos_df is None:
        log.warning("missing positions in %s, skipping", sample_dir)
        return None
    adata = sc.read_10x_h5(h5[0])
    adata.var_names_make_unique()
    common = adata.obs_names.intersection(pos_df.index)
    adata = adata[common].copy()
    pos_df = pos_df.loc[common]
    in_tissue = pos_df["in_tissue"].astype(int) == 1
    adata = adata[in_tissue].copy()
    pos_df = pos_df[in_tissue]

    # normalize like SNUH (log1p of CPM-ish)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # project onto HVG set (missing genes -> 0)
    X = np.zeros((adata.n_obs, len(hvg_genes)), dtype=np.float32)
    var_idx = {g: i for i, g in enumerate(adata.var_names)}
    for j, g in enumerate(hvg_genes):
        if g in var_idx:
            col = adata.X[:, var_idx[g]]
            X[:, j] = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else col
    coords = pos_df[["array_row", "array_col"]].values.astype(float)
    return X, coords, sample_dir.name, adata


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--greenwald-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hvg", required=True)
    ap.add_argument("--out", default="results/external")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    import json

    hvg_genes = json.loads(Path(args.hvg).read_text())

    model = DeepGraphGBM(
        in_dim=len(hvg_genes),
        hidden_dims=cfg["model"]["hidden_dims"],
        dropout=cfg["model"]["dropout"],
    )
    ckpt = torch.load(args.ckpt, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    for sample_dir in sorted(Path(args.greenwald_dir).iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith("GBM"):
            continue
        res = load_greenwald_sample(sample_dir, hvg_genes)
        if res is None:
            continue
        X, coords, name, adata = res
        ei, _ = build_knn_graph(coords, k=cfg["data"]["knn_k"])
        o = model(torch.from_numpy(X), ei)
        prob = torch.softmax(o["niche_logits"], 1).numpy()
        yhat = prob.argmax(1)
        mes_prob = o["mes_prob"].numpy()

        # Neftel MES reference score
        present = [g for g in NEFTEL_MES if g in adata.var_names]
        ref = np.asarray(adata[:, present].X.mean(axis=1)).ravel() if present else np.full(adata.n_obs, np.nan)

        region = next((v for k, v in REGION_MAP.items() if k.lower() in name.lower()), "unknown")
        frac = pd.Series(yhat).map(lambda i: NICHE_CLASSES[i]).value_counts(normalize=True)
        summary[name] = {
            "region": region,
            "n_spots": int(len(yhat)),
            "risk": float(o["risk"].item()),
            "mes_pearson_vs_neftel": float(np.corrcoef(mes_prob, ref)[0, 1]) if present else np.nan,
            **{f"frac_{c}": float(frac.get(c, 0.0)) for c in NICHE_CLASSES},
        }
        log.info(
            "%-18s region %-18s spots %4d risk %+.3f MES r=%.3f | %s",
            name, region, len(yhat), o["risk"].item(),
            summary[name]["mes_pearson_vs_neftel"],
            {c: round(frac.get(c, 0), 2) for c in NICHE_CLASSES},
        )
        for i in range(len(yhat)):
            rows.append(
                {
                    "sample": name, "region": region, "spot_idx": i,
                    "pred_niche": NICHE_CLASSES[yhat[i]], "pred_prob": prob[i, yhat[i]],
                    "mes_prob": mes_prob[i], "neftel_mes": ref[i], "risk": o["risk"].item(),
                    "array_row": coords[i, 0], "array_col": coords[i, 1],
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / "greenwald_predictions.csv", index=False)
    save_json(summary, out_dir / "greenwald_summary.json")
    log.info("saved -> %s", out_dir)


if __name__ == "__main__":
    main()
