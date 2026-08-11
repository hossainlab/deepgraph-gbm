"""Build per-section PyG graphs from the SNUH atlas download.

Steps:
  1. Load matrix.mtx.gz (genes x spots, log1p-normalized), features, meta, coords.
  2. Keep GBM sections (IDH-wildtype glioblastoma) only.
  3. Assign niche + MES labels from configs/labels.yaml.
  4. Patient-level train/val/test split (FFPE-stratified, fixed seed).
  5. Select top-N HVGs on TRAIN sections only (no leakage).
  6. Build kNN graph per section; save PyG data list + metadata.

Usage: python scripts/02_build_graphs.py --data-dir <snuh_dir> --out-dir <processed_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepgraph_gbm.data.dataset import build_graph_data
from deepgraph_gbm.data.labels import assign_niche_labels, mes_target
from deepgraph_gbm.training.survival import slide_pseudo_risk
from deepgraph_gbm.utils import load_config, load_label_rules, set_seed, setup_logging

log = logging.getLogger("deepgraph_gbm")


def make_patient_splits(meta_gbm: pd.DataFrame, seed: int = 42) -> dict:
    """FFPE-stratified patient-level split: 8 train / 2 val / 3 test.

    Patients with any Frozen sections (SNU18, SNU21) are pinned to train so the
    model sees both preservation types; remaining FFPE-only patients are split
    to balance total spots and niche-class coverage.
    """
    rng = np.random.default_rng(seed)
    patients = sorted(meta_gbm["sample"].unique())
    has_frozen = meta_gbm.groupby("sample")["source"].apply(lambda s: (s == "Frozen").any())
    frozen_patients = sorted(has_frozen[has_frozen].index.tolist())
    ffpe_only = [p for p in patients if p not in frozen_patients]

    spots = meta_gbm.groupby("sample").size()
    shuffled = rng.permutation(ffpe_only).tolist()

    # greedy assignment to balance spot counts: test gets 3, val gets 2
    test, val = [], []
    for p in shuffled:
        if len(test) < 3 and sum(spots.get(x, 0) for x in test) < 0.25 * spots.sum():
            test.append(p)
        elif len(val) < 2:
            val.append(p)
    train = [p for p in patients if p not in test and p not in val]
    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "all_patients": patients,
        "frozen_patients_in_train": frozen_patients,
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--labels", default="configs/labels.yaml")
    ap.add_argument("--splits-out", default="configs/splits.json")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    rules = load_label_rules(args.labels)
    set_seed(cfg["seed"])
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load raw files ----
    log.info("loading metadata")
    meta = pd.read_csv(data_dir / "meta.tsv", sep="\t", index_col=0)
    log.info("loading coordinates")
    coords = pd.read_csv(
        data_dir / "Visium.coords.tsv.gz", sep="\t", index_col=0, header=None, names=["x", "y"]
    )
    genes = pd.read_csv(data_dir / "features.tsv.gz", header=None)[0].astype(str)
    log.info("loading expression matrix (genes x spots) ...")
    X = sio.mmread(data_dir / "matrix.mtx.gz")
    X = sp.csc_matrix(X)  # CSC: cheap column slicing (spots are columns)
    log.info("matrix: %s, genes: %d, meta spots: %d", X.shape, len(genes), len(meta))
    assert X.shape == (len(genes), len(meta)), "matrix/meta mismatch"

    # align coords to meta order
    coords = coords.reindex(meta.index)
    assert coords.notna().all().all(), "missing coordinates for some spots"

    # ---- GBM only ----
    gbm_mask = (meta["Diagnosis"] == "Glioblastoma multiforme").values
    gbm_idx = np.where(gbm_mask)[0]
    meta_g = meta.iloc[gbm_idx].copy()
    X_gbm = X[:, gbm_idx]  # CSC column slice (cheap); genes x GBM spots
    del X  # free the full matrix
    log.info("GBM spots: %d across %d sections", len(meta_g), meta_g["orig.ident"].nunique())

    # ---- labels ----
    niche = assign_niche_labels(meta_g, rules)
    mes = mes_target(meta_g, rules)
    meta_g["niche"] = niche.values
    meta_g["mes"] = mes.values
    log.info("niche distribution:\n%s", niche.value_counts().to_string())

    # ---- patient splits ----
    splits = make_patient_splits(meta_g, seed=cfg["seed"])
    Path(args.splits_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.splits_out, "w") as f:
        json.dump(splits, f, indent=2)
    log.info("splits: train=%s val=%s test=%s", splits["train"], splits["val"], splits["test"])

    # ---- HVG selection on TRAIN sections only (seurat_v3 on reconstructed counts) ----
    # The SNUH matrix is log1p-normalized. Raw dispersion (var/mean) on log data
    # ranks housekeeping genes (MT-*, RPL*, GAPDH) at the top and excludes every
    # informative GBM marker. We reconstruct approximate counts via expm1 and run
    # scanpy's seurat_v3 HVG (NB model on counts), which correctly surfaces
    # biologically variable genes (IG*, MMP9, CCL18, EGFR-pathway, MBP...).
    import scanpy as sc
    import anndata as ad

    train_mask = meta_g["sample"].isin(splits["train"]).values
    train_sections = meta_g.loc[train_mask, "orig.ident"].unique().tolist()
    log.info("selecting %d HVGs on %d train spots (%d sections)", cfg["data"]["n_hvgs"], int(train_mask.sum()), len(train_sections))
    train_cols = np.where(train_mask)[0]
    Xtr = X_gbm[:, train_cols].T.tocsr()  # spots x genes (log-normalized)
    counts = Xtr.copy()
    counts.data = np.expm1(counts.data)  # approximate normalized counts for seurat_v3
    ad_tr = ad.AnnData(X=counts, var=pd.DataFrame(index=genes.tolist()))
    sc.pp.highly_variable_genes(
        ad_tr, n_top_genes=cfg["data"]["n_hvgs"], flavor="seurat_v3", subset=False
    )
    hvg_genes = ad_tr.var.index[ad_tr.var["highly_variable"]].tolist()
    gene_to_idx = {g: i for i, g in enumerate(genes.tolist())}
    hvg_idx = np.array([gene_to_idx[g] for g in hvg_genes])
    with open(out_dir / "hvg_genes.json", "w") as f:
        json.dump(hvg_genes, f)
    log.info("HVGs selected; top: %s", hvg_genes[:8])
    del Xtr, counts, ad_tr

    # ---- per-section graphs ----
    sections = meta_g["orig.ident"].unique().tolist()
    data_list, meta_rows = [], []
    for sec in sections:
        sec_mask = (meta_g["orig.ident"] == sec).values
        sec_meta = meta_g.iloc[np.where(sec_mask)[0]]
        Xs = X_gbm[:, sec_mask][hvg_idx].T.tocsr()  # spots x HVGs (CSR)
        cs = coords.loc[sec_meta.index].values
        patient = sec_meta["sample"].iloc[0]
        risk = slide_pseudo_risk(sec_meta["niche"])
        d = build_graph_data(
            Xs, cs, sec_meta["niche"], sec_meta["mes"], risk, sec, patient, k=cfg["data"]["knn_k"]
        )
        data_list.append(d)
        np.save(out_dir / f"{sec}_coords.npy", cs)  # for spatial plotting
        sec_meta[["niche", "mes"]].to_csv(out_dir / f"{sec}_spot_labels.csv")
        meta_rows.append(
            {
                "section": sec,
                "patient": patient,
                "source": sec_meta["source"].iloc[0],
                "n_spots": len(sec_meta),
                "n_labeled": int((sec_meta["niche"] != "exclude").sum()),
                "risk": risk,
                "split": (
                    "train" if patient in splits["train"]
                    else "val" if patient in splits["val"] else "test"
                ),
            }
        )
        log.info(
            "section %-14s patient %-6s spots %4d edges %6d risk %+.3f",
            sec, patient, d.n_spots, d.edge_index.shape[1], risk,
        )

    torch.save(data_list, out_dir / "graphs.pt")
    pd.DataFrame(meta_rows).to_csv(out_dir / "sections_meta.csv", index=False)
    log.info("saved %d graphs -> %s", len(data_list), out_dir)


if __name__ == "__main__":
    main()
