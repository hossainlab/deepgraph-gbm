"""TCGA-GBM survival validation of the niche-composition risk score.

Scores TCGA-GBM bulk tumors with the four niche signatures, combines them into
the risk score, and tests association with overall survival (Cox PH + KM).

Usage: python scripts/06_survival_validation.py --tcga-dir <dir> --out results/survival
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepgraph_gbm.training.survival import cox_validation, signature_scores
from deepgraph_gbm.utils import save_json, setup_logging

log = logging.getLogger("deepgraph_gbm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga-dir", required=True)
    ap.add_argument("--out", default="results/survival")
    args = ap.parse_args()

    setup_logging()
    tcga = Path(args.tcga_dir)
    expr = pd.read_csv(tcga / "tcga_gbm_expression.tsv.gz", sep="\t", index_col=0)
    clin = pd.read_csv(tcga / "tcga_gbm_clinical.tsv", sep="\t")
    log.info("expression %s | clinical n=%d", expr.shape, len(clin))

    scores = signature_scores(expr)
    res = cox_validation(scores, clin, sample_col="sample")
    df = res.pop("_df")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(res, out / "tcga_survival_results.json")
    df.to_csv(out / "tcga_scores_clinical.csv", index=False)
    for k, v in res.items():
        log.info("%s: %s", k, v)


if __name__ == "__main__":
    main()
