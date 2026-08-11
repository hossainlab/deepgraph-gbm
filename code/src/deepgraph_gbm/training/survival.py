"""Survival risk: slide-level pseudo-risk targets + TCGA external validation.

No survival data exists for the Visium patients, so Head 3 is trained on a
biologically motivated pseudo-risk derived from niche composition, and the
*signatures* of the predicted niches are validated against TCGA-GBM overall
survival (Cox PH + Kaplan-Meier), following the SNUH atlas paper's approach.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("deepgraph_gbm")

# pseudo-risk weights per niche (biology: cold/necrotic -> worse, hot/normal -> better)
RISK_WEIGHTS = {"immune_cold": 1.0, "necrotic": 0.8, "immune_hot": -0.6, "normal": -0.8}


def slide_pseudo_risk(niche_labels: pd.Series) -> float:
    """Compute a slide-level pseudo-risk from labeled niche composition.

    Weighted mean of niche fractions; excludes 'exclude' spots.
    Returns 0.0 when no labeled spots exist.
    """
    lab = niche_labels[niche_labels != "exclude"]
    if len(lab) == 0:
        return 0.0
    frac = lab.value_counts(normalize=True)
    return float(sum(frac.get(k, 0.0) * w for k, w in RISK_WEIGHTS.items()))


# --- TCGA validation -------------------------------------------------------

NICHE_SIGNATURES = {
    "immune_cold": [
        "CD44", "CHI3L1", "VIM", "ANXA1", "S100A4", "TIMP1", "LGALS1", "LGALS3",
        "S100A6", "NAMPT", "FTL", "FTH1", "SERPINE1", "TGFB1", "IL10", "CD163",
        "MRC1", "CSF1R", "AIF1", "C1QA", "C1QB", "TYROBP",
    ],
    "immune_hot": [
        "CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "IL2RG", "GZMB", "PRF1",
        "NKG7", "LCK", "ZAP70", "CD2", "TRAC", "TRBC1", "TRBC2", "CXCR3", "IFNG",
    ],
    "necrotic": [
        "VEGFA", "HIF1A", "CA9", "SLC2A1", "LDHA", "PGK1", "ENO1", "BNIP3",
        "NDRG1", "ADM", "ANGPTL4", "PDK1", "HK2", "ALDOA", "GAPDH", "P4HA1",
    ],
    "normal": [
        "MBP", "MOG", "PLP1", "MAG", "OLIG1", "OLIG2", "SNAP25", "SYT1",
        "SLC1A2", "SLC1A3", "AQP4", "GFAP", "ALDH1L1", "GJA1", "CLU", "S100B",
    ],
}


def signature_scores(expr: pd.DataFrame, signatures: dict | None = None) -> pd.DataFrame:
    """Mean z-scored expression of each niche signature per sample.

    expr: genes x samples DataFrame (any normalized scale; z-scored per gene).
    Returns samples x signatures DataFrame.
    """
    signatures = signatures or NICHE_SIGNATURES
    z = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1).replace(0, np.nan), axis=0)
    scores = {}
    for name, genes in signatures.items():
        present = [g for g in genes if g in z.index]
        if len(present) < 3:
            log.warning("signature %s: only %d/%d genes present", name, len(present), len(genes))
        scores[name] = z.loc[present].mean(axis=0) if present else pd.Series(np.nan, index=expr.columns)
    return pd.DataFrame(scores)


def risk_from_scores(scores: pd.DataFrame) -> pd.Series:
    """Combine niche signature scores into a single risk score."""
    return (
        scores["immune_cold"] * RISK_WEIGHTS["immune_cold"]
        + scores["necrotic"] * RISK_WEIGHTS["necrotic"]
        + scores["immune_hot"] * RISK_WEIGHTS["immune_hot"]
        + scores["normal"] * RISK_WEIGHTS["normal"]
    )


def cox_validation(scores: pd.DataFrame, clinical: pd.DataFrame, sample_col: str = "sample"):
    """Univariate + multivariate Cox PH and KM split for the risk score.

    clinical must have columns: sample, os_days, os_event, age.
    Returns dict with fitted summaries (lifelines objects not returned).
    """
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test

    df = clinical.merge(scores, left_on=sample_col, right_index=True).dropna(subset=["os_days"])
    df = df[df["os_days"] > 0].copy()
    df["risk"] = risk_from_scores(scores).reindex(df[sample_col]).values
    df = df.dropna(subset=["risk"])
    out = {"n": len(df), "events": int(df["os_event"].sum())}

    cph = CoxPHFitter()
    cph.fit(df[["os_days", "os_event", "risk"]], "os_days", "os_event")
    s = cph.summary.loc["risk"]
    out["cox_univariate"] = {"HR": float(np.exp(s["coef"])), "p": float(s["p"]), "ci": [float(np.exp(s["coef lower 95%"])), float(np.exp(s["coef upper 95%"]))]}
    out["c_index"] = float(cph.concordance_index_)

    if "age" in df and df["age"].notna().sum() > 30:
        d2 = df.dropna(subset=["age"])
        cph2 = CoxPHFitter()
        cph2.fit(d2[["os_days", "os_event", "risk", "age"]], "os_days", "os_event")
        s2 = cph2.summary.loc["risk"]
        out["cox_age_adjusted"] = {"HR": float(np.exp(s2["coef"])), "p": float(s2["p"])}

    med = df["risk"].median()
    hi, lo = df[df["risk"] > med], df[df["risk"] <= med]
    lr = logrank_test(hi["os_days"], lo["os_days"], hi["os_event"], lo["os_event"])
    out["km_logrank_p"] = float(lr.p_value)
    out["km_median_os_high"] = float(KaplanMeierFitter().fit(hi["os_days"], hi["os_event"]).median_survival_time_)
    out["km_median_os_low"] = float(KaplanMeierFitter().fit(lo["os_days"], lo["os_event"]).median_survival_time_)
    out["_df"] = df  # kept for plotting
    return out
