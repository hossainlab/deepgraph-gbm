# DeepGraph-GBM — Local Reproduction Guide

This bundle contains everything needed to reproduce the DeepGraph-GBM results **offline**:
the complete source code, all raw data, the preprocessed graphs, the trained model, and every
generated figure and table. No download is required unless you want to re-fetch the data from
the original sources.

```
deepgraph-gbm/
├── code/                     # the full source repository (installable package)
│   ├── src/deepgraph_gbm/    # data / models / training / interpret modules
│   ├── scripts/              # numbered pipeline 01 → 07 + run_baselines
│   ├── configs/              # default.yaml, labels.yaml, splits.json
│   ├── tests/                # pytest suite (17 tests)
│   ├── notebooks/            # 03_demo_inference.ipynb (see note below)
│   └── README.md             # project overview & science
├── data/
│   ├── raw/                  # original downloaded data
│   │   ├── snuh/             # SNUH 2026 GBM atlas (matrix, features, meta, coords)
│   │   ├── greenwald/        # 13 Greenwald Cell 2024 Visium samples
│   │   └── tcga/             # TCGA-GBM expression + clinical
│   └── preprocessed/         # graphs.pt, hvg_genes.json, sections_meta.csv,
│                             # per-section *_coords.npy + *_spot_labels.csv
├── results/
│   ├── figures/
│   │   ├── main/             # Figure 1–5 (publication-quality, PNG + SVG)
│   │   └── supplement/       # Figure S1–S4 (per-section niche + MES maps)
│   ├── tables/
│   │   ├── main/             # Table 1–3 (model comparison, test metrics, survival)
│   │   └── supplement/       # Table S1–S5 (baselines, external, per-section, per-spot)
│   └── models/               # best_model.pt + seed7 metrics + history
└── README_REPRODUCE.md       # this file
```

---

## 1. Environment setup

Python 3.10–3.11, CPU-only (a GPU is optional and not required).

**Option A — conda (recommended, matches development):**
```bash
cd code
conda env create -f environment.yml
conda activate deepgraph-gbm
pip install -e .
```

**Option B — pip / venv:**
```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `deepgraph_gbm` package from `code/src` so the scripts can
import it. Versions used to produce these results: PyTorch 2.7, torch_geometric 2.8,
scanpy 1.11, lifelines 0.30, umap-learn, scikit-misc (needed for the `seurat_v3` HVG flavor).

**Sanity check — run the test suite (17 tests, ~30 s):**
```bash
cd code
python -m pytest tests -q
```

---

## 2. Quick start — run inference with the trained model (no retraining)

Everything is already computed. To load the bundled model and reproduce predictions on the
held-out test patients:

```bash
cd code
python scripts/04_evaluate.py \
    --config configs/default.yaml \
    --data ../data/preprocessed \
    --ckpt ../results/models/best_model.pt \
    --out /tmp/eval_out
```
This writes `test_metrics.json` and a per-spot `test_predictions.csv`. Compare against the
bundled `results/tables/`.

Or open the interactive demo (loads graphs + model, runs inference, draws spatial maps):
```bash
cd code
jupyter notebook notebooks/03_demo_inference.ipynb
```
Edit the paths in the first cell to point at `../data/preprocessed` and
`../results/models/best_model.pt`.

---

## 3. Full pipeline from raw data (end-to-end reproduction)

Run from the `code/` directory. Paths below assume the bundle layout (`../data/...`).

```bash
cd code

# (optional) re-download raw data from the original sources instead of using data/raw:
#   bash scripts/01_download_data.sh ../data/raw

# Step 2 — build per-section kNN graphs + niche/MES labels + patient splits + HVGs
#           (reads data/raw/snuh, writes data/preprocessed). ~20–40 min, ~8 GB RAM.
python scripts/02_build_graphs.py \
    --data-dir ../data/raw/snuh \
    --out-dir ../data/preprocessed \
    --config configs/default.yaml \
    --labels configs/labels.yaml \
    --splits-out configs/splits.json

# Step 3 — train the multi-task GNN (uses configs/splits.json). ~30–60 min CPU.
python scripts/03_train.py \
    --config configs/default.yaml \
    --data ../data/preprocessed \
    --out ../results/models --seed 7

# Step 4 — evaluate on held-out test patients
python scripts/04_evaluate.py \
    --config configs/default.yaml \
    --data ../data/preprocessed \
    --ckpt ../results/models/seed7/best_model.pt \
    --out ../results/tables

# Baselines (logistic regression + MLP without graph) for the comparison table
python scripts/run_baselines.py \
    --config configs/default.yaml \
    --data ../data/preprocessed \
    --out ../results/tables

# Step 5 — external validation on the Greenwald cohort
python scripts/05_external_validation.py \
    --config configs/default.yaml \
    --greenwald-dir ../data/raw/greenwald \
    --ckpt ../results/models/seed7/best_model.pt \
    --hvg ../data/preprocessed/hvg_genes.json \
    --out ../results/tables

# Step 6 — TCGA-GBM survival validation (Cox PH + Kaplan–Meier)
python scripts/06_survival_validation.py \
    --tcga-dir ../data/raw/tcga \
    --out ../results/tables

# Step 7 — render all figures (niche maps, MES maps, UMAP, ROC, confusion, KM)
python scripts/07_make_figures.py \
    --config configs/default.yaml \
    --data ../data/preprocessed \
    --ckpt ../results/models/seed7/best_model.pt \
    --survival ../results/tables/tcga_scores_clinical.csv \
    --out ../results/figures
```

> **Note on Step 2 and splits.** `02_build_graphs.py` regenerates `configs/splits.json` with a
> greedy spot-balancing algorithm. The bundled `configs/splits.json` (and the bundled
> `data/preprocessed/graphs.pt`) were produced with a specific stratified split
> (train = SNU17/18/21/23/25/27/43/46, val = SNU24/51, test = SNU16/33/34). To reproduce the
> published numbers exactly, **use the bundled `data/preprocessed/` and `configs/splits.json`
> as-is** and skip re-running Step 2, or restore the bundled `splits.json` before Step 3.

---

## 4. What each result file is

All figures are **publication-quality**: no embedded titles (titles belong in captions), 300 dpi PNG +
editable-text SVG, Liberation Sans, and an Okabe-Ito colorblind-safe palette.

**`results/figures/main/`** — the core story
| Figure | Content |
|---|---|
| `Figure1_umap_niche` | UMAP of test-spot embeddings colored by niche |
| `Figure2_roc_curves` | one-vs-rest ROC for the 4 niche classes (with AUCs) |
| `Figure3_confusion_matrix` | row-normalized confusion matrix |
| `Figure4a/b/c` | representative test section SNU16A: true niche map, predicted niche map, MES probability map |
| `Figure5_km_survival` | TCGA Kaplan–Meier curves, high vs low predicted risk (log-rank p annotated in-plot) |

**`results/figures/supplement/`** — remaining test sections (SNU16B, SNU33A, SNU33B, SNU34A)
| Figure | Content |
|---|---|
| `FigureS1–S4` (a/b/c per section) | true niche map, predicted niche map, MES probability map |

**`results/tables/main/`**
| Table | Content |
|---|---|
| `Table1_model_comparison.json` | GNN (3 seeds) vs MLP vs logistic regression |
| `Table2_test_metrics.json` | full val + test metrics for the trained model (seed 7) |
| `Table3_survival_results.json` | TCGA Cox HR, C-index, KM log-rank for the risk score |

**`results/tables/supplement/`**
| Table | Content |
|---|---|
| `TableS1_baseline_metrics.json` | logistic + MLP baseline metrics |
| `TableS2_greenwald_summary.json` | per-sample external-validation summary (13 samples) |
| `TableS3_sections_meta.csv` | per-section spot counts, split assignment, pseudo-risk |
| `TableS4_greenwald_predictions.csv` | per-spot external predictions + Neftel MES reference |
| `TableS5_tcga_scores_clinical.csv` | per-patient TCGA niche scores + survival covariates |

**`results/models/`**
| File | Content |
|---|---|
| `best_model.pt` | trained multi-task GNN weights (seed 7, ~7 MB) |
| `seed7_metrics.json` | full val + test metrics (same as Table 2) |
| `seed7_history.csv` | per-epoch training/validation loss history |

---

## 5. Headline results (for reference when you re-run)

**Held-out test patients (SNU16, SNU33, SNU34), seed 7:**
- Niche classification: macro-F1 **0.50**, accuracy **0.74**
- Per-class AUROC: immune_cold **0.82**, immune_hot **0.98**, normal **0.73**, necrotic **0.96**
- MES probability: AUROC **0.75**, Pearson r **0.34**
- The GNN (macro-F1 0.50) outperforms the no-graph MLP (0.45) and logistic regression (0.35).

**External validation (Greenwald Cell 2024, 13 samples):** niche maps are region-coherent
(necrotic-region samples → high necrotic fraction; bulk → high necrotic/MES) and MES
probability correlates with the Neftel MES meta-module (mean r ≈ 0.3, up to 0.65).

**TCGA survival validation — an honest negative result.** The niche-composition risk score
does **not** significantly stratify TCGA-GBM overall survival. On the expression-matched cohort
(n = 166; 132 deaths, 34 censored): univariate Cox HR = 1.12 (95% CI 0.92–1.37, p = 0.25),
C-index = 0.52, KM log-rank p = 0.42 (median OS 424 vs 444 days, high vs low risk). Treat the
survival head as **discovery-level / non-significant** — it is trained on a niche-composition
pseudo-risk (no Visium cohort has matched survival), and this TCGA analysis is the validation,
which is negative.

---

## 6. Known limitations (read before extending)

- **Silver-standard labels.** Niche labels are derived from the SNUH atlas's own anatomical-feature
  and metaprogram annotations by the rules in `code/configs/labels.yaml` — not pathologist ground truth.
- **Class imbalance / over-prediction of immune_cold.** immune_hot is ~9% of spots; the model
  over-predicts immune_cold on test (see `confusion_matrix.png`). Per-class thresholds are tuned on
  the validation patients and applied to test (`tune_thresholds` in `training/evaluate.py`).
- **HVG selection matters.** HVGs are selected with scanpy's `seurat_v3` flavor on
  `expm1`-reconstructed counts (the SNUH matrix is log1p-normalized). An earlier raw-dispersion
  approach selected only housekeeping genes and destroyed external-cohort performance — do not revert
  this in `02_build_graphs.py`.
- **Survival head** is validated, not trained, on real outcomes, and the validation is non-significant
  (above).
- **TCGA censoring.** Overall-survival times combine `days_to_death` (events) with
  `days_to_last_follow_up` (censored), the latter read from the GDC `diagnoses` record. The
  expression-matched cohort (n = 166) includes 34 censored patients; right-censoring is handled by the
  Cox/KM estimators in `training/survival.py`. Median follow-up is short (~9 months for censored
  patients), which limits power to detect a survival association.

---

## 7. A note on notebooks

There is **one** notebook: `code/notebooks/03_demo_inference.ipynb` (load graphs → load model →
inference → spatial maps). The project `README.md` previously referenced `01_` and `02_` notebooks;
those were never created — the pipeline is fully covered by the numbered `scripts/`. All result
figures are generated reproducibly by `scripts/07_make_figures.py`, not by a notebook.

---

## Data sources (please cite)

- **SNUH 2026 GBM atlas** — Shah N. et al., *Nat Commun* (2026), PRJNA1337938, via UCSC Cell Browser (`cells.ucsc.edu/?ds=multiomic-gbm`).
- **Greenwald et al.** — *Cell* 187(10):2485–2501 (2024), Zenodo record 12624108 (CC-BY).
- **TCGA-GBM** — NCI Genomic Data Commons.
- **Neftel et al.** — *Cell* 178(4):835–849 (2019) (MES meta-module reference).