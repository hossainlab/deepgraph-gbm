# DeepGraph-GBM

**Graph Neural Network for predicting immunosuppressive spatial niches in glioblastoma from 10x Visium spatial transcriptomics.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](https://pytorch.org/)
[![PyG 2.8](https://img.shields.io/badge/PyG-2.8-3c8dbc.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## The problem

Glioblastoma is spatially heterogeneous. Within one tumor there are distinct niches with very different biology and prognosis:

| Niche | Composition | Prognosis |
|---|---|---|
| **Immune cold** | MES cancer cells + suppressive myeloids, no T cells | worst |
| **Immune hot** | T cells present, fewer myeloids | better (rare in GBM) |
| **Normal brain** | peritumoral tissue | — |
| **Necrotic core** | dead/hypoxic tissue | poor |

Today these niches are identified by manual pathologist annotation or slow deconvolution. DeepGraph-GBM predicts them **directly from the spatial transcriptomics graph** — a Visium slide is already a graph (spots = nodes, physical neighbors = edges), so a GNN learns the same spatial rules a pathologist uses.

## Model

```
Input: Visium slide as a graph
  nodes: spots, features = 3,000 highly variable genes (log-normalized)
  edges: k-nearest neighbors in physical space (k = 6), distance-weighted
        │
        ▼
GraphSAGE encoder (3 layers: 3000 → 256 → 256 → 128)
  spot embeddings that incorporate neighborhood context
        │
        ├─► Head 1: niche classifier      (4 classes, per spot)
        ├─► Head 2: MES probability        (0–1, per spot)
        └─► Head 3: survival risk score    (per slide, attention-pooled)
```

Multi-task loss: `L = 1.0·CE(niche) + 0.5·BCE(MES) + 0.5·MSE(pseudo-risk)`, class-weighted for imbalance.

## Data

| Dataset | Role | Content |
|---|---|---|
| **SNUH 2026 GBM atlas** ([Nat Commun 2026](https://www.nature.com/articles/s41467-026-69716-2), PRJNA1337938, via [UCSC Cell Browser](https://cells.ucsc.edu/?ds=multiomic-gbm)) | **train/val/test** | 100,488 spots, 28 sections, 13 IDH-wildtype GBM patients, per-spot anatomical-feature + metaprogram annotations |
| **Greenwald et al.** ([Cell 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)00320-9), [Zenodo](https://zenodo.org/records/12624108), CC-BY) | **external validation** | 13 GBM Visium samples, region-labeled |
| **TCGA-GBM** (GDC) | **survival validation** | bulk expression + overall survival |

**Niche labels** are silver-standard, derived from the SNUH atlas's own annotations by auditable rules in [`configs/labels.yaml`](configs/labels.yaml): necrotic = PAN/PNZ anatomical features; immune hot = immune metaprogram; normal = concordant leading-edge grey/white matter; immune cold = malignant (CT/IT/hypoxic) metaprograms in tumor regions. Vascular and discordant spots (24.7%) are excluded from classifier training.

**No survival data exists for any Visium GBM cohort**, so Head 3 is trained on a niche-composition pseudo-risk and *validated* on TCGA-GBM overall survival (Cox PH + Kaplan–Meier) — the same strategy used by the SNUH atlas paper.

## Quickstart

```bash
# 1. install
pip install -e .

# 2. download data (~4 GB)
bash scripts/01_download_data.sh data

# 3. build graphs (per-section kNN graphs + labels + patient splits)
python scripts/02_build_graphs.py --data-dir data/snuh --out-dir data/processed

# 4. train
python scripts/03_train.py --config configs/default.yaml --data data/processed

# 5. evaluate on held-out patients
python scripts/04_evaluate.py --data data/processed --ckpt models/seed42/best_model.pt

# 6. external validation (Greenwald cohort)
python scripts/05_external_validation.py --greenwald-dir data/greenwald \
    --ckpt models/seed42/best_model.pt --hvg data/processed/hvg_genes.json

# 7. TCGA survival validation
python scripts/06_survival_validation.py --tcga-dir data/tcga
```

Or run the end-to-end demo on one slide: [`notebooks/03_demo_inference.ipynb`](notebooks/03_demo_inference.ipynb).

## Evaluation design

- **Patient-level split** (8 train / 2 val / 3 test of 13 GBM patients, FFPE-stratified, fixed seed in `configs/splits.json`) — no spot/patient leakage.
- HVGs selected on train patients only.
- Metrics: macro/weighted F1, per-class AUROC/AUPRC, confusion matrix; MES Pearson + AUROC; TCGA C-index, Cox HR, KM log-rank.
- Baselines: MLP (no graph) and logistic regression, to quantify the spatial-context gain.
- External cohort: Greenwald Cell 2024 (different institution, fresh-frozen).

## Results

Metrics are written to `models/seed*/metrics.json` and `results/test_metrics.json` by the scripts above, and all result figures are rendered by `scripts/07_make_figures.py`. In this bundle, precomputed metrics, predictions, and figures are under `../results/`.

## Repository layout

```
src/deepgraph_gbm/
├── data/        downloaders (SNUH/Greenwald/TCGA), label rules, kNN graph, PyG dataset
├── models/      GraphSAGE encoder, task heads, multi-task model, baselines
├── training/    train loop, evaluation, survival validation
└── interpret/   spatial niche maps, neighbor-importance
scripts/         01_download → 07_make_figures (numbered pipeline) + run_baselines
configs/         default.yaml (hyperparameters), labels.yaml (label rules), splits.json
tests/           pytest: labels, graph, model, splits
notebooks/       03_demo_inference.ipynb (end-to-end demo on one slide)
```

## Limitations

- Niche labels are **silver-standard** (derived from atlas annotations, not pathologist ground truth).
- Survival head is validated, not trained, on real outcomes; treat as discovery-level.
- CPU-friendly by design (GraphSAGE, ≤256 hidden); a GPU enables larger variants.

## Citation

If you use this code, please cite the data sources:

- Shah N. et al. *A spatially resolved human glioblastoma atlas reveals distinct cellular and molecular patterns of anatomical niches.* Nat Commun (2026). https://www.nature.com/articles/s41467-026-69716-2
- Greenwald A.C., Galili Darnell N., Hoefflin R. et al. *Integrative spatial analysis reveals a multi-layered organization of glioblastoma.* Cell 187(10):2485–2501 (2024).
- Neftel C. et al. *An integrative model of cellular states, plasticity, and genetics for glioblastoma.* Cell 178(4):835–849 (2019).

## License

MIT — see [LICENSE](LICENSE).
