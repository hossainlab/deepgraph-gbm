# DeepGraph-GBM: Graph Neural Network for Predicting Immunosuppressive Spatial Niches in Glioblastoma

## The Problem
In GBM, the tumor is not uniform. It has spatial niches:
- Immune Cold: MES cancer cells + suppressive myeloids, no T-cells → worst prognosis
- Immune Hot: T-cells present, fewer myeloids → better prognosis (rare in GBM)
- Normal Brain: Peritumoral tissue
- Necrotic Core: Dead tissue

Currently, identifying these niches requires manual pathologist annotation or slow deconvolution. Can a deep learning model predict the niche type directly from the spatial transcriptomics graph?

## Objectives
Build a Graph Neural Network (GNN) that takes a Visium spatial transcriptomics slide as input and predicts:
- Niche type per spot (immune cold / hot / normal / necrotic)
- MES cancer cell probability per spot
- Patient survival risk from the spatial niche composition
- This combines your three strengths: spatial transcriptomics, GBM biology, and deep learning.

## Dataset
| Dataset         | Source       | What You Get                                     |
| --------------- | ------------ | ------------------------------------------------ |
| **GSE194329**   | GEO          | Visium spatial transcriptomics from GBM patients |
| **GSE237183**   | GEO          | Greenwald Visium dataset (Nature 2024)           |
| **Zenodo/UCSC** | PRJNA1337938 | 2026 Nature Communications multi-omic GBM atlas  |


## Architecture
Input: Spatial transcriptomics graph
├── Node features: Gene expression vector per spot (2,000–3,000 HVGs)
├── Edge features: Physical distance between spots (adjacency matrix)
├── Edge connections: k-nearest neighbors in physical space (k=6)
│
├─> GNN Encoder (3 layers of GraphSAGE or GAT)
│   └── Learns: Spot embeddings that incorporate neighborhood context
│
├─> Multi-Task Heads:
│   ├── Head 1: Niche Classifier (4 classes: cold/hot/normal/necrotic)
│   ├── Head 2: MES Probability Regressor (0–1 per spot)
│   └── Head 3: Patient Survival Predictor (high/low risk)
│
└─> Output: Spatial niche map + survival risk score


## Why GNN?
- A Visium slide is literally a graph (spots = nodes, physical neighbors = edges)
- GNNs learn that a spot's biology depends on its neighbors (cancer cells recruit myeloids from adjacent spots)
- This is biologically interpretable — the model learns the same spatial rules you would manually annotate

