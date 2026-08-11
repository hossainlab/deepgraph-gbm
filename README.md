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
flowchart TD

    A["Spatial Transcriptomics Graph"]

    A --> B["Node Features<br/>Gene expression vector<br/>2,000–3,000 HVGs"]
    A --> C["Edge Features<br/>Physical distance<br/>Adjacency matrix"]
    A --> D["Edge Connections<br/>k-NN in physical space<br/>k = 6"]

    B --> E["GNN Encoder"]
    C --> E
    D --> E

    E --> E1["GraphSAGE / GAT<br/>3 Layers"]
    E1 --> F["Spot Embeddings<br/>Neighborhood-aware representations"]

    F --> G["Multi-Task Learning Heads"]

    G --> H["Head 1<br/>Niche Classifier"]
    G --> I["Head 2<br/>MES Probability Regressor"]
    G --> J["Head 3<br/>Patient Survival Predictor"]

    H --> H1["4 Classes<br/>Cold / Hot / Normal / Necrotic"]
    I --> I1["MES Probability<br/>0–1 per spot"]
    J --> J1["Survival Risk<br/>High / Low"]

    H1 --> K["Spatial Niche Map"]
    I1 --> K
    J1 --> L["Survival Risk Score"]

    K --> M["Final Output"]
    L --> M


## Why GNN?
- A Visium slide is literally a graph (spots = nodes, physical neighbors = edges)
- GNNs learn that a spot's biology depends on its neighbors (cancer cells recruit myeloids from adjacent spots)
- This is biologically interpretable — the model learns the same spatial rules you would manually annotate

