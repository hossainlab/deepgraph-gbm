#!/usr/bin/env bash
# Download all datasets for DeepGraph-GBM.
set -euo pipefail
DATA_DIR=${1:-data}

echo "=== SNUH 2026 GBM atlas (UCSC Cell Browser) ==="
python -m deepgraph_gbm.data.download_snuh "$DATA_DIR/snuh"

echo "=== Greenwald et al. Cell 2024 (Zenodo) ==="
python -m deepgraph_gbm.data.download_greenwald "$DATA_DIR/greenwald"

echo "=== TCGA-GBM expression + clinical (GDC) ==="
python -m deepgraph_gbm.data.download_tcga "$DATA_DIR/tcga"

echo "All downloads complete under $DATA_DIR"
