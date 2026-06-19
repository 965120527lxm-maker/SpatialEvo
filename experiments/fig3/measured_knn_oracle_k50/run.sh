#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/measured_knn_oracle_k50/outputs"
python scripts/fig3/run_fig3_measured_knn.py --k 50 --out_dir /root/autodl-tmp/SpatialEx/experiments/fig3/measured_knn_oracle_k50/outputs
