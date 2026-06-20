#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p experiments/fig2/gt512_breast_cancer/outputs
conda run -n spatialex python scripts/fig2/run_fig2_gt.py \
  --hidden_dim 512 \
  --out_dir experiments/fig2/gt512_breast_cancer/outputs \
  --device cuda:0
