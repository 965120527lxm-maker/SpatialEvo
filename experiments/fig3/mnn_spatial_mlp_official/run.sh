#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/mnn_spatial_mlp_official/outputs"
for mode in none concat blend delta; do
  python scripts/fig3/run_fig3_mnn_spatial_mlp.py \
    --panel_csv data/panel_split_official.csv \
    --spatial_mode "$mode" \
    --out_dir experiments/fig3/mnn_spatial_mlp_official/outputs/$mode
done

