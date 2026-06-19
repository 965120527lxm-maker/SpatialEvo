#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/conditional_gt_mnn_strict_official/outputs"
python scripts/fig3/run_fig3_panel_split.py --model conditional_gt_mnn --hidden_dim 128 --panel_csv data/panel_split_official.csv --out_dir /root/autodl-tmp/SpatialEx/experiments/fig3/conditional_gt_mnn_strict_official/outputs
