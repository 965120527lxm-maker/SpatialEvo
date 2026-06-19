#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/mlp_hidden_sweep_official/outputs"
python scripts/fig3/run_fig3_mlp_hidden_sweep.py --quiet --device cuda:1 --batch_size 512 --out_dir /root/autodl-tmp/SpatialEx/experiments/fig3/mlp_hidden_sweep_official/outputs
