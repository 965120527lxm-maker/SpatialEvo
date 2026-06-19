#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/hgnn_cycle_sweep_official/outputs"
python scripts/fig3/run_fig3_hgnn_cycle_sweep.py --resume --quiet --device cuda:0 --out_dir /root/autodl-tmp/SpatialEx/experiments/fig3/hgnn_cycle_sweep_official/outputs
