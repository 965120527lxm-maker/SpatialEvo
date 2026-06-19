#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/mnn_sweep_random/outputs"
python scripts/fig3/run_fig3_mnn_sweep.py --out_dir experiments/fig3/mnn_sweep_random/outputs
