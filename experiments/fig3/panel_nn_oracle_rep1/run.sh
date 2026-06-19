#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/panel_nn_oracle_rep1/outputs"
python scripts/fig3/panel_nn_oracle.py --k 5 --out_dir /root/autodl-tmp/SpatialEx/experiments/fig3/panel_nn_oracle_rep1/outputs
