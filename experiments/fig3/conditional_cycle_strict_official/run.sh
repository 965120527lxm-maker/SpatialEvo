#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/conditional_cycle_strict_official/outputs"
python scripts/fig3/run_fig3_conditional_cycle.py --panel_csv data/panel_split_official.csv --no_use_he --out_dir /root/autodl-tmp/SpatialEx/experiments/fig3/conditional_cycle_strict_official/outputs
