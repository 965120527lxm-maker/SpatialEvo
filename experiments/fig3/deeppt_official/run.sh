#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/deeppt_official/outputs"
python scripts/baselines/run_deeppt_fig3.py --panel_csv data/panel_split_official.csv --out_dir /root/autodl-tmp/SpatialEx/experiments/fig3/deeppt_official/outputs
