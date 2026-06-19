#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/spatialexp_official/outputs"
python scripts/fig3/run_fig3_panel_split.py --model spatialexp --panel_csv data/panel_split_official.csv --out_dir /root/autodl-tmp/SpatialEx/experiments/fig3/spatialexp_official/outputs
