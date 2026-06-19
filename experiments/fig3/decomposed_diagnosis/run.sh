#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/decomposed_diagnosis/outputs"
python scripts/fig3/diagnose_spatialex_vs_cycle_outputs.py --panel_csv data/panel_split_official.csv --out_dir /root/autodl-tmp/SpatialEx/experiments/fig3/decomposed_diagnosis/outputs
