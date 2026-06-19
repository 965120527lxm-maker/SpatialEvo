#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/per_gene_pcc/outputs"
python scripts/fig3/analyze_per_gene_pcc.py --panel_csv data/panel_split_official.csv --out_dir /root/autodl-tmp/SpatialEx/experiments/fig3/per_gene_pcc/outputs
