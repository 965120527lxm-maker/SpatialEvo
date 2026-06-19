#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p "/root/autodl-tmp/SpatialEx/experiments/fig3/two_step_alignment_benchmark_official/outputs"
python scripts/fig3/run_fig3_alignment_benchmark.py --panel_csv data/panel_split_official.csv --pipelines mnn_he+mnn_bpanel mnn_he+landmark_ot landmark_ot+landmark_ot pca50_mnn+pca50_mnn_bpanel --out_dir experiments/fig3/two_step_alignment_benchmark_official/outputs
