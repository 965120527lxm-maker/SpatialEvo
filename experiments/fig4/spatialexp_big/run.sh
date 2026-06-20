#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
mkdir -p experiments/fig4/spatialexp_big/outputs logs/fig4
PYTHONUNBUFFERED=1 conda run -n spatialex python scripts/fig4/run_fig4_spatialexp_big.py \
  --device cuda:0 \
  --out_dir experiments/fig4/spatialexp_big/outputs \
  "$@"
