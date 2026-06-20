#!/usr/bin/env bash
# Remaining Fig.3 sweeps: spatial MLP nn (cuda:0), MNN multi-seed (cuda:1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/fig3

wait_proc() {
  local pattern="$1"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    sleep 20
  done
}

echo "[spatial-nn] waiting for latent_mnn on cuda:0..."
wait_proc "run_fig3_latent_mnn.py"

echo "[spatial-nn] blend mode num_neighbors sweep"
for nn in 5 7 10 15; do
  out="experiments/fig3/mnn_spatial_mlp_nn_sweep/outputs/nn_${nn}"
  mkdir -p "$out"
  conda run -n spatialex python scripts/fig3/run_fig3_mnn_spatial_mlp.py \
    --panel_csv data/panel_split_official.csv \
    --spatial_mode blend --num_neighbors "$nn" \
    --out_dir "$out" --device cuda:0
done
echo "[spatial-nn] done"

echo "[seeds] waiting for hgnn lambda sweep on cuda:1..."
wait_proc "hgnn_lambda_cycle_sweep_official"

echo "[seeds] strict MNN multi-seed sweep"
for seed in 0 1 2 3 4; do
  out="experiments/fig3/mnn_pseudo_strict_seeds/outputs/seed_${seed}"
  mkdir -p "$out"
  conda run -n spatialex python scripts/fig3/run_fig3_mnn_pseudo.py \
    --panel_csv data/panel_split_official.csv \
    --seed "$seed" --out_dir "$out" --device cuda:1
done
echo "[seeds] done"
