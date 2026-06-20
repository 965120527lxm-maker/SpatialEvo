#!/usr/bin/env bash
# Poll data/ until big_1.npy and big_2.npy are present (stable size), then launch Fig.4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$ROOT/data"
LOG="$ROOT/logs/fig4/spatialexp_big.log"
OUT="$ROOT/experiments/fig4/spatialexp_big/outputs"
mkdir -p "$ROOT/logs/fig4" "$OUT"

pick() {
  local i="$1"
  for f in \
    "$DATA/Big${i}_uni.npy" "$DATA/Big${i}.npy" \
    "$DATA/big_${i}.npy" "$DATA/big${i}.npy" \
    "$DATA/big_${i}.h5ad" "$DATA/big${i}.h5ad"; do
    [[ -f "$f" ]] && { echo "$f"; return 0; }
  done
  return 1
}

stable_file() {
  local f="$1"
  local s1 s2
  s1=$(stat -c%s "$f")
  sleep 5
  s2=$(stat -c%s "$f")
  [[ "$s1" == "$s2" && "$s1" -gt 1000 ]]
}

echo "[fig4-wait] polling $DATA for big slice files ..."
while true; do
  f1=$(pick 1 || true)
  f2=$(pick 2 || true)
  if [[ -n "${f1:-}" && -n "${f2:-}" ]]; then
    if stable_file "$f1" && stable_file "$f2"; then
      echo "[fig4-wait] found stable inputs:"
      echo "  $f1 ($(stat -c%s "$f1") bytes)"
      echo "  $f2 ($(stat -c%s "$f2") bytes)"
      break
    fi
    echo "[fig4-wait] files present but still growing ..."
  else
    echo "[fig4-wait] waiting ... ($(date -Iseconds))"
  fi
  sleep 20
done

echo "[fig4-wait] probing loader ..."
export ROOT
conda run -n spatialex python - <<PY || { echo "[fig4-wait] loader probe failed"; exit 1; }
import os, sys
root = os.environ["ROOT"]
sys.path.insert(0, os.path.join(root, "scripts", "fig4"))
from load_big_data import load_big_slice
data_root = os.path.join(root, "data")
for i in (1, 2):
    ad = load_big_slice(data_root, i)
    print(f"slice{i}: n_obs={ad.n_obs}, n_vars={ad.n_vars}, he={ad.obsm['he'].shape}")
PY

echo "[fig4-wait] starting SpatialExP_Big ..."
cd "$ROOT"
nohup conda run -n spatialex python scripts/fig4/run_fig4_spatialexp_big.py \
  --device cuda:0 \
  --out_dir "$OUT" \
  > "$LOG" 2>&1 &
echo "[fig4-wait] pid $! log=$LOG"
