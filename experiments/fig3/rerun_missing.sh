#!/usr/bin/env bash
# Re-run Fig.3 experiments that have scripts but missing outputs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
conda run -n spatialex bash experiments/fig3/panel_nn_oracle_rep1/run.sh
