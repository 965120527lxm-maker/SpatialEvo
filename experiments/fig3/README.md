# Fig.3 Experiments Index

Each subdirectory is one experiment: **`README.md`** + **`run.sh`** + **`outputs/`**.

All Fig.3 results live here; `outputs/` at repo root is deprecated (see `outputs/README.md`).

## Quick start

```bash
conda activate spatialex
cd /root/autodl-tmp/SpatialEx

# List all experiments and whether outputs exist
cat experiments/fig3/INDEX.csv

# Run one experiment
./experiments/fig3/mnn_pseudo_strict_official/run.sh

# Re-sync after moving files manually
python scripts/fig3/sync_experiments.py
```

## Official split — main results

| Experiment folder | What it runs | Key metric file |
|-------------------|--------------|-----------------|
| `mnn_pseudo_strict_official/` | MLP + Strict MNN | `outputs/mnn_metrics.csv` |
| `mnn_sweep_official/` | MNN k sweep | `outputs/mnn_sweep.csv` |
| `mlp_hidden_sweep_official/` | MLP hidden_dim sweep | `outputs/mlp_hidden_sweep.csv` |
| `hgnn_cycle_sweep_official/` | HGNN+Cycle nn sweep | `outputs/hgnn_cycle_sweep.csv` |
| `alignment_benchmark_official/` | Step1 alignment × MNN step2 | `outputs/alignment_results.csv` |
| `two_step_alignment_benchmark_official/` | Step1×Step2 pipelines | `outputs/alignment_results.csv` |
| `mnn_spatial_mlp_official/` | Spatial KNN + MNN MLP | `outputs/{none,blend,...}/metrics_spatial_mlp.csv` |
| `spatialexp_official/` | HGNN + Cycle | `outputs/metrics_spatialexp.csv` |
| `spatialexp_gt_official/` | GT-128 + Cycle | `outputs/metrics_spatialexp_gt.csv` |
| `deeppt_official/` | DeepPT baseline | `outputs/metrics_deeppt.csv` |
| `ssim_comparison/` | Method SSIM/PCC bar charts | `outputs/ssim_pcc_summary.csv` |

## Shared code (not per-experiment)

- `scripts/fig3/*.py` — Python entry points (referenced by each `run.sh`)
- `scripts/fig3/alignment_ot.py` — OT helpers for alignment benchmark
- `scripts/fig3/sync_experiments.py` — migrate outputs + regenerate run scripts

## Default output paths in code

New runs should use:

```python
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'fig3'))
from exp_paths import output_dir
# ...
parser.add_argument('--out_dir', default=output_dir('my_experiment_slug'))
```

See `experiments/fig3/exp_paths.py`.
