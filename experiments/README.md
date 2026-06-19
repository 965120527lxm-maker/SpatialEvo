# Experiments

Reproducible experiment folders: **`README.md`** + **`run.sh`** + **`outputs/`** (local, gitignored).

```text
experiments/
├── fig3/                  # Fig.3 panel diagonal integration (see INDEX.csv)
├── multi_omics/           # Transcriptomics–proteomics integration
├── baselines/             # Full-panel SpatialEx baselines
├── archive/               # Obsolete runs
└── summaries/             # Markdown / CSV summaries
```

## Fig.3 quick start

```bash
conda activate spatialex
cd /path/to/SpatialEx

cat experiments/fig3/INDEX.csv
./experiments/fig3/mnn_pseudo_strict_official/run.sh
```

Shared Python lives under `scripts/fig3/`. Default `--out_dir` uses `experiments/fig3/exp_paths.py`.

## Sync / migrate

```bash
# Regenerate run.sh, README, INDEX.csv
python scripts/fig3/sync_experiments.py

# One-time move from legacy outputs/ layout
python scripts/fig3/migrate_outputs_to_experiments.py
```

Result files under `experiments/**/outputs/` are not committed (large predictions). Re-run `run.sh` or copy metrics locally after cloning.
