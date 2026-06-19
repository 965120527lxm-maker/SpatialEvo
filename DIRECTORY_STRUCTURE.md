# Project Directory Layout

```
SpatialEx/
├── SpatialEx/                 # Core Python package (models, trainers, utils)
│   ├── SpatialEx.py
│   ├── SpatialEx_conditional.py
│   ├── SpatialEx_conditional_mlp.py
│   ├── SpatialEx_improved.py
│   ├── model.py
│   ├── model_improved.py
│   ├── preprocess.py
│   └── utils.py
├── data/                      # Input h5ad files (not tracked by git)
├── experiments/               # Reproducible runs (README + run.sh + local outputs/)
│   ├── fig3/                  # Fig.3: one folder per experiment; see INDEX.csv
│   ├── multi_omics/
│   ├── baselines/
│   └── README.md
├── outputs/                   # Deprecated — see outputs/README.md → experiments/
├── scripts/                   # Executable experiment scripts
│   ├── baselines/             # run_baseline_spatialex.py
│   ├── fig3/                  # Fig. 3 panel-split experiments
│   ├── multi_omics/           # Transcriptomics-proteomics integration
│   └── tests/                 # Smoke / unit tests
├── docs/                      # Documentation and reports
├── curriculum/                # Step-by-step tutorial notebooks
├── tutorials/                 # High-level tutorial notebooks
├── README.md
├── requirements.txt
└── setup.py
```

## Running scripts after reorganization

All scripts use an auto-detected `PROJECT_ROOT`, so they can be run from anywhere:

```bash
conda activate spatialex
cd /path/to/SpatialEx

# Fig. 3 panel split
python scripts/fig3/run_fig3_panel_split.py --model conditional_mlp --mlp_mode measured_pseudo --pseudo_k 50

# Official-split SOTA (MLP + Strict MNN)
./experiments/fig3/mnn_pseudo_strict_official/run.sh

# Measured-panel kNN oracle
python scripts/fig3/run_fig3_measured_knn.py --k 50

# Pseudo-label diagnostic
python scripts/fig3/diagnose_pseudo_labels.py

# Multi-omics
python scripts/multi_omics/run_multi_omics.py --epochs 500

# Smoke tests
python scripts/tests/test_spatialexp_small.py
python scripts/tests/test_translator_hidden_dim.py
```
