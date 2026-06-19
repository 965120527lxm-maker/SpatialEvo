#!/usr/bin/env python3
"""
Organize Fig.3 scripts + outputs into experiments/fig3/<slug>/.

Each experiment folder contains:
  README.md   — description + key metrics pointer
  run.sh      — reproducible command
  outputs/    — result files

Legacy paths under outputs/ are **moved** into experiments/fig3/<slug>/outputs/
(no symlinks). Use migrate_outputs_to_experiments.py for one-time migration.
"""

import os
import shutil
import stat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG3_EXP = os.path.join(PROJECT_ROOT, 'experiments', 'fig3')

# slug, title, script (relative), cli args, legacy output dirs, required result globs
EXPERIMENTS = [
    {
        'slug': 'mnn_pseudo_strict_official',
        'title': 'MLP + Strict MNN pseudo-labels (official panel split)',
        'script': 'scripts/fig3/run_fig3_mnn_pseudo.py',
        'args': '--panel_csv data/panel_split_official.csv',
        'legacy': ['outputs/conditional/fig3_mnn_pseudo_strict_official'],
        'required': ['mnn_metrics.csv'],
    },
    {
        'slug': 'mnn_sweep_official',
        'title': 'MNN k / mnn_k sensitivity (official split)',
        'script': 'scripts/fig3/run_fig3_mnn_sweep.py',
        'args': '--resume --quiet --device cuda:1 --batch_size 512',
        'legacy': ['outputs/conditional/fig3_mnn_sweep_official'],
        'required': ['mnn_sweep.csv'],
    },
    {
        'slug': 'mlp_hidden_sweep_official',
        'title': 'MLP hidden_dim sweep under Strict MNN',
        'script': 'scripts/fig3/run_fig3_mlp_hidden_sweep.py',
        'args': '--quiet --device cuda:1 --batch_size 512',
        'legacy': ['outputs/conditional/fig3_mlp_hidden_sweep_official'],
        'required': ['mlp_hidden_sweep.csv'],
    },
    {
        'slug': 'hgnn_cycle_sweep_official',
        'title': 'HGNN+Cycle num_neighbors sweep (official split)',
        'script': 'scripts/fig3/run_fig3_hgnn_cycle_sweep.py',
        'args': '--resume --quiet --device cuda:0',
        'legacy': ['outputs/conditional/fig3_hgnn_cycle_sweep_official'],
        'required': ['hgnn_cycle_sweep.csv'],
    },
    {
        'slug': 'alignment_benchmark_official',
        'title': 'Step1 alignment methods (OT / MNN / PCA; Step2=mnn_bpanel)',
        'script': 'scripts/fig3/run_fig3_alignment_benchmark.py',
        'args': '--panel_csv data/panel_split_official.csv',
        'legacy': ['outputs/conditional/fig3_alignment_benchmark_official'],
        'required': ['alignment_results.csv'],
    },
    {
        'slug': 'two_step_alignment_benchmark_official',
        'title': 'Step1×Step2 alignment pipeline benchmark',
        'script': 'scripts/fig3/run_fig3_alignment_benchmark.py',
        'args': (
            '--panel_csv data/panel_split_official.csv '
            '--pipelines mnn_he+mnn_bpanel mnn_he+landmark_ot '
            'landmark_ot+landmark_ot pca50_mnn+pca50_mnn_bpanel '
            '--out_dir experiments/fig3/two_step_alignment_benchmark_official/outputs'
        ),
        'legacy': ['outputs/conditional/fig3_two_step_benchmark_official'],
        'required': ['alignment_results.csv'],
    },
    {
        'slug': 'mnn_spatial_mlp_official',
        'title': 'MLP + Strict MNN + within-slice spatial KNN context',
        'script': 'scripts/fig3/run_fig3_mnn_spatial_mlp.py',
        'args': '--panel_csv data/panel_split_official.csv --spatial_mode MODE',
        'legacy': ['outputs/conditional/fig3_mnn_spatial_mlp_official'],
        'required': ['none/metrics_spatial_mlp.csv', 'blend/metrics_spatial_mlp.csv'],
        'run_all': (
            'for mode in none concat blend delta; do\n'
            '  python scripts/fig3/run_fig3_mnn_spatial_mlp.py \\\n'
            '    --panel_csv data/panel_split_official.csv \\\n'
            '    --spatial_mode "$mode" \\\n'
            '    --out_dir experiments/fig3/mnn_spatial_mlp_official/outputs/$mode\n'
            'done\n'
        ),
    },
    {
        'slug': 'latent_mnn_official',
        'title': 'PCA / CORAL H&E latent + MNN (official split)',
        'script': 'scripts/fig3/run_fig3_latent_mnn.py',
        'args': '--panel_csv data/panel_split_official.csv',
        'legacy': ['outputs/conditional/fig3_latent_mnn_official'],
        'required': ['latent_mnn_results.csv'],
    },
    {
        'slug': 'spatialexp_official',
        'title': 'HGNN + Cycle (SpatialExP, official split)',
        'script': 'scripts/fig3/run_fig3_panel_split.py',
        'args': '--model spatialexp --panel_csv data/panel_split_official.csv',
        'legacy': ['outputs/conditional/fig3_spatialexp_official'],
        'required': ['metrics_spatialexp.csv'],
    },
    {
        'slug': 'spatialexp_gt_official',
        'title': 'GT-128 + Cycle (official split)',
        'script': 'scripts/fig3/run_fig3_panel_split.py',
        'args': '--model spatialexp_gt --hidden_dim 128 --panel_csv data/panel_split_official.csv',
        'legacy': ['outputs/conditional/fig3_spatialexp_gt_official'],
        'required': ['metrics_spatialexp_gt.csv'],
    },
    {
        'slug': 'conditional_gt_mnn_strict_official',
        'title': 'GT-128 + Strict MNN',
        'script': 'scripts/fig3/run_fig3_panel_split.py',
        'args': (
            '--model conditional_gt_mnn --hidden_dim 128 '
            '--panel_csv data/panel_split_official.csv'
        ),
        'legacy': ['outputs/conditional/fig3_conditional_gt_mnn_strict_official'],
        'required': ['metrics_conditional_gt_mnn.csv'],
    },
    {
        'slug': 'conditional_hgnn_mnn_strict_official',
        'title': 'HGNN-512 + Strict MNN',
        'script': 'scripts/fig3/run_fig3_panel_split.py',
        'args': '--model conditional_hgnn_mnn --panel_csv data/panel_split_official.csv',
        'legacy': ['outputs/conditional/fig3_conditional_hgnn_mnn_strict_official'],
        'required': ['metrics_conditional_hgnn_mnn.csv'],
    },
    {
        'slug': 'mnn_cycle_strict_official',
        'title': 'MLP + Strict MNN + Cycle',
        'script': 'scripts/fig3/run_fig3_panel_split.py',
        'args': (
            '--model conditional_mnn_cycle_mlp '
            '--panel_csv data/panel_split_official.csv'
        ),
        'legacy': ['outputs/conditional/fig3_mnn_cycle_strict_official'],
        'required': ['metrics_conditional_mnn_cycle_mlp.csv'],
    },
    {
        'slug': 'conditional_cycle_strict_official',
        'title': 'MLP + Cycle only (no MNN)',
        'script': 'scripts/fig3/run_fig3_conditional_cycle.py',
        'args': '--panel_csv data/panel_split_official.csv --no_use_he',
        'legacy': ['outputs/conditional/fig3_conditional_cycle_strict_official'],
        'required': ['metrics_conditional_cycle.csv'],
    },
    {
        'slug': 'per_gene_pcc',
        'title': 'Per-gene PCC: raw kNN vs MNN',
        'script': 'scripts/fig3/analyze_per_gene_pcc.py',
        'args': '--panel_csv data/panel_split_official.csv',
        'legacy': ['outputs/conditional/fig3_per_gene_pcc'],
        'required': ['per_gene_pcc.csv'],
    },
    {
        'slug': 'marker_visualization_slice2',
        'title': 'Marker gene spatial plots (Slice2)',
        'script': 'scripts/fig3/visualize_marker_genes.py',
        'args': '--slice slice2 --panel_csv data/panel_split_official.csv',
        'legacy': ['outputs/conditional/fig3_marker_visualization'],
        'required': ['marker_genes_slice2.png'],
    },
    {
        'slug': 'decomposed_diagnosis',
        'title': 'Branch decomposition diagnosis (H&E vs panel vs fusion)',
        'script': 'scripts/fig3/diagnose_spatialex_vs_cycle_outputs.py',
        'args': '--panel_csv data/panel_split_official.csv',
        'legacy': ['outputs/conditional/fig3_decomposed_diagnosis'],
        'required': ['decomposed_metrics.csv'],
    },
    {
        'slug': 'deeppt_official',
        'title': 'DeepPT baseline (official split)',
        'script': 'scripts/baselines/run_deeppt_fig3.py',
        'args': '--panel_csv data/panel_split_official.csv',
        'legacy': ['outputs/baselines/fig3_deeppt_official'],
        'required': ['metrics_deeppt.csv'],
    },
    {
        'slug': 'ssim_comparison',
        'title': 'DeepPT vs HGNN/GT/MLP SSIM+PCC comparison',
        'script': 'scripts/fig3/analyze_ssim_comparison.py',
        'args': '',
        'legacy': ['outputs/baselines/fig3_ssim_comparison'],
        'required': ['ssim_pcc_summary.csv'],
    },
    {
        'slug': 'intrinsic_dimension',
        'title': 'PR + TwoNN intrinsic dimension estimate',
        'script': 'scripts/fig3/estimate_intrinsic_dimension.py',
        'args': '',
        'legacy': ['outputs/baselines/fig3_id_estimate'],
        'required': ['intrinsic_dimension_pr_twonN.csv'],
    },
    {
        'slug': 'measured_knn_oracle_k5',
        'title': 'Oracle: same-slice measured panel kNN (k=5)',
        'script': 'scripts/fig3/run_fig3_measured_knn.py',
        'args': '--k 5',
        'legacy': ['outputs/oracles/fig3_measured_knn_k5'],
        'required': ['metrics_measured_knn_k5.csv'],
    },
    {
        'slug': 'measured_knn_oracle_k50',
        'title': 'Oracle: same-slice measured panel kNN (k=50)',
        'script': 'scripts/fig3/run_fig3_measured_knn.py',
        'args': '--k 50',
        'legacy': ['outputs/oracles/fig3_measured_knn_k50'],
        'required': ['metrics_measured_knn_k50.csv'],
    },
    {
        'slug': 'panel_nn_oracle_rep1',
        'title': 'Oracle: Rep1 same-slice PanelA kNN → PanelB',
        'script': 'scripts/fig3/panel_nn_oracle.py',
        'args': '--k 5',
        'legacy': ['outputs/oracles/fig3_panel_nn_oracle_rep1'],
        'required': ['oracle_metrics.txt'],
    },
    {
        'slug': 'mnn_sweep_random',
        'title': 'MNN k sweep (random panel split, legacy diagnostic)',
        'script': 'scripts/fig3/run_fig3_mnn_sweep.py',
        'args': '--out_dir experiments/fig3/mnn_sweep_random/outputs',
        'legacy': ['outputs/conditional/fig3_mnn_sweep'],
        'required': ['mnn_sweep.csv'],
    },
]


def _has_required(out_dir, patterns):
    if not os.path.isdir(out_dir):
        return False
    for pat in patterns:
        if not os.path.exists(os.path.join(out_dir, pat)):
            return False
    return True


def _merge_legacy_into(exp_out, legacy_paths):
    os.makedirs(exp_out, exist_ok=True)
    for leg in legacy_paths:
        leg_abs = os.path.join(PROJECT_ROOT, leg)
        if not os.path.isdir(leg_abs):
            continue
        # resolve symlink
        if os.path.islink(leg_abs):
            continue
        for name in os.listdir(leg_abs):
            src = os.path.join(leg_abs, name)
            dst = os.path.join(exp_out, name)
            if os.path.exists(dst):
                continue
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


def _remove_legacy_paths(legacy_paths):
    for leg in legacy_paths:
        leg_abs = os.path.join(PROJECT_ROOT, leg)
        if os.path.islink(leg_abs):
            os.unlink(leg_abs)
        elif os.path.isdir(leg_abs):
            shutil.rmtree(leg_abs, ignore_errors=True)
        elif os.path.isfile(leg_abs):
            os.remove(leg_abs)


def _write_readme(exp_dir, meta):
    readme = os.path.join(exp_dir, 'README.md')
    slug = meta['slug']
    out_rel = f'experiments/fig3/{slug}/outputs'
    lines = [
        f"# {meta['title']}",
        '',
        f"- **Script:** `{meta['script']}`",
        f"- **Outputs:** `{out_rel}/`",
        '',
        '## Run',
        '',
        '```bash',
        'conda activate spatialex',
        'cd /path/to/SpatialEx',
        './run.sh',
        '```',
        '',
    ]
    req = meta.get('required', [])
    if req:
        lines += ['## Expected files', ''] + [f'- `{r}`' for r in req] + ['']
    with open(readme, 'w') as f:
        f.write('\n'.join(lines))


def _write_run_sh(exp_dir, meta):
    out = os.path.join(exp_dir, 'outputs')
    if meta.get('run_all'):
        body = meta['run_all']
    else:
        args = meta['args'].strip()
        if args and '--out_dir' not in args:
            args = f'{args} --out_dir {out}'
        body = f'python {meta["script"]} {args}'.strip()
    script = '\n'.join([
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        'ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"',
        'cd "$ROOT"',
        'mkdir -p "' + out + '"',
        body,
        '',
    ])
    path = os.path.join(exp_dir, 'run.sh')
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)


def sync(dry_run=False):
    missing = []
    index_rows = []
    for meta in EXPERIMENTS:
        slug = meta['slug']
        exp_dir = os.path.join(FIG3_EXP, slug)
        exp_out = os.path.join(exp_dir, 'outputs')
        if not dry_run:
            os.makedirs(exp_dir, exist_ok=True)
            _merge_legacy_into(exp_out, meta['legacy'])
            _write_readme(exp_dir, meta)
            _write_run_sh(exp_dir, meta)
            _remove_legacy_paths(meta['legacy'])
        ok = _has_required(exp_out, meta['required'])
        index_rows.append({
            'slug': slug,
            'title': meta['title'],
            'script': meta['script'],
            'has_results': ok,
            'path': f'experiments/fig3/{slug}/outputs',
        })
        if not ok:
            missing.append(meta)
        print(f"{'OK' if ok else 'MISSING':7} {slug}")

    index_path = os.path.join(FIG3_EXP, 'INDEX.csv')
    if not dry_run:
        import pandas as pd
        pd.DataFrame(index_rows).to_csv(index_path, index=False)
    return missing


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    missing = sync(dry_run=args.dry_run)
    if missing:
        print(f'\n{len(missing)} experiment(s) need re-run — see experiments/fig3/rerun_missing.sh')
    else:
        print('\nAll experiments have required outputs.')
