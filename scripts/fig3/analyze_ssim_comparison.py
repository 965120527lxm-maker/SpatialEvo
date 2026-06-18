#!/usr/bin/env python3
"""Per-gene PCC/SSIM comparison: DeepPT vs SpatialEx-family methods (official split)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import SpatialEx as se


METHODS = {
    'DeepPT': {
        'dir': 'outputs/baselines/fig3_deeppt_official',
        'pred_b1': 'pred_panelB1_deeppt.csv',
        'pred_a2': 'pred_panelA2_deeppt.csv',
        'color': '#e15759',
    },
    'HGNN+Cycle': {
        'dir': 'outputs/conditional/fig3_spatialexp_official',
        'pred_b1': 'pred_panelB1_spatialexp.csv',
        'pred_a2': 'pred_panelA2_spatialexp.csv',
        'color': '#4e79a7',
    },
    'GT+Cycle': {
        'dir': 'outputs/conditional/fig3_spatialexp_gt_official',
        'pred_b1': 'pred_panelB1_spatialexp_gt.csv',
        'pred_a2': 'pred_panelA2_spatialexp_gt.csv',
        'color': '#59a14f',
    },
    'MLP+Strict MNN': {
        'dir': 'outputs/conditional/fig3_mnn_pseudo_strict_official',
        'pred_b1': 'pred_panelB1_mnn.csv',
        'pred_a2': 'pred_panelA2_mnn.csv',
        'color': '#f28e2b',
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    parser.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    parser.add_argument('--panel_csv', type=str, default=os.path.join(PROJECT_ROOT, 'data', 'panel_split_official.csv'))
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'baselines', 'fig3_ssim_comparison'))
    return parser.parse_args()


def load_gt(args):
    panel_df = pd.read_csv(args.panel_csv)
    panelA = panel_df[panel_df['panel'] == 'panelA']['gene'].astype(str).tolist()
    panelB = panel_df[panel_df['panel'] == 'panelB']['gene'].astype(str).tolist()

    rep1 = sc.read_h5ad(os.path.join(args.data_root, args.rep1))
    rep2 = sc.read_h5ad(os.path.join(args.data_root, args.rep2))
    rep1.var_names = rep1.var_names.astype(str)
    rep2.var_names = rep2.var_names.astype(str)
    if 'spatial' not in rep1.obsm:
        rep1.obsm['spatial'] = rep1.obs[['x_centroid', 'y_centroid']].values
    if 'spatial' not in rep2.obsm:
        rep2.obsm['spatial'] = rep2.obs[['x_centroid', 'y_centroid']].values

    gt_b1 = rep1[:, panelB].X
    gt_a2 = rep2[:, panelA].X
    gt_b1 = gt_b1.toarray() if hasattr(gt_b1, 'toarray') else np.asarray(gt_b1)
    gt_a2 = gt_a2.toarray() if hasattr(gt_a2, 'toarray') else np.asarray(gt_a2)

    graph1 = se.pp.Build_graph(
        rep1.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo',
    )
    graph2 = se.pp.Build_graph(
        rep2.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo',
    )
    return panelA, panelB, gt_b1, gt_a2, graph1, graph2


def per_gene_metrics(gt, pred, graph):
    pcc, _ = se.utils.Compute_metrics(gt, pred, metric='pcc', reduce='mean')
    ssim, _ = se.utils.Compute_metrics(gt, pred, metric='ssim', graph=graph, reduce='mean')
    return np.asarray(pcc), np.asarray(ssim)


def load_pred(path):
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float32), df.columns.astype(str).tolist()


def analyze_slice(gt, graph, genes, slice_label, pred_b_or_a, methods, out_dir):
    rows = []
    per_method = {}
    for name, cfg in methods.items():
        pred_path = os.path.join(PROJECT_ROOT, cfg['dir'], cfg[pred_b_or_a])
        if not os.path.exists(pred_path):
            print(f'Skip {name}: missing {pred_path}')
            continue
        pred, pred_genes = load_pred(pred_path)
        assert pred_genes == genes, f'{name} gene order mismatch on {slice_label}'
        pcc, ssim = per_gene_metrics(gt, pred, graph)
        per_method[name] = {'pcc': pcc, 'ssim': ssim}
        rows.append({
            'method': name,
            'slice': slice_label,
            'pcc_mean': float(np.nanmean(pcc)),
            'ssim_mean': float(np.nanmean(ssim)),
            'pcc_median': float(np.nanmedian(pcc)),
            'ssim_median': float(np.nanmedian(ssim)),
        })
    return pd.DataFrame(rows), per_method


def plot_scatter(deeppt, other, other_name, slice_label, metric, out_path):
    x = deeppt[metric]
    y = other[metric]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=18, alpha=0.65, c='#bab0ac', edgecolors='none')
    lim = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (lim[1] - lim[0] + 1e-8)
    lim = (lim[0] - pad, lim[1] + pad)
    ax.plot(lim, lim, 'k--', linewidth=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'DeepPT {metric.upper()}')
    ax.set_ylabel(f'{other_name} {metric.upper()}')
    ax.set_title(f'{slice_label}: per-gene {metric.upper()} (n={len(x)})')

    above = (y > x).sum()
    ax.text(0.05, 0.95, f'{other_name} wins: {above}/{len(x)} genes',
            transform=ax.transAxes, va='top', fontsize=9)
    ax.text(0.05, 0.88, f'mean DeepPT={x.mean():.3f}, {other_name}={y.mean():.3f}',
            transform=ax.transAxes, va='top', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out_path)


def plot_summary_bar(summary_df, out_path):
    methods = summary_df['method'].unique()
    slices = ['Slice1 PanelB', 'Slice2 PanelA']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {m: METHODS[m]['color'] for m in methods if m in METHODS}

    for ax, metric, title in zip(
        axes,
        ['pcc_mean', 'ssim_mean'],
        ['Mean gene-level PCC', 'Mean gene-level SSIM'],
    ):
        x = np.arange(len(methods))
        width = 0.35
        for i, sl in enumerate(slices):
            vals = []
            for m in methods:
                row = summary_df[(summary_df['method'] == m) & (summary_df['slice'] == sl)]
                vals.append(row[metric].iloc[0] if len(row) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width, label=sl, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha='right')
        ax.set_ylabel(metric.replace('_mean', '').upper())
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved', out_path)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    panelA, panelB, gt_b1, gt_a2, graph1, graph2 = load_gt(args)

    summary1, per1 = analyze_slice(gt_b1, graph1, panelB, 'Slice1 PanelB', 'pred_b1', METHODS, args.out_dir)
    summary2, per2 = analyze_slice(gt_a2, graph2, panelA, 'Slice2 PanelA', 'pred_a2', METHODS, args.out_dir)
    summary = pd.concat([summary1, summary2], ignore_index=True)
    summary.to_csv(os.path.join(args.out_dir, 'ssim_pcc_summary.csv'), index=False)

    if summary.empty:
        print('No predictions found; check METHODS paths.')
        return

    plot_summary_bar(summary, os.path.join(args.out_dir, 'ssim_pcc_bar_comparison.png'))

    if 'DeepPT' not in per1 or 'DeepPT' not in per2:
        print('DeepPT predictions missing; aborting scatter plots.')
        return

    for other in [m for m in METHODS if m != 'DeepPT']:
        if other not in per1:
            continue
        for sl, per, tag in [('Slice1 PanelB', per1, 'slice1'), ('Slice2 PanelA', per2, 'slice2')]:
            safe = other.replace('+', '_').replace(' ', '_')
            for metric in ['pcc', 'ssim']:
                plot_scatter(
                    per['DeepPT'], per[other], other, sl, metric,
                    os.path.join(args.out_dir, f'{tag}_{metric}_deeppt_vs_{safe}.png'),
                )

    # DeepPT SSIM advantage table vs HGNN+Cycle
    rows = []
    for sl, per, genes in [
        ('Slice1 PanelB', per1, panelB),
        ('Slice2 PanelA', per2, panelA),
    ]:
        if 'DeepPT' not in per or 'HGNN+Cycle' not in per:
            continue
        dp, hg = per['DeepPT'], per['HGNN+Cycle']
        for i, g in enumerate(genes):
            rows.append({
                'slice': sl,
                'gene': g,
                'deeppt_pcc': dp['pcc'][i],
                'hgnn_pcc': hg['pcc'][i],
                'deeppt_ssim': dp['ssim'][i],
                'hgnn_ssim': hg['ssim'][i],
                'ssim_delta': dp['ssim'][i] - hg['ssim'][i],
                'pcc_delta': dp['pcc'][i] - hg['pcc'][i],
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(os.path.join(args.out_dir, 'per_gene_deeppt_vs_hgnn.csv'), index=False)

    for sl in detail['slice'].unique():
        d = detail[detail['slice'] == sl]
        print(f"\n{sl} — DeepPT vs HGNN+Cycle:")
        print(f"  SSIM: DeepPT wins { (d['ssim_delta'] > 0).sum() }/{len(d)} genes, "
              f"mean delta={d['ssim_delta'].mean():+.4f}")
        print(f"  PCC:  DeepPT wins { (d['pcc_delta'] > 0).sum() }/{len(d)} genes, "
              f"mean delta={d['pcc_delta'].mean():+.4f}")

    print(f"\nSummary saved to {args.out_dir}")


if __name__ == '__main__':
    main()
