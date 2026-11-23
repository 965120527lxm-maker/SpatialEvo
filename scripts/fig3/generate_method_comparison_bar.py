#!/usr/bin/env python3
"""
Generate a bar chart comparing Fig. 3 methods.

Methods compared:
  - MLP + Strict MNN
  - HGNN + MNN v2
  - GT + Cycle
  - HGNN + Cycle (SpatialExP)
  - DeepPT

For each method we show Slice1 PanelB and Slice2 PanelA metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'fig3'))
from exp_paths import output_dir


def load_metrics(path):
    df = pd.read_csv(path)
    # Each metrics file has a single row for the model.
    row = df.iloc[0]
    return {
        'slice1_pcc': float(row['slice1_pcc']),
        'slice1_ssim': float(row['slice1_ssim']),
        'slice2_pcc': float(row['slice2_pcc']),
        'slice2_ssim': float(row['slice2_ssim']),
    }


def main():
    # Method definitions: (display name, metrics file path)
    methods = [
        ('MLP+MNN',
         os.path.join(PROJECT_ROOT, 'experiments/fig3/mnn_pseudo_strict_official/outputs/mnn_metrics.csv'),
         {'method': 'MNN kNN', 'what': 'learned MLP'}),
        ('HGNN+MNN v2',
         os.path.join(PROJECT_ROOT, 'experiments/fig3/conditional_hgnn_mnn_v2_official/outputs/metrics_conditional_hgnn_mnn_v2.csv'),
         None),
        ('GT+Cycle',
         os.path.join(PROJECT_ROOT, 'experiments/fig3/spatialexp_gt_official/outputs/metrics_spatialexp_gt.csv'),
         None),
        ('HGNN+Cycle',
         os.path.join(PROJECT_ROOT, 'experiments/fig3/spatialexp_official/outputs/metrics_spatialexp.csv'),
         None),
        ('DeepPT',
         os.path.join(PROJECT_ROOT, 'experiments/fig3/deeppt_official/outputs/metrics_deeppt.csv'),
         None),
    ]

    records = []
    for name, path, row_selector in methods:
        df = pd.read_csv(path)
        if row_selector is not None:
            mask = np.ones(len(df), dtype=bool)
            for col, val in row_selector.items():
                mask &= (df[col] == val)
            row = df[mask].iloc[0]
        else:
            row = df.iloc[0]
        records.append({
            'method': name,
            'slice1_pcc': float(row['slice1_pcc']),
            'slice1_ssim': float(row['slice1_ssim']),
            'slice2_pcc': float(row['slice2_pcc']),
            'slice2_ssim': float(row['slice2_ssim']),
        })

    df = pd.DataFrame(records)

    out_dir = os.path.join(PROJECT_ROOT, 'experiments/fig3/method_comparison_bar')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'method_comparison.csv'), index=False)
    print('Saved summary CSV to', os.path.join(out_dir, 'method_comparison.csv'))

    # Plotting
    methods = df['method'].tolist()
    x = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PCC subplot
    ax = axes[0]
    bars1 = ax.bar(x - width/2, df['slice1_pcc'], width, label='Slice1 PanelB', color='steelblue')
    bars2 = ax.bar(x + width/2, df['slice2_pcc'], width, label='Slice2 PanelA', color='coral')
    ax.set_ylabel('PCC')
    ax.set_title('Pearson Correlation Coefficient (PCC)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, max(df[['slice1_pcc', 'slice2_pcc']].max()) * 1.15)

    def add_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    add_labels(ax, bars1)
    add_labels(ax, bars2)

    # SSIM subplot
    ax = axes[1]
    bars1 = ax.bar(x - width/2, df['slice1_ssim'], width, label='Slice1 PanelB', color='steelblue')
    bars2 = ax.bar(x + width/2, df['slice2_ssim'], width, label='Slice2 PanelA', color='coral')
    ax.set_ylabel('SSIM')
    ax.set_title('Structural Similarity Index (SSIM)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, max(df[['slice1_ssim', 'slice2_ssim']].max()) * 1.15)

    add_labels(ax, bars1)
    add_labels(ax, bars2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'method_comparison_bar.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print('Saved figure to', out_path)

    # Also print table
    print('\nMethod comparison:')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
