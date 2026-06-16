#!/usr/bin/env python3
"""Generate supporting figures for Fig.3 panel diagonal integration report."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'image', 'fig3_diagnosis')
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved', path)
    plt.close(fig)


def plot_branch_decomposition():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_decomposed_diagnosis', 'decomposed_metrics.csv'))
    variants = df['variant'].tolist()
    slice1 = df['slice1_pcc'].values
    slice2 = df['slice2_pcc'].values

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(variants))
    width = 0.35
    ax.bar(x - width/2, slice1, width, label='Slice1 PanelB PCC', color='#4c78a8')
    ax.bar(x + width/2, slice2, width, label='Slice2 PanelA PCC', color='#f58518')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.set_ylabel('PCC')
    ax.set_title('Branch decomposition: H&E vs panel-to-panel vs fusion')
    ax.legend()
    save(fig, '01_branch_decomposition.png')


def plot_mnn_sweep():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_mnn_sweep', 'mnn_sweep.csv'))
    df['config'] = df['k'].astype(str) + '/' + df['mnn_k'].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, (ax, col, title) in enumerate(zip(axes, ['slice1_learned_pcc', 'slice2_learned_pcc'], ['Slice1 PanelB', 'Slice2 PanelA'])):
        ax.plot(df['config'], df[col], marker='o', color='#e45756' if idx == 0 else '#59a14f', linewidth=2)
        ax.set_ylim(0, None if idx == 0 else 0.3)
        ax.set_xlabel('(k, mnn_k)')
        ax.set_ylabel('Learned MLP PCC')
        ax.set_title(f'{title}: MNN parameter sensitivity')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for i, v in enumerate(df[col]):
            ax.text(i, v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    save(fig, '02_mnn_sweep.png')


def plot_per_gene_scatter():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_per_gene_pcc', 'per_gene_pcc.csv'))
    colors = {'Slice1': '#4c78a8', 'Slice2': '#f58518'}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, sl in zip(axes, ['Slice1', 'Slice2']):
        d = df[df['slice'] == sl]
        ax.scatter(d['raw_learned_pcc'], d['mnn_learned_pcc'], c=colors[sl], s=20, alpha=0.6, edgecolors='none')
        lim = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot(lim, lim, 'k--', linewidth=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('raw kNN MLP PCC')
        ax.set_ylabel('MNN MLP PCC')
        ax.set_title(f'{sl}: per-gene PCC (n={len(d)})')
        improved = (d['mnn_learned_pcc'] > d['raw_learned_pcc']).sum()
        ax.text(0.05, 0.95, f'improved: {improved}/{len(d)}', transform=ax.transAxes, va='top')
    save(fig, '03_per_gene_scatter.png')


def plot_mnn_gain_volcano():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_per_gene_pcc', 'per_gene_pcc.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, sl in zip(axes, ['Slice1', 'Slice2']):
        d = df[df['slice'] == sl].copy()
        d['gain'] = d['mnn_learned_pcc'] - d['raw_learned_pcc']
        d = d.sort_values('gain', ascending=False)
        colors = ['#59a14f' if g > 0 else '#e45756' for g in d['gain']]
        ax.barh(range(len(d)), d['gain'].values, color=colors, height=1.0)
        ax.invert_yaxis()
        ax.set_xlabel('MNN gain (ΔPCC)')
        ax.set_title(f'{sl}: per-gene MNN gain')
        ax.set_yticks([])
        ax.axvline(0, color='black', linewidth=0.5)
    save(fig, '04_mnn_gain_distribution.png')


def plot_latent_alignment():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_latent_mnn', 'latent_mnn_results.csv'))
    methods = df['method'].tolist()
    slice1 = df['slice1_learned_pcc'].values
    slice2 = df['slice2_learned_pcc'].values

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, slice1, width, label='Slice1 PanelB PCC', color='#4c78a8')
    ax.bar(x + width/2, slice2, width, label='Slice2 PanelA PCC', color='#f58518')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Learned MLP PCC')
    ax.set_title('Shared latent alignment + MNN')
    ax.legend()
    save(fig, '05_latent_alignment.png')


def plot_pipeline_mnn():
    """Schematic of MNN pseudo-label pipeline."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    boxes = [
        (1, 3.5, 'Slice1\nmeasured panel A'),
        (4, 3.5, 'Slice2\nmeasured panel A'),
        (7, 3.5, 'Slice2\ntarget panel B'),
        (4, 1.0, 'MNN matching\n(A1 ↔ A2)'),
        (1, 1.0, 'Pseudo-label\n~Y_B^1'),
    ]
    for x, y, text in boxes:
        box = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1.0, boxstyle="round,pad=0.05", edgecolor='black', facecolor='#e8f0f7')
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9)

    arrows = [
        ((1.7, 3.5), (3.3, 3.5)),
        ((4.7, 3.5), (6.3, 3.5)),
        ((1, 3.0), (1, 1.5)),
        ((7, 3.0), (4.7, 1.5)),
    ]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.set_title('MNN-filtered pseudo-label pipeline (Slice1 → Slice2)', fontsize=12)
    save(fig, '06_mnn_pipeline.png')


def plot_cycle_trap_schematic():
    """Illustrate cycle self-consistency trap."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    labels = ['Y_A', 'Ŷ_B', 'Y_A\'']
    positions = [(1.5, 2), (5, 2), (8.5, 2)]
    for (x, y), lab in zip(positions, labels):
        circle = plt.Circle((x, y), 0.6, color='#f6e8c3', ec='black')
        ax.add_patch(circle)
        ax.text(x, y, lab, ha='center', va='center', fontsize=11)

    ax.annotate('', xy=(4.3, 2), xytext=(2.2, 2), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(7.8, 2), xytext=(5.7, 2), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.text(3.25, 2.5, 'F_{B←A}', ha='center', fontsize=10)
    ax.text(6.75, 2.5, 'F_{A←B}', ha='center', fontsize=10)
    ax.text(5, 0.8, 'Cycle loss can be low while Ŷ_B ≠ true Y_B', ha='center', fontsize=10, style='italic')
    ax.set_title('Cycle self-consistency trap', fontsize=12)
    save(fig, '07_cycle_trap.png')


def plot_signal_contribution():
    """Conceptual bar: where does the useful signal come from?"""
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['H&E branch', 'Panel branch', 'Late fusion']
    slice2 = [0.010, 0.248, 0.177]
    colors = ['#b279a2', '#59a14f', '#79706e']
    bars = ax.barh(categories, slice2, color=colors)
    ax.set_xlabel('Slice2 PanelA PCC')
    ax.set_title('Signal source: H&E vs panel-to-panel')
    for bar, val in zip(bars, slice2):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
    save(fig, '08_signal_contribution.png')


def plot_top_marker_gains():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_per_gene_pcc', 'per_gene_pcc.csv'))
    d = df[df['slice'] == 'Slice2'].nlargest(10, 'mnn_gain')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(d['gene'][::-1], d['mnn_gain'][::-1], color='#59a14f')
    ax.set_xlabel('MNN gain (ΔPCC)')
    ax.set_title('Top 10 Slice2 marker genes improved by MNN')
    save(fig, '09_top_marker_gains.png')


def plot_pcc_distribution():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_per_gene_pcc', 'per_gene_pcc.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sl, color in zip(axes, ['Slice1', 'Slice2'], ['#4c78a8', '#f58518']):
        d = df[df['slice'] == sl]
        ax.hist(d['raw_learned_pcc'], bins=30, alpha=0.5, label='raw kNN', color=color)
        ax.hist(d['mnn_learned_pcc'], bins=30, alpha=0.5, label='MNN', color='#59a14f')
        ax.set_xlabel('Per-gene PCC')
        ax.set_ylabel('Count')
        ax.set_title(f'{sl}: per-gene PCC distribution')
        ax.legend()
    save(fig, '10_pcc_distribution.png')


def main():
    plot_branch_decomposition()
    plot_mnn_sweep()
    plot_per_gene_scatter()
    plot_mnn_gain_volcano()
    plot_latent_alignment()
    plot_pipeline_mnn()
    plot_cycle_trap_schematic()
    plot_signal_contribution()
    plot_top_marker_gains()
    plot_pcc_distribution()
    print(f'\nAll figures saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
