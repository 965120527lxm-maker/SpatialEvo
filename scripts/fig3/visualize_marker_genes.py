#!/usr/bin/env python3
"""Visualize marker gene spatial predictions: raw kNN MLP vs MNN MLP vs ground truth."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'fig3'))
import run_fig3_mnn_pseudo as mnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    parser.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    parser.add_argument('--panelA_size', type=int, default=150)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--mnn_k', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--genes', nargs='+', default=['CTLA4', 'PTPRC', 'ESR1', 'CLEC14A'])
    parser.add_argument('--slice', choices=['slice1', 'slice2'], default='slice2')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_marker_visualization'))
    return parser.parse_args()


def per_gene_pcc(y_true, y_pred):
    pccs = []
    for g in range(y_true.shape[1]):
        a = y_true[:, g]; b = y_pred[:, g]
        if a.std() < 1e-8 or b.std() < 1e-8:
            pccs.append(np.nan)
        else:
            pccs.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(pccs)


def plot_spatial(ax, coords, values, title, cmap='viridis', s=1, alpha=0.8):
    vmax = np.percentile(values, 99)
    vmin = np.percentile(values, 1)
    scat = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap,
                      vmin=vmin, vmax=vmax, s=s, alpha=alpha, edgecolors='none')
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.axis('off')
    return scat


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    rep1 = mnn.load_slice(args.rep1, args.data_root)
    rep2 = mnn.load_slice(args.rep2, args.data_root)

    genes = rep1.var_names.values
    np.random.shuffle(genes)
    panelA = genes[:args.panelA_size].tolist()
    panelB = genes[args.panelA_size:].tolist()

    Y_A1 = mnn.get_X(rep1, panelA).astype(np.float32)
    Y_B1 = mnn.get_X(rep1, panelB).astype(np.float32)
    Y_A2 = mnn.get_X(rep2, panelA).astype(np.float32)
    Y_B2 = mnn.get_X(rep2, panelB).astype(np.float32)
    HE1 = np.asarray(rep1.obsm['he'], dtype=np.float32)
    HE2 = np.asarray(rep2.obsm['he'], dtype=np.float32)

    # Raw kNN
    print('Training raw kNN MLP...')
    raw_B1 = mnn.build_raw_pseudo(HE1, HE2, Y_B2, k=args.k, device=device)
    raw_A2 = mnn.build_raw_pseudo(Y_B2, raw_B1, Y_A1, k=args.k, device=device)
    model_raw1, model_raw2 = mnn.train_panel_mlp(
        Y_A1, raw_B1, Y_B2, raw_A2,
        Y_A1.shape[1], Y_B2.shape[1], Y_A1.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    # MNN kNN
    print('Training MNN kNN MLP...')
    mnn_B1 = mnn.build_mnn_pseudo(HE1, HE2, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)
    mnn_A2 = mnn.build_mnn_pseudo(Y_B2, mnn_B1, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)
    model_mnn1, model_mnn2 = mnn.train_panel_mlp(
        Y_A1, mnn_B1, Y_B2, mnn_A2,
        Y_A1.shape[1], Y_B2.shape[1], Y_A1.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    def predict(model, x, y_mean, y_std):
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(mnn.zscore(x), dtype=torch.float32, device=device)).cpu().numpy()
        return pred * y_std + y_mean

    yB_mean, yB_std = Y_B2.mean(axis=0), Y_B2.std(axis=0) + 1e-8
    yA_mean, yA_std = Y_A1.mean(axis=0), Y_A1.std(axis=0) + 1e-8
    pred_raw_B1 = predict(model_raw1, Y_A1, yB_mean, yB_std)
    pred_raw_A2 = predict(model_raw2, Y_B2, yA_mean, yA_std)
    pred_mnn_B1 = predict(model_mnn1, Y_A1, yB_mean, yB_std)
    pred_mnn_A2 = predict(model_mnn2, Y_B2, yA_mean, yA_std)

    if args.slice == 'slice2':
        coords = rep2.obsm['spatial']
        gt = Y_A2
        pred_raw = pred_raw_A2
        pred_mnn = pred_mnn_A2
        panel = panelA
        target_name = 'PanelA (Slice2)'
    else:
        coords = rep1.obsm['spatial']
        gt = Y_B1
        pred_raw = pred_raw_B1
        pred_mnn = pred_mnn_B1
        panel = panelB
        target_name = 'PanelB (Slice1)'

    pcc_raw = per_gene_pcc(gt, pred_raw)
    pcc_mnn = per_gene_pcc(gt, pred_mnn)

    n_genes = len(args.genes)
    fig, axes = plt.subplots(n_genes, 3, figsize=(10, 3.2 * n_genes))
    if n_genes == 1:
        axes = axes.reshape(1, -1)

    for i, gene in enumerate(args.genes):
        if gene not in panel:
            print(f'Warning: {gene} not in {target_name}, skipping.')
            continue
        idx = panel.index(gene)
        pr = pcc_raw[idx]
        pm = pcc_mnn[idx]
        plot_spatial(axes[i, 0], coords, gt[:, idx], f'{gene} Ground truth')
        plot_spatial(axes[i, 1], coords, pred_raw[:, idx], f'{gene} raw kNN MLP\nPCC={pr:.3f}')
        scat = plot_spatial(axes[i, 2], coords, pred_mnn[:, idx], f'{gene} MNN MLP\nPCC={pm:.3f}')
        fig.colorbar(scat, ax=axes[i, :], fraction=0.02, pad=0.02)

    fig.suptitle(f'{target_name} marker gene predictions', fontsize=12, y=1.02)
    out_path = os.path.join(args.out_dir, f'marker_genes_{args.slice}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')

    # Save prediction tables for these genes
    df = pd.DataFrame({
        'gene': panel,
        'raw_pcc': pcc_raw,
        'mnn_pcc': pcc_mnn,
        'mnn_gain': pcc_mnn - pcc_raw,
    })
    df.to_csv(os.path.join(args.out_dir, f'marker_gene_pcc_{args.slice}.csv'), index=False)


if __name__ == '__main__':
    main()
