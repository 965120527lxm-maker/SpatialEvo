#!/usr/bin/env python3
"""
Fig. 3 panel diagonal integration with HGNN v2 + strict MNN pseudo-labels.

This is the cleaned-up HGNN+MNN variant that:
  - z-score normalises inputs and targets,
  - uses direct cell-level MSE (no pseudo-spot aggregation),
  - uses a linear output head.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'fig3'))
from exp_paths import output_dir

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import SpatialEx as se


def parse_args():
    parser = argparse.ArgumentParser(description='HGNN v2 + Strict MNN (Fig. 3)')
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    parser.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    parser.add_argument('--panel_csv', type=str, default=None,
                        help='CSV with columns gene,panel (panelA/panelB); overrides random split')
    parser.add_argument('--panelA_size', type=int, default=150)
    parser.add_argument('--use_he', action='store_true',
                        help='Concatenate H&E to the measured panel input')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pseudo_k', type=int, default=5)
    parser.add_argument('--mnn_k', type=int, default=20)
    parser.add_argument('--num_neighbors', type=int, default=7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=output_dir('conditional_hgnn_mnn_v2_official'))
    return parser.parse_args()


def load_slice(path, data_root):
    full_path = os.path.join(data_root, path)
    print(f'Loading {full_path}')
    adata = sc.read_h5ad(full_path)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata


def split_panels(adata, panelA_genes, panelB_genes):
    return adata[:, panelA_genes].copy(), adata[:, panelB_genes].copy()


def evaluate(adata_gt, pred, label='Slice'):
    graph = se.pp.Build_graph(
        adata_gt.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo'
    )
    gt_X = adata_gt.X.toarray() if hasattr(adata_gt.X, 'toarray') else np.array(adata_gt.X.copy())
    pred_X = np.array(pred.copy())

    pcc, pcc_reduce = se.utils.Compute_metrics(gt_X, pred_X, metric='pcc')
    ssim, ssim_reduce = se.utils.Compute_metrics(gt_X, pred_X, metric='ssim', graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt_X, pred_X, metric='cmd')

    print(f'[{label}] gene-level PCC: {pcc_reduce:.6f}, SSIM: {ssim_reduce:.6f}, CMD: {cmd_reduce:.6f}')
    return {
        'pcc_per_gene': pcc,
        'pcc': float(pcc_reduce),
        'ssim': float(ssim_reduce),
        'cmd': float(cmd_reduce),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Using device: {device}')

    # 1. Load full slices
    rep1_full = load_slice(args.rep1, args.data_root)
    rep2_full = load_slice(args.rep2, args.data_root)
    print(f'Rep1: {rep1_full.shape}, Rep2: {rep2_full.shape}')

    # 2. Split panel A/B
    if args.panel_csv and os.path.exists(args.panel_csv):
        panel_df = pd.read_csv(args.panel_csv)
        panelA_genes = panel_df[panel_df['panel'] == 'panelA']['gene'].astype(str).tolist()
        panelB_genes = panel_df[panel_df['panel'] == 'panelB']['gene'].astype(str).tolist()
        print(f'Loaded panel split from {args.panel_csv}')
    else:
        genes = rep1_full.var_names.values
        np.random.seed(args.seed)
        np.random.shuffle(genes)
        panelA_genes = genes[:args.panelA_size].tolist()
        panelB_genes = genes[args.panelA_size:].tolist()
    print(f'Panel A: {len(panelA_genes)} genes, Panel B: {len(panelB_genes)} genes')

    rep1_A, rep1_B = split_panels(rep1_full, panelA_genes, panelB_genes)
    rep2_A, rep2_B = split_panels(rep2_full, panelA_genes, panelB_genes)

    # 3. Build hypergraphs
    print('\n=== Building hypergraphs ===')
    graph1 = se.pp.Build_hypergraph_spatial_and_HE(
        rep1_A, args.num_neighbors, graph_kind='spatial', return_type='csr'
    )
    graph2 = se.pp.Build_hypergraph_spatial_and_HE(
        rep2_B, args.num_neighbors, graph_kind='spatial', return_type='csr'
    )

    # 4. Train model
    print('\n=== Training conditional_hgnn_mnn_v2 ===')
    trainer = se.SpatialExP_ConditionalHGNNv2(
        rep1_A, rep2_B, graph1, graph2,
        use_he=args.use_he,
        pseudo_k=args.pseudo_k,
        mnn_k=args.mnn_k,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        standardize=True,
        device=device,
        seed=args.seed,
    )
    trainer.train()

    # 5. Inference
    print('\n=== Inference ===')
    pred_B1 = trainer.predict_panelB_on_slice1(
        rep1_A.obsm['he'],
        np.asarray(rep1_A.X.toarray() if hasattr(rep1_A.X, 'toarray') else rep1_A.X),
        graph1
    )
    pred_A2 = trainer.predict_panelA_on_slice2(
        rep2_B.obsm['he'],
        np.asarray(rep2_B.X.toarray() if hasattr(rep2_B.X, 'toarray') else rep2_B.X),
        graph2
    )

    pred_B1_df = pd.DataFrame(pred_B1, index=rep1_A.obs_names, columns=panelB_genes)
    pred_A2_df = pd.DataFrame(pred_A2, index=rep2_B.obs_names, columns=panelA_genes)
    pred_B1_df.to_csv(os.path.join(args.out_dir, 'pred_panelB1_conditional_hgnn_mnn_v2.csv'))
    pred_A2_df.to_csv(os.path.join(args.out_dir, 'pred_panelA2_conditional_hgnn_mnn_v2.csv'))

    # 6. Evaluation
    print('\n=== Evaluation ===')
    gt_B1 = rep1_B[rep1_A.obs_names, panelB_genes]
    gt_A2 = rep2_A[rep2_B.obs_names, panelA_genes]

    metrics_B1 = evaluate(gt_B1, pred_B1_df.values, label='Slice1 PanelB prediction (HGNN v2)')
    metrics_A2 = evaluate(gt_A2, pred_A2_df.values, label='Slice2 PanelA prediction (HGNN v2)')

    summary = {
        'model': 'conditional_hgnn_mnn_v2',
        'slice1_pcc': metrics_B1['pcc'],
        'slice1_ssim': metrics_B1['ssim'],
        'slice1_cmd': metrics_B1['cmd'],
        'slice2_pcc': metrics_A2['pcc'],
        'slice2_ssim': metrics_A2['ssim'],
        'slice2_cmd': metrics_A2['cmd'],
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.out_dir, 'metrics_conditional_hgnn_mnn_v2.csv'), index=False)
    print(f'\nOutputs saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
