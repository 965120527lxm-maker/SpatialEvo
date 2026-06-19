#!/usr/bin/env python3
"""
Non-parametric measured-panel conditional oracle for Fig. 3.

For each cell, the missing panel is imputed from cross-slice nearest neighbours
in the measured-panel space:

* Slice 1 (measured A -> missing B): find k A-nearest neighbours in slice 2,
  transfer their B expression.
* Slice 2 (measured B -> missing A): first build pseudo B for slice 1 using the
  same A-matching, then find k pseudo-B-nearest neighbours in slice 1 and
  transfer their A expression.

This is an oracle because the other slice's held-out panel is used to build
pseudo-labels, but the held-out panel of the target slice is never used.
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
import scipy.sparse
import scanpy as sc
import torch
import SpatialEx as se


def batched_cosine_knn_pseudo(query, ref, y_ref, k=5, batch_size=4096, device='cuda'):
    query = torch.as_tensor(query, dtype=torch.float32, device=device)
    ref = torch.as_tensor(ref, dtype=torch.float32, device=device)
    y_ref = torch.as_tensor(y_ref, dtype=torch.float32, device=device)
    query = query / (query.norm(dim=1, keepdim=True) + 1e-8)
    ref = ref / (ref.norm(dim=1, keepdim=True) + 1e-8)
    k = min(k, ref.shape[0])
    n_query = query.shape[0]
    pseudo = torch.zeros(n_query, y_ref.shape[1], device='cpu', dtype=torch.float32)
    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        sim = query[start:end] @ ref.T
        topk_sim, topk_idx = torch.topk(sim, k, dim=1)
        weights = topk_sim / (topk_sim.sum(dim=1, keepdim=True) + 1e-8)
        pseudo[start:end] = (y_ref[topk_idx] * weights.unsqueeze(-1)).sum(dim=1).cpu()
    return pseudo.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    parser.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    parser.add_argument('--panelA_size', type=int, default=150)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = output_dir(f'measured_knn_oracle_k{args.k}')

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    rep1 = sc.read_h5ad(os.path.join(args.data_root, args.rep1))
    rep2 = sc.read_h5ad(os.path.join(args.data_root, args.rep2))
    rep1.var_names = rep1.var_names.astype(str)
    rep2.var_names = rep2.var_names.astype(str)

    genes = rep1.var_names.values
    np.random.seed(args.seed)
    np.random.shuffle(genes)
    panelA_genes = genes[:args.panelA_size].tolist()
    panelB_genes = genes[args.panelA_size:].tolist()

    def get(adata, glist):
        X = adata[:, glist].X
        return X.toarray() if scipy.sparse.issparse(X) else np.array(X)

    Y_A1 = get(rep1, panelA_genes)
    Y_B1 = get(rep1, panelB_genes)
    Y_A2 = get(rep2, panelA_genes)
    Y_B2 = get(rep2, panelB_genes)

    pseudo_B1 = batched_cosine_knn_pseudo(Y_A1, Y_A2, Y_B2, k=args.k, device=device)
    pseudo_A2 = batched_cosine_knn_pseudo(Y_B2, pseudo_B1, Y_A1, k=args.k, device=device)

    graph1 = se.pp.Build_graph(rep1.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')
    graph2 = se.pp.Build_graph(rep2.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')

    def evaluate(gt, pred, label, graph):
        pcc, pcc_reduce = se.utils.Compute_metrics(gt, pred, metric='pcc')
        ssim, ssim_reduce = se.utils.Compute_metrics(gt, pred, metric='ssim', graph=graph)
        cmd, cmd_reduce = se.utils.Compute_metrics(gt, pred, metric='cmd')
        print(f'[{label}] gene-level PCC: {pcc_reduce:.6f}, SSIM: {ssim_reduce:.6f}, CMD: {cmd_reduce:.6f}')
        return {'pcc': float(pcc_reduce), 'ssim': float(ssim_reduce), 'cmd': float(cmd_reduce)}

    m1 = evaluate(Y_B1, pseudo_B1, f'Slice1 PanelB prediction (measured_knn k={args.k})', graph1)
    m2 = evaluate(Y_A2, pseudo_A2, f'Slice2 PanelA prediction (measured_knn k={args.k})', graph2)

    pred_B1_df = pd.DataFrame(pseudo_B1, index=rep1.obs_names, columns=panelB_genes)
    pred_A2_df = pd.DataFrame(pseudo_A2, index=rep2.obs_names, columns=panelA_genes)
    pred_B1_df.to_csv(os.path.join(args.out_dir, f'pred_panelB1_measured_knn_k{args.k}.csv'))
    pred_A2_df.to_csv(os.path.join(args.out_dir, f'pred_panelA2_measured_knn_k{args.k}.csv'))

    summary = {
        'model': f'measured_knn_k{args.k}',
        'slice1_pcc': m1['pcc'], 'slice1_ssim': m1['ssim'], 'slice1_cmd': m1['cmd'],
        'slice2_pcc': m2['pcc'], 'slice2_ssim': m2['ssim'], 'slice2_cmd': m2['cmd'],
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, f'metrics_measured_knn_k{args.k}.csv'), index=False)
    print(f'\nOutputs saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
