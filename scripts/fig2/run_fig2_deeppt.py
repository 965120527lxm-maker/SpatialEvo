#!/usr/bin/env python3
"""
Fig.2 DeepPT baseline: per-cell H&E -> full transcriptome.

  - Train on Rep1 (HE1 -> RNA1), predict Rep2
  - Train on Rep2 (HE2 -> RNA2), predict Rep1
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'baselines'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'fig2'))
from exp_paths import output_dir

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch
from torch.utils.data import DataLoader, TensorDataset

import SpatialEx as se
from run_deeppt_fig3 import train_deeppt, split_train_valid, predict, init_seed


def parse_args():
    p = argparse.ArgumentParser(description='Fig.2 DeepPT H&E-to-omics')
    p.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    p.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    p.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--valid_frac', type=float, default=0.1)
    p.add_argument('--num_neighbors', type=int, default=7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--out_dir', type=str, default=output_dir('deeppt_breast_cancer'))
    return p.parse_args()


def load_slice(path, data_root):
    adata = sc.read_h5ad(os.path.join(data_root, path))
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata


def get_matrix(adata):
    X = adata.X
    return X.toarray() if scipy.sparse.issparse(X) else np.asarray(X, dtype=np.float32)


def eval_pred(gt, pred, graph, slice_name):
    pcc, pcc_reduce = se.utils.Compute_metrics(gt, pred, metric='pcc')
    ssim, ssim_reduce = se.utils.Compute_metrics(gt, pred, metric='ssim', graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt, pred, metric='cmd')
    print(f'[{slice_name}] PCC={pcc_reduce:.4f} SSIM={ssim_reduce:.4f} CMD={cmd_reduce:.4f}')
    return {
        'slice': slice_name,
        'pcc_mean': float(pcc_reduce),
        'ssim_mean': float(ssim_reduce),
        'cmd_mean': float(cmd_reduce),
        'pcc_per_gene': pcc.tolist() if hasattr(pcc, 'tolist') else list(pcc),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    init_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    adata1 = load_slice(args.rep1, args.data_root)
    adata2 = load_slice(args.rep2, args.data_root)
    genes = adata1.var_names.astype(str).tolist()

    he1 = np.asarray(adata1.obsm['he'], dtype=np.float32)
    he2 = np.asarray(adata2.obsm['he'], dtype=np.float32)
    rna1 = get_matrix(adata1)
    rna2 = get_matrix(adata2)

    graph1 = se.pp.Build_graph(
        adata1.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo', num_neighbors=args.num_neighbors)
    graph2 = se.pp.Build_graph(
        adata2.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo', num_neighbors=args.num_neighbors)

    print('=== DeepPT: train Rep1, predict Rep2 ===')
    tr_idx, va_idx = split_train_valid(len(he1), args.valid_frac, args.seed)
    model_1to2 = train_deeppt(he1[tr_idx], rna1[tr_idx], he1[va_idx], rna1[va_idx], args, device)
    pred_loader = DataLoader(
        TensorDataset(torch.tensor(he2, dtype=torch.float32), torch.zeros(len(he2), 1)),
        batch_size=args.batch_size, shuffle=False)
    pred2 = predict(model_1to2, pred_loader, device)
    m2 = eval_pred(rna2, pred2, graph2, 'Slice2 (train Rep1)')

    print('=== DeepPT: train Rep2, predict Rep1 ===')
    tr_idx, va_idx = split_train_valid(len(he2), args.valid_frac, args.seed + 1)
    model_2to1 = train_deeppt(he2[tr_idx], rna2[tr_idx], he2[va_idx], rna2[va_idx], args, device)
    pred_loader = DataLoader(
        TensorDataset(torch.tensor(he1, dtype=torch.float32), torch.zeros(len(he1), 1)),
        batch_size=args.batch_size, shuffle=False)
    pred1 = predict(model_2to1, pred_loader, device)
    m1 = eval_pred(rna1, pred1, graph1, 'Slice1 (train Rep2)')

    summary = pd.DataFrame([
        {k: v for k, v in m1.items() if k != 'pcc_per_gene'},
        {k: v for k, v in m2.items() if k != 'pcc_per_gene'},
    ])
    summary.to_csv(os.path.join(args.out_dir, 'metrics_deeppt.csv'), index=False)

    per_gene_rows = []
    for m in (m1, m2):
        for g, p in zip(genes, m['pcc_per_gene']):
            per_gene_rows.append({'slice': m['slice'], 'gene': g, 'pcc': p})
    pd.DataFrame(per_gene_rows).to_csv(os.path.join(args.out_dir, 'per_gene_pcc.csv'), index=False)

    pd.DataFrame(pred1, index=adata1.obs_names, columns=genes).to_csv(
        os.path.join(args.out_dir, 'pred_slice1.csv'))
    pd.DataFrame(pred2, index=adata2.obs_names, columns=genes).to_csv(
        os.path.join(args.out_dir, 'pred_slice2.csv'))

    print(f'\nSaved to {args.out_dir}')


if __name__ == '__main__':
    main()
