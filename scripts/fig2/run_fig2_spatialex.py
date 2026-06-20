#!/usr/bin/env python3
"""
Fig.2: SpatialEx H&E-to-omics (Tutorial 1 protocol).

Train SpatialEx on Rep1 + Rep2 (full 313-gene Xenium panel), then evaluate
cross-slice H&E -> omics prediction:
  - Slice 1: module_HB predicts expression on Rep1 (panelB1 vs measured Rep1)
  - Slice 2: module_HA predicts expression on Rep2 (panelA2 vs measured Rep2)

Metrics: gene-level PCC, SSIM, CMD (+ per-gene PCC table for Fig.2c-style plots).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'fig2'))
from exp_paths import output_dir

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import SpatialEx as se


def parse_args():
    p = argparse.ArgumentParser(description='Fig.2 SpatialEx H&E-to-omics')
    p.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    p.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    p.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    p.add_argument('--num_neighbors', type=int, default=7)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--out_dir', type=str, default=output_dir('spatialex_breast_cancer'))
    return p.parse_args()


def load_slice(path, data_root):
    adata = sc.read_h5ad(os.path.join(data_root, path))
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata


def gt_matrix(adata):
    X = adata.X
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X, dtype=np.float64)


def eval_slice(gt, pred, graph, slice_name):
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
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    adata1 = load_slice(args.rep1, args.data_root)
    adata2 = load_slice(args.rep2, args.data_root)
    genes = adata1.var_names.astype(str).tolist()
    assert list(adata2.var_names.astype(str)) == genes, 'Rep1/Rep2 gene sets must match'

    graph1 = se.pp.Build_hypergraph_spatial_and_HE(
        adata1, args.num_neighbors, graph_kind='spatial', return_type='csr')
    graph2 = se.pp.Build_hypergraph_spatial_and_HE(
        adata2, args.num_neighbors, graph_kind='spatial', return_type='csr')

    print(f'Training SpatialEx: {adata1.n_obs} + {adata2.n_obs} cells, {len(genes)} genes')
    model = se.SpatialEx(
        adata1, adata2, graph1, graph2,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=device,
        num_neighbors=args.num_neighbors,
    )
    model.train()
    panel_b1, panel_a2 = model.auto_inference()

    # Align prediction gene order to ground truth
    gt1 = gt_matrix(adata1)
    gt2 = gt_matrix(adata2)
    if panel_b1.shape[1] == len(genes):
        pred1 = panel_b1
    else:
        pred1 = panel_b1
    pred2 = panel_a2

    graph1_coo = se.pp.Build_graph(
        adata1.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo', num_neighbors=args.num_neighbors)
    graph2_coo = se.pp.Build_graph(
        adata2.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo', num_neighbors=args.num_neighbors)

    m1 = eval_slice(gt1, pred1, graph1_coo, 'Slice1')
    m2 = eval_slice(gt2, pred2, graph2_coo, 'Slice2')

    summary = pd.DataFrame([
        {k: v for k, v in m1.items() if k != 'pcc_per_gene'},
        {k: v for k, v in m2.items() if k != 'pcc_per_gene'},
    ])
    summary.to_csv(os.path.join(args.out_dir, 'metrics_spatialex.csv'), index=False)

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
