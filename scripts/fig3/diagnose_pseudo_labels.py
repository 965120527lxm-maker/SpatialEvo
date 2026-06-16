#!/usr/bin/env python3
"""Quick diagnostic of cross-slice pseudo-label quality under different matchers."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import scipy.sparse
import scanpy as sc
import torch


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


def pcc_mean(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    denom = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum(axis=0)) + 1e-8
    return ((xc * yc).sum(axis=0) / denom).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    parser.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    parser.add_argument('--panelA_size', type=int, default=150)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

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

    HE1 = np.asarray(rep1.obsm['he'], dtype=np.float32)
    HE2 = np.asarray(rep2.obsm['he'], dtype=np.float32)
    Y_A1 = get(rep1, panelA_genes)
    Y_B1 = get(rep1, panelB_genes)
    Y_A2 = get(rep2, panelA_genes)
    Y_B2 = get(rep2, panelB_genes)

    def zscore(x):
        return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

    for k in [5, 20, 50, 100]:
        print(f'\n=== k={k} ===')
        # H&E only pseudo labels
        pseudo_B1_he = batched_cosine_knn_pseudo(zscore(HE1), zscore(HE2), Y_B2, k=k, device=device)
        pseudo_A2_he = batched_cosine_knn_pseudo(zscore(HE2), zscore(HE1), Y_A1, k=k, device=device)
        print(f'Slice1 H&E-only pseudo B PCC: {pcc_mean(pseudo_B1_he, Y_B1):.4f}')
        print(f'Slice2 H&E-only pseudo A PCC: {pcc_mean(pseudo_A2_he, Y_A2):.4f}')

        # Direct measured A -> B for slice1
        pseudo_B1 = batched_cosine_knn_pseudo(Y_A1, Y_A2, Y_B2, k=k, device=device)
        print(f'Slice1 A->A->B PCC: {pcc_mean(pseudo_B1, Y_B1):.4f}')

        # Hybrid HE+A -> B for slice1
        hya1 = np.concatenate([zscore(HE1), zscore(Y_A1)], axis=1)
        hya2 = np.concatenate([zscore(HE2), zscore(Y_A2)], axis=1)
        pseudo_B1_hyb = batched_cosine_knn_pseudo(hya1, hya2, Y_B2, k=k, device=device)
        print(f'Slice1 HE+A hybrid PCC: {pcc_mean(pseudo_B1_hyb, Y_B1):.4f}')

        # Iterative measured B -> pseudoB -> A for slice2
        pseudo_A2 = batched_cosine_knn_pseudo(Y_B2, pseudo_B1, Y_A1, k=k, device=device)
        print(f'Slice2 B->pseudoB->A PCC: {pcc_mean(pseudo_A2, Y_A2):.4f}')

        # Direct measured B -> A for slice2 if we had B1 (cheat for upper bound)
        pseudo_A2_direct = batched_cosine_knn_pseudo(Y_B2, Y_B1, Y_A1, k=k, device=device)
        print(f'Slice2 B->B->A (uses held-out B1) PCC: {pcc_mean(pseudo_A2_direct, Y_A2):.4f}')

        # Direct measured A -> B for slice2 if we had A1/A2? Not applicable


if __name__ == '__main__':
    main()
