#!/usr/bin/env python3
"""
Fig. 3 panel diagonal integration with MNN-filtered measured-panel pseudo-labels.

Compares:
  1. Raw cosine kNN pseudo-labels (baseline).
  2. Mutual-nearest-neighbor (MNN) filtered pseudo-labels.

For each, reports:
  - pseudo-label ceiling (direct evaluation vs held-out truth)
  - learned panel-to-panel MLP performance (trained on pseudo-labels)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc
import torch
import torch.nn as nn
from tqdm import tqdm

import SpatialEx as se


class PanelMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    parser.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    parser.add_argument('--panelA_size', type=int, default=150)
    parser.add_argument('--k', type=int, default=5, help='Final number of neighbors for pseudo-label averaging')
    parser.add_argument('--mnn_k', type=int, default=20, help='Neighbors considered for MNN filtering')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_mnn_pseudo'))
    return parser.parse_args()


def load_slice(path, data_root):
    adata = sc.read_h5ad(os.path.join(data_root, path))
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata


def get_X(adata, genes):
    X = adata[:, genes].X
    return X.toarray() if scipy.sparse.issparse(X) else np.array(X)


def zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def batched_topk_indices(query, ref, k=20, batch_size=4096, device='cuda'):
    """Return top-k ref indices for each query vector."""
    query = torch.as_tensor(query, dtype=torch.float32, device=device)
    ref = torch.as_tensor(ref, dtype=torch.float32, device=device)
    query = query / (query.norm(dim=1, keepdim=True) + 1e-8)
    ref = ref / (ref.norm(dim=1, keepdim=True) + 1e-8)
    k = min(k, ref.shape[0])
    n_query = query.shape[0]
    indices = torch.zeros(n_query, k, dtype=torch.long, device='cpu')
    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        sim = query[start:end] @ ref.T
        indices[start:end] = torch.topk(sim, k, dim=1).indices.cpu()
    return indices.numpy()


def build_mnn_pseudo(query, ref, y_ref, k=5, mnn_k=20, batch_size=4096, device='cuda'):
    """Build pseudo-labels with MNN filtering; fall back to raw kNN when no MNN."""
    print('Building forward kNN...')
    fwd = batched_topk_indices(query, ref, k=mnn_k, batch_size=batch_size, device=device)
    print('Building reverse kNN...')
    rev = batched_topk_indices(ref, query, k=mnn_k, batch_size=batch_size, device=device)

    n_query = query.shape[0]
    n_ref = ref.shape[0]
    # Build reverse membership sets for fast lookup
    rev_sets = [set(rev[j].tolist()) for j in range(n_ref)]

    selected = []
    for i in range(n_query):
        mnn = [j for j in fwd[i].tolist() if i in rev_sets[j]]
        if len(mnn) == 0:
            mnn = fwd[i, :k].tolist()
        selected.append(mnn)
    pseudo = np.stack([y_ref[idx].mean(axis=0) for idx in selected]).astype(np.float32)
    return pseudo


def build_raw_pseudo(query, ref, y_ref, k=5, batch_size=4096, device='cuda'):
    query = torch.as_tensor(query, dtype=torch.float32, device=device)
    ref = torch.as_tensor(ref, dtype=torch.float32, device=device)
    y_ref = torch.as_tensor(y_ref, dtype=torch.float32, device=device)
    query = query / (query.norm(dim=1, keepdim=True) + 1e-8)
    ref = ref / (ref.norm(dim=1, keepdim=True) + 1e-8)
    k = min(k, ref.shape[0])
    n_query = query.shape[0]
    pseudo = torch.zeros(n_query, y_ref.shape[1], dtype=torch.float32, device='cpu')
    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        sim = query[start:end] @ ref.T
        topk_sim, topk_idx = torch.topk(sim, k, dim=1)
        weights = topk_sim / (topk_sim.sum(dim=1, keepdim=True) + 1e-8)
        pseudo[start:end] = (y_ref[topk_idx] * weights.unsqueeze(-1)).sum(dim=1).cpu()
    return pseudo.numpy()


def evaluate(gt, pred, graph):
    pcc, pcc_reduce = se.utils.Compute_metrics(gt, pred, metric='pcc')
    ssim, ssim_reduce = se.utils.Compute_metrics(gt, pred, metric='ssim', graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt, pred, metric='cmd')
    return float(pcc_reduce), float(ssim_reduce), float(cmd_reduce)


def train_panel_mlp(X1, Y1_pseudo, X2, Y2_pseudo, in_dim, out_dim1, out_dim2,
                    hidden_dim, epochs, lr, device):
    model1 = PanelMLP(in_dim, hidden_dim, out_dim1).to(device)
    model2 = PanelMLP(out_dim1, hidden_dim, out_dim2).to(device)
    opt = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=lr)
    crit = nn.MSELoss()
    X1_t = torch.tensor(zscore(X1), dtype=torch.float32, device=device)
    X2_t = torch.tensor(zscore(X2), dtype=torch.float32, device=device)
    Y1_t = torch.tensor(zscore(Y1_pseudo), dtype=torch.float32, device=device)
    Y2_t = torch.tensor(zscore(Y2_pseudo), dtype=torch.float32, device=device)
    for epoch in tqdm(range(epochs), desc='train panel MLP'):
        model1.train(); model2.train()
        opt.zero_grad()
        loss = crit(model1(X1_t), Y1_t) + crit(model2(X2_t), Y2_t)
        loss.backward()
        opt.step()
    return model1, model2


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    rep1 = load_slice(args.rep1, args.data_root)
    rep2 = load_slice(args.rep2, args.data_root)

    genes = rep1.var_names.values
    np.random.shuffle(genes)
    panelA = genes[:args.panelA_size].tolist()
    panelB = genes[args.panelA_size:].tolist()

    Y_A1 = get_X(rep1, panelA).astype(np.float32)
    Y_B1 = get_X(rep1, panelB).astype(np.float32)
    Y_A2 = get_X(rep2, panelA).astype(np.float32)
    Y_B2 = get_X(rep2, panelB).astype(np.float32)

    graph1 = se.pp.Build_graph(rep1.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')
    graph2 = se.pp.Build_graph(rep2.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')

    # Build pseudo labels
    print('=== Raw cosine kNN pseudo labels ===')
    pseudo_B1_raw = build_raw_pseudo(Y_A1, Y_A2, Y_B2, k=args.k, device=device)
    pseudo_A2_raw = build_raw_pseudo(Y_B2, pseudo_B1_raw, Y_A1, k=args.k, device=device)

    print('\n=== MNN-filtered pseudo labels ===')
    pseudo_B1_mnn = build_mnn_pseudo(Y_A1, Y_A2, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)
    pseudo_A2_mnn = build_mnn_pseudo(Y_B2, pseudo_B1_mnn, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)

    # Pseudo-label ceilings
    raw_ceil1 = evaluate(Y_B1, pseudo_B1_raw, graph1)
    raw_ceil2 = evaluate(Y_A2, pseudo_A2_raw, graph2)
    mnn_ceil1 = evaluate(Y_B1, pseudo_B1_mnn, graph1)
    mnn_ceil2 = evaluate(Y_A2, pseudo_A2_mnn, graph2)

    print('\n=== Training panel MLPs on raw pseudo labels ===')
    model_B1_raw, model_A2_raw = train_panel_mlp(
        Y_A1, pseudo_B1_raw, Y_B2, pseudo_A2_raw,
        Y_A1.shape[1], Y_B2.shape[1], Y_A1.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    print('\n=== Training panel MLPs on MNN pseudo labels ===')
    model_B1_mnn, model_A2_mnn = train_panel_mlp(
        Y_A1, pseudo_B1_mnn, Y_B2, pseudo_A2_mnn,
        Y_A1.shape[1], Y_B2.shape[1], Y_A1.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    # Inference
    def predict(model, x, y_mean, y_std):
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(zscore(x), dtype=torch.float32, device=device)
            pred = model(x_t).cpu().numpy()
        pred = pred * y_std + y_mean
        return pred

    yB_mean2, yB_std2 = Y_B2.mean(axis=0), Y_B2.std(axis=0) + 1e-8
    yA_mean1, yA_std1 = Y_A1.mean(axis=0), Y_A1.std(axis=0) + 1e-8

    pred_B1_raw = predict(model_B1_raw, Y_A1, yB_mean2, yB_std2)
    pred_A2_raw = predict(model_A2_raw, Y_B2, yA_mean1, yA_std1)
    pred_B1_mnn = predict(model_B1_mnn, Y_A1, yB_mean2, yB_std2)
    pred_A2_mnn = predict(model_A2_mnn, Y_B2, yA_mean1, yA_std1)

    raw_learn1 = evaluate(Y_B1, pred_B1_raw, graph1)
    raw_learn2 = evaluate(Y_A2, pred_A2_raw, graph2)
    mnn_learn1 = evaluate(Y_B1, pred_B1_mnn, graph1)
    mnn_learn2 = evaluate(Y_A2, pred_A2_mnn, graph2)

    rows = [
        {'method': 'raw kNN', 'what': 'pseudo ceiling',
         'slice1_pcc': raw_ceil1[0], 'slice1_ssim': raw_ceil1[1], 'slice1_cmd': raw_ceil1[2],
         'slice2_pcc': raw_ceil2[0], 'slice2_ssim': raw_ceil2[1], 'slice2_cmd': raw_ceil2[2]},
        {'method': 'raw kNN', 'what': 'learned MLP',
         'slice1_pcc': raw_learn1[0], 'slice1_ssim': raw_learn1[1], 'slice1_cmd': raw_learn1[2],
         'slice2_pcc': raw_learn2[0], 'slice2_ssim': raw_learn2[1], 'slice2_cmd': raw_learn2[2]},
        {'method': 'MNN kNN', 'what': 'pseudo ceiling',
         'slice1_pcc': mnn_ceil1[0], 'slice1_ssim': mnn_ceil1[1], 'slice1_cmd': mnn_ceil1[2],
         'slice2_pcc': mnn_ceil2[0], 'slice2_ssim': mnn_ceil2[1], 'slice2_cmd': mnn_ceil2[2]},
        {'method': 'MNN kNN', 'what': 'learned MLP',
         'slice1_pcc': mnn_learn1[0], 'slice1_ssim': mnn_learn1[1], 'slice1_cmd': mnn_learn1[2],
         'slice2_pcc': mnn_learn2[0], 'slice2_ssim': mnn_learn2[1], 'slice2_cmd': mnn_learn2[2]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'mnn_metrics.csv'), index=False)
    print('\n=== Results ===')
    print(df.to_string(index=False))

    # Save predictions
    pd.DataFrame(pred_B1_mnn, index=rep1.obs_names, columns=panelB).to_csv(
        os.path.join(args.out_dir, 'pred_panelB1_mnn.csv'))
    pd.DataFrame(pred_A2_mnn, index=rep2.obs_names, columns=panelA).to_csv(
        os.path.join(args.out_dir, 'pred_panelA2_mnn.csv'))
    print(f'\nOutputs saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
