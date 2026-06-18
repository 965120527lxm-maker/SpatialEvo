#!/usr/bin/env python3
"""MNN pseudo-label parameter sensitivity sweep."""

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
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_mnn_sweep'))
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
    query = torch.as_tensor(query, dtype=torch.float32, device=device)
    ref = torch.as_tensor(ref, dtype=torch.float32, device=device)
    query = query / (query.norm(dim=1, keepdim=True) + 1e-8)
    ref = ref / (ref.norm(dim=1, keepdim=True) + 1e-8)
    k = min(k, ref.shape[0])
    n_query = query.shape[0]
    indices = torch.zeros(n_query, k, dtype=torch.long, device='cpu')
    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        indices[start:end] = torch.topk(query[start:end] @ ref.T, k, dim=1).indices.cpu()
    return indices.numpy()


def build_mnn_pseudo(query, ref, y_ref, k=5, mnn_k=20, device='cuda'):
    fwd = batched_topk_indices(query, ref, k=mnn_k, device=device)
    rev = batched_topk_indices(ref, query, k=mnn_k, device=device)
    n_ref = ref.shape[0]
    rev_sets = [set(rev[j].tolist()) for j in range(n_ref)]
    selected = []
    for i in range(query.shape[0]):
        mnn = [j for j in fwd[i].tolist() if i in rev_sets[j]]
        if len(mnn) == 0:
            mnn = fwd[i, :k].tolist()
        selected.append(mnn)
    return np.stack([y_ref[idx].mean(axis=0) for idx in selected]).astype(np.float32)


def evaluate(gt, pred, graph):
    pcc, pcc_reduce = se.utils.Compute_metrics(gt, pred, metric='pcc')
    ssim, ssim_reduce = se.utils.Compute_metrics(gt, pred, metric='ssim', graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt, pred, metric='cmd')
    return float(pcc_reduce), float(ssim_reduce), float(cmd_reduce)


def train_panel_mlps(Y_A1, pseudo_B1, Y_B2, pseudo_A2, hidden_dim, epochs, lr, device):
    model1 = PanelMLP(Y_A1.shape[1], hidden_dim, pseudo_B1.shape[1]).to(device)
    model2 = PanelMLP(Y_B2.shape[1], hidden_dim, pseudo_A2.shape[1]).to(device)
    opt = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=lr)
    crit = nn.MSELoss()
    X1 = torch.tensor(zscore(Y_A1), dtype=torch.float32, device=device)
    X2 = torch.tensor(zscore(Y_B2), dtype=torch.float32, device=device)
    T1 = torch.tensor(zscore(pseudo_B1), dtype=torch.float32, device=device)
    T2 = torch.tensor(zscore(pseudo_A2), dtype=torch.float32, device=device)
    for _ in tqdm(range(epochs), desc='train MLP', leave=False):
        opt.zero_grad()
        loss = crit(model1(X1), T1) + crit(model2(X2), T2)
        loss.backward()
        opt.step()
    return model1, model2


def predict(model, x, y_mean, y_std, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(zscore(x), dtype=torch.float32, device=device)).cpu().numpy()
    return pred * y_std + y_mean


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
    HE1 = np.asarray(rep1.obsm['he'], dtype=np.float32)
    HE2 = np.asarray(rep2.obsm['he'], dtype=np.float32)

    graph1 = se.pp.Build_graph(rep1.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')
    graph2 = se.pp.Build_graph(rep2.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')

    configs = [(5, 10), (5, 20), (10, 20), (10, 50), (20, 50)]
    rows = []

    for k, mnn_k in configs:
        print(f'\n=== Config k={k}, mnn_k={mnn_k} ===')
        pseudo_B1 = build_mnn_pseudo(HE1, HE2, Y_B2, k=k, mnn_k=mnn_k, device=device)
        pseudo_A2 = build_mnn_pseudo(Y_B2, pseudo_B1, Y_A1, k=k, mnn_k=mnn_k, device=device)

        direct1 = evaluate(Y_B1, pseudo_B1, graph1)
        direct2 = evaluate(Y_A2, pseudo_A2, graph2)

        model1, model2 = train_panel_mlps(Y_A1, pseudo_B1, Y_B2, pseudo_A2,
                                          args.hidden_dim, args.epochs, args.lr, device)

        yB_mean, yB_std = Y_B2.mean(axis=0), Y_B2.std(axis=0) + 1e-8
        yA_mean, yA_std = Y_A1.mean(axis=0), Y_A1.std(axis=0) + 1e-8
        pred_B1 = predict(model1, Y_A1, yB_mean, yB_std, device)
        pred_A2 = predict(model2, Y_B2, yA_mean, yA_std, device)
        learned1 = evaluate(Y_B1, pred_B1, graph1)
        learned2 = evaluate(Y_A2, pred_A2, graph2)

        rows.append({
            'k': k, 'mnn_k': mnn_k,
            'slice1_direct_pcc': direct1[0], 'slice1_learned_pcc': learned1[0],
            'slice2_direct_pcc': direct2[0], 'slice2_learned_pcc': learned2[0],
            'slice1_direct_ssim': direct1[1], 'slice1_learned_ssim': learned1[1],
            'slice2_direct_ssim': direct2[1], 'slice2_learned_ssim': learned2[1],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'mnn_sweep.csv'), index=False)
    print('\n=== MNN sweep results ===')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
