#!/usr/bin/env python3
"""
Ablation: compare forward vs reverse two-step MNN pseudo-label construction.

Forward (official):
  Step1: pseudo_YB1 = MNN(HE1, HE2, YB2)          # H&E bridge
  Step2: pseudo_YA2 = MNN(YB2, pseudo_YB1, YA1)   # B-panel bridge

Reverse (proposed):
  Step1: pseudo_YA2 = MNN(HE2, HE1, YA1)          # H&E bridge
  Step2: pseudo_YB1 = MNN(YA1, pseudo_YA2, YB2)   # A-panel bridge

This script trains separate MLPs on both pseudo-label sets and evaluates them
on the held-out panels to see which bridge direction is better for each slice.
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
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--mnn_k', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--panel_csv', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=output_dir('mnn_pseudo_reverse_ablation'))
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
        sim = query[start:end] @ ref.T
        indices[start:end] = torch.topk(sim, k, dim=1).indices.cpu()
    return indices.numpy()


def build_mnn_pseudo(query, ref, y_ref, k=5, mnn_k=20, batch_size=4096, device='cuda'):
    fwd = batched_topk_indices(query, ref, k=mnn_k, batch_size=batch_size, device=device)
    rev = batched_topk_indices(ref, query, k=mnn_k, batch_size=batch_size, device=device)
    rev_sets = [set(rev[j].tolist()) for j in range(ref.shape[0])]
    selected = []
    for i in range(query.shape[0]):
        mnn = [j for j in fwd[i].tolist() if i in rev_sets[j]]
        if len(mnn) == 0:
            mnn = fwd[i, :k].tolist()
        selected.append(mnn)
    y_ref = np.asarray(y_ref, dtype=np.float32)
    return np.stack([y_ref[idx].mean(axis=0) for idx in selected]).astype(np.float32)


def evaluate(gt, pred, graph):
    pcc, pcc_reduce = se.utils.Compute_metrics(gt, pred, metric='pcc')
    ssim, ssim_reduce = se.utils.Compute_metrics(gt, pred, metric='ssim', graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt, pred, metric='cmd')
    return float(pcc_reduce), float(ssim_reduce), float(cmd_reduce)


def train_panel_mlp(X, Y_pseudo, in_dim, out_dim, hidden_dim, epochs, lr, device):
    model = PanelMLP(in_dim, hidden_dim, out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    X_t = torch.tensor(zscore(X), dtype=torch.float32, device=device)
    Y_t = torch.tensor(zscore(Y_pseudo), dtype=torch.float32, device=device)
    for _ in tqdm(range(epochs), desc='train panel MLP', leave=False):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_t), Y_t)
        loss.backward()
        opt.step()
    return model


def predict(model, x, y_mean, y_std, device):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(zscore(x), dtype=torch.float32, device=device)
        pred = model(x_t).cpu().numpy()
    return pred * y_std + y_mean


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rep1 = load_slice(args.rep1, args.data_root)
    rep2 = load_slice(args.rep2, args.data_root)

    if args.panel_csv and os.path.exists(args.panel_csv):
        panel_df = pd.read_csv(args.panel_csv)
        panelA = panel_df[panel_df['panel'] == 'panelA']['gene'].astype(str).tolist()
        panelB = panel_df[panel_df['panel'] == 'panelB']['gene'].astype(str).tolist()
    else:
        genes = rep1.var_names.values
        np.random.seed(args.seed)
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

    # Forward pseudo-labels (official)
    print('\n=== Forward pseudo-labels ===')
    pseudo_B1_fwd = build_mnn_pseudo(HE1, HE2, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)
    pseudo_A2_fwd = build_mnn_pseudo(Y_B2, pseudo_B1_fwd, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)

    # Reverse pseudo-labels (proposed)
    print('\n=== Reverse pseudo-labels ===')
    pseudo_A2_rev = build_mnn_pseudo(HE2, HE1, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)
    pseudo_B1_rev = build_mnn_pseudo(Y_A1, pseudo_A2_rev, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)

    # Direct pseudo-label ceilings (before MLP)
    fwd_ceil1 = evaluate(Y_B1, pseudo_B1_fwd, graph1)
    fwd_ceil2 = evaluate(Y_A2, pseudo_A2_fwd, graph2)
    rev_ceil1 = evaluate(Y_B1, pseudo_B1_rev, graph1)
    rev_ceil2 = evaluate(Y_A2, pseudo_A2_rev, graph2)

    # Train MLPs for Slice 1 (A -> B)
    print('\n=== Training Slice1 MLP on forward pseudo-YB1 ===')
    model_B1_fwd = train_panel_mlp(
        Y_A1, pseudo_B1_fwd, Y_A1.shape[1], Y_B2.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    print('\n=== Training Slice1 MLP on reverse pseudo-YB1 ===')
    model_B1_rev = train_panel_mlp(
        Y_A1, pseudo_B1_rev, Y_A1.shape[1], Y_B2.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    # Train MLPs for Slice 2 (B -> A)
    print('\n=== Training Slice2 MLP on forward pseudo-YA2 ===')
    model_A2_fwd = train_panel_mlp(
        Y_B2, pseudo_A2_fwd, Y_B2.shape[1], Y_A1.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    print('\n=== Training Slice2 MLP on reverse pseudo-YA2 ===')
    model_A2_rev = train_panel_mlp(
        Y_B2, pseudo_A2_rev, Y_B2.shape[1], Y_A1.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    # Inference
    yB_mean2, yB_std2 = Y_B2.mean(axis=0), Y_B2.std(axis=0) + 1e-8
    yA_mean1, yA_std1 = Y_A1.mean(axis=0), Y_A1.std(axis=0) + 1e-8

    pred_B1_fwd = predict(model_B1_fwd, Y_A1, yB_mean2, yB_std2, device)
    pred_B1_rev = predict(model_B1_rev, Y_A1, yB_mean2, yB_std2, device)
    pred_A2_fwd = predict(model_A2_fwd, Y_B2, yA_mean1, yA_std1, device)
    pred_A2_rev = predict(model_A2_rev, Y_B2, yA_mean1, yA_std1, device)

    fwd_learn1 = evaluate(Y_B1, pred_B1_fwd, graph1)
    rev_learn1 = evaluate(Y_B1, pred_B1_rev, graph1)
    fwd_learn2 = evaluate(Y_A2, pred_A2_fwd, graph2)
    rev_learn2 = evaluate(Y_A2, pred_A2_rev, graph2)

    rows = [
        {'direction': 'forward', 'what': 'pseudo ceiling slice1',
         'slice1_pcc': fwd_ceil1[0], 'slice1_ssim': fwd_ceil1[1], 'slice1_cmd': fwd_ceil1[2]},
        {'direction': 'forward', 'what': 'learned MLP slice1',
         'slice1_pcc': fwd_learn1[0], 'slice1_ssim': fwd_learn1[1], 'slice1_cmd': fwd_learn1[2]},
        {'direction': 'reverse', 'what': 'pseudo ceiling slice1',
         'slice1_pcc': rev_ceil1[0], 'slice1_ssim': rev_ceil1[1], 'slice1_cmd': rev_ceil1[2]},
        {'direction': 'reverse', 'what': 'learned MLP slice1',
         'slice1_pcc': rev_learn1[0], 'slice1_ssim': rev_learn1[1], 'slice1_cmd': rev_learn1[2]},
        {'direction': 'forward', 'what': 'pseudo ceiling slice2',
         'slice2_pcc': fwd_ceil2[0], 'slice2_ssim': fwd_ceil2[1], 'slice2_cmd': fwd_ceil2[2]},
        {'direction': 'forward', 'what': 'learned MLP slice2',
         'slice2_pcc': fwd_learn2[0], 'slice2_ssim': fwd_learn2[1], 'slice2_cmd': fwd_learn2[2]},
        {'direction': 'reverse', 'what': 'pseudo ceiling slice2',
         'slice2_pcc': rev_ceil2[0], 'slice2_ssim': rev_ceil2[1], 'slice2_cmd': rev_ceil2[2]},
        {'direction': 'reverse', 'what': 'learned MLP slice2',
         'slice2_pcc': rev_learn2[0], 'slice2_ssim': rev_learn2[1], 'slice2_cmd': rev_learn2[2]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'reverse_ablation_metrics.csv'), index=False)

    print('\n=== Results ===')
    print(df.to_string(index=False))

    # Save predictions
    pd.DataFrame(pred_B1_fwd, index=rep1.obs_names, columns=panelB).to_csv(
        os.path.join(args.out_dir, 'pred_panelB1_forward.csv'))
    pd.DataFrame(pred_B1_rev, index=rep1.obs_names, columns=panelB).to_csv(
        os.path.join(args.out_dir, 'pred_panelB1_reverse.csv'))
    pd.DataFrame(pred_A2_fwd, index=rep2.obs_names, columns=panelA).to_csv(
        os.path.join(args.out_dir, 'pred_panelA2_forward.csv'))
    pd.DataFrame(pred_A2_rev, index=rep2.obs_names, columns=panelA).to_csv(
        os.path.join(args.out_dir, 'pred_panelA2_reverse.csv'))
    print(f'\nOutputs saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
