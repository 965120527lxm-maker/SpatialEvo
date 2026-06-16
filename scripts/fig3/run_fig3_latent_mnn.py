#!/usr/bin/env python3
"""
Shared latent alignment + MNN for measured-panel pseudo-labels.

The latent space is used only for cross-slice matching; the MLP still takes
raw measured panel as input and predicts the missing panel.

We compare:
  1. Raw measured panel MNN (no alignment)
  2. PCA latent + MNN
  3. CORAL domain alignment + MNN

No ground-truth leakage: Slice2 -> Slice1 matching only uses pseudo panel B on Slice1,
never the true Y_B1.
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
    parser.add_argument('--pca_dim', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_latent_mnn'))
    return parser.parse_args()


def zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def fit_pca(X1, X2, n_components=50):
    from sklearn.decomposition import PCA
    X = np.concatenate([zscore(X1), zscore(X2)], axis=0)
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X)
    Z1 = pca.transform(zscore(X1))
    Z2 = pca.transform(zscore(X2))
    return Z1.astype(np.float32), Z2.astype(np.float32)


def coral_alignment(X_src, X_tgt):
    """Linear CORAL: align source to target covariance."""
    xs = zscore(X_src)
    xt = zscore(X_tgt)
    Cs = np.cov(xs, rowvar=False) + np.eye(xs.shape[1]) * 1e-5
    Ct = np.cov(xt, rowvar=False) + np.eye(xt.shape[1]) * 1e-5

    def mat_sqrt(M):
        u, s, vt = np.linalg.svd(M)
        return u @ np.diag(np.sqrt(s)) @ vt

    W = mat_sqrt(Ct) @ np.linalg.inv(mat_sqrt(Cs))
    X_src_aligned = xs @ W.T
    return X_src_aligned.astype(np.float32)


def train_panel_mlps(Y_A1, pseudo_B1, Y_B2, pseudo_A2, hidden_dim, epochs, lr, device):
    model1 = mnn.PanelMLP(Y_A1.shape[1], hidden_dim, pseudo_B1.shape[1]).to(device)
    model2 = mnn.PanelMLP(Y_B2.shape[1], hidden_dim, pseudo_A2.shape[1]).to(device)
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


def evaluate(gt, pred, graph):
    import SpatialEx as se
    pcc, pcc_reduce = se.utils.Compute_metrics(gt, pred, metric='pcc')
    ssim, ssim_reduce = se.utils.Compute_metrics(gt, pred, metric='ssim', graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt, pred, metric='cmd')
    return float(pcc_reduce), float(ssim_reduce), float(cmd_reduce)


def run_pipeline(name, a1, a2, b2, pseudoB1_aligned_to_b2,
                 Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device):
    """
    a1/a2: A-space embeddings for Slice1/Slice2 -> build pseudoB1.
    b2: B-space embedding for Slice2 query -> build pseudoA2.
    pseudoB1_aligned_to_b2: pseudoB1 aligned into b2 space, used as ref for pseudoA2 matching.
    """
    print(f'\n=== {name} ===')

    pseudo_B1 = mnn.build_mnn_pseudo(a1, a2, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)
    pseudo_A2 = mnn.build_mnn_pseudo(b2, pseudoB1_aligned_to_b2, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)

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

    return {
        'method': name,
        'slice1_direct_pcc': direct1[0], 'slice1_learned_pcc': learned1[0],
        'slice2_direct_pcc': direct2[0], 'slice2_learned_pcc': learned2[0],
    }


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

    import SpatialEx as se
    graph1 = se.pp.Build_graph(rep1.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')
    graph2 = se.pp.Build_graph(rep2.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')

    rows = []

    # 1) Raw measured panel
    pseudo_B1_raw = mnn.build_mnn_pseudo(Y_A1, Y_A2, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)
    rows.append(run_pipeline('raw measured panel',
                             Y_A1, Y_A2, Y_B2, pseudo_B1_raw,
                             Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device))

    # 2) PCA latent
    Z_A1, Z_A2 = fit_pca(Y_A1, Y_A2, n_components=args.pca_dim)
    pseudo_B1_pca = mnn.build_mnn_pseudo(Z_A1, Z_A2, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)
    Z_B2, Z_pseudoB1 = fit_pca(Y_B2, pseudo_B1_pca, n_components=args.pca_dim)
    rows.append(run_pipeline('PCA latent',
                             Z_A1, Z_A2, Z_B2, Z_pseudoB1,
                             Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device))

    # 3) CORAL aligned
    Y_A2_coral_to_A1 = coral_alignment(Y_A2, Y_A1)
    pseudo_B1_coral = mnn.build_mnn_pseudo(Y_A1, Y_A2_coral_to_A1, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)
    pseudo_B1_coral_to_B2 = coral_alignment(pseudo_B1_coral, Y_B2)
    rows.append(run_pipeline('CORAL aligned',
                             Y_A1, Y_A2_coral_to_A1, Y_B2, pseudo_B1_coral_to_B2,
                             Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'latent_mnn_results.csv'), index=False)
    print('\n=== Latent + MNN results ===')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
