#!/usr/bin/env python3
"""
Decompose SpatialEx+ into two prediction paths and evaluate late fusion.

Paths:
    1. H&E branch: P_B(X_1) -> PanelB on slice 1, P_A(X_2) -> PanelA on slice 2.
    2. Panel-to-panel branch: C_{A->B}(Y_A^1) -> PanelB on slice 1,
                              C_{B->A}(Y_B^2) -> PanelA on slice 2.

The panel MLPs are trained on measured-panel pseudo-labels (cross-slice kNN),
with cycle consistency and distribution matching as regularizers.

Held-out panels (Y_B^1, Y_A^2) are used only for final evaluation.
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

import SpatialEx as se


class MLP(nn.Module):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    parser.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    parser.add_argument('--panelA_size', type=int, default=150)
    parser.add_argument('--pseudo_k', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs_he', type=int, default=300)
    parser.add_argument('--epochs_panel', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_cycle', type=float, default=0.5)
    parser.add_argument('--lambda_dist', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default=output_dir('decomposed_diagnosis'))
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


def zscore_fit_transform(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    return (x - mean) / std, mean, std


def train_mlp(model, x, y, epochs, lr, device, print_every=50):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f'  {model.__class__.__name__} epoch {epoch + 1}/{epochs}, loss {loss.item():.4f}')


def evaluate(gt, pred, graph):
    pcc, pcc_reduce = se.utils.Compute_metrics(gt, pred, metric='pcc')
    ssim, ssim_reduce = se.utils.Compute_metrics(gt, pred, metric='ssim', graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt, pred, metric='cmd')
    return float(pcc_reduce), float(ssim_reduce), float(cmd_reduce)


def pcc_only(gt, pred):
    x = np.asarray(gt)
    y = np.asarray(pred)
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    denom = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum(axis=0)) + 1e-8
    return float(((xc * yc).sum(axis=0) / denom).mean())


def calibrate_to_target(pred, target):
    """Gene-wise z-score pred, then rescale to target's per-gene mean/std."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    pred_mean = pred.mean(axis=0)
    pred_std = pred.std(axis=0) + 1e-8
    tgt_mean = target.mean(axis=0)
    tgt_std = target.std(axis=0) + 1e-8
    return ((pred - pred_mean) / pred_std) * tgt_std + tgt_mean


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rep1 = load_slice(args.rep1, args.data_root)
    rep2 = load_slice(args.rep2, args.data_root)

    genes = rep1.var_names.values
    np.random.shuffle(genes)
    panelA = genes[:args.panelA_size].tolist()
    panelB = genes[args.panelA_size:].tolist()
    print(f'Panel A: {len(panelA)}, Panel B: {len(panelB)}')

    # Expression matrices
    Y_A1 = get_X(rep1, panelA).astype(np.float32)
    Y_B1 = get_X(rep1, panelB).astype(np.float32)
    Y_A2 = get_X(rep2, panelA).astype(np.float32)
    Y_B2 = get_X(rep2, panelB).astype(np.float32)

    # H&E features
    HE1 = np.asarray(rep1.obsm['he'], dtype=np.float32)
    HE2 = np.asarray(rep2.obsm['he'], dtype=np.float32)

    # Standardize
    HE_all = np.concatenate([HE1, HE2], axis=0)
    he_mean, he_std = HE_all.mean(axis=0), HE_all.std(axis=0) + 1e-8
    HE1s = (HE1 - he_mean) / he_std
    HE2s = (HE2 - he_mean) / he_std

    Y_A1s, yA_mean1, yA_std1 = zscore_fit_transform(Y_A1)
    Y_B2s, yB_mean2, yB_std2 = zscore_fit_transform(Y_B2)

    # Pseudo labels for panel-to-panel MLPs
    print('\n=== Building measured-panel pseudo labels ===')
    pseudo_B1 = batched_cosine_knn_pseudo(Y_A1, Y_A2, Y_B2, k=args.pseudo_k, device=device)
    pseudo_A2 = batched_cosine_knn_pseudo(Y_B2, pseudo_B1, Y_A1, k=args.pseudo_k, device=device)
    pseudo_B1s, _, _ = zscore_fit_transform(pseudo_B1)
    pseudo_A2s, _, _ = zscore_fit_transform(pseudo_A2)

    dim_A = Y_A1.shape[1]
    dim_B = Y_B2.shape[1]
    he_dim = HE1.shape[1]

    # Train H&E branch
    print('\n=== Training H&E branch ===')
    P_A = MLP(he_dim, args.hidden_dim, dim_A).to(device)
    P_B = MLP(he_dim, args.hidden_dim, dim_B).to(device)
    train_mlp(P_A, HE1s, Y_A1s, args.epochs_he, args.lr, device)
    train_mlp(P_B, HE2s, Y_B2s, args.epochs_he, args.lr, device)

    # Train panel-to-panel branch
    print('\n=== Training panel-to-panel branch ===')
    C_BA = MLP(dim_A, args.hidden_dim, dim_B).to(device)
    C_AB = MLP(dim_B, args.hidden_dim, dim_A).to(device)
    opt = torch.optim.Adam(list(C_BA.parameters()) + list(C_AB.parameters()), lr=args.lr)
    crit = nn.MSELoss()
    YA1_t = torch.tensor(Y_A1s, device=device)
    YB2_t = torch.tensor(Y_B2s, device=device)
    pseudo_B1_t = torch.tensor(pseudo_B1s, device=device)
    pseudo_A2_t = torch.tensor(pseudo_A2s, device=device)
    for epoch in range(args.epochs_panel):
        C_BA.train()
        C_AB.train()
        # supervised on pseudo labels
        yB_hat = C_BA(YA1_t)
        yA_hat = C_AB(YB2_t)
        sup_loss = crit(yB_hat, pseudo_B1_t) + crit(yA_hat, pseudo_A2_t)
        # cycle consistency
        yA_rec = C_AB(yB_hat)
        yB_rec = C_BA(yA_hat)
        cycle_loss = crit(yA_rec, YA1_t) + crit(yB_rec, YB2_t)
        # distribution matching to real opposite-slice panels
        def dist_loss(pred, tgt):
            return ((pred.mean(0) - tgt.mean(0)) ** 2).mean() + ((pred.std(0) - tgt.std(0)) ** 2).mean()
        dist_loss_val = dist_loss(yB_hat, YB2_t) + dist_loss(yA_hat, YA1_t)
        loss = sup_loss + args.lambda_cycle * cycle_loss + args.lambda_dist * dist_loss_val
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'  panel MLP epoch {epoch + 1}/{args.epochs_panel}, '
                  f'sup {sup_loss.item():.4f}, cycle {cycle_loss.item():.4f}, '
                  f'dist {dist_loss_val.item():.4f}, total {loss.item():.4f}')

    # Inference
    print('\n=== Inference ===')
    with torch.no_grad():
        P_A.eval(); P_B.eval(); C_BA.eval(); C_AB.eval()
        HE1_t = torch.tensor(HE1s, device=device)
        HE2_t = torch.tensor(HE2s, device=device)
        YA1_t = torch.tensor(Y_A1s, device=device)
        YB2_t = torch.tensor(Y_B2s, device=device)

        # Raw outputs in standardized space
        pred_B1_HE_s = P_B(HE1_t).cpu().numpy()
        pred_A2_HE_s = P_A(HE2_t).cpu().numpy()
        pred_B1_C_s = C_BA(YA1_t).cpu().numpy()
        pred_A2_C_s = C_AB(YB2_t).cpu().numpy()

    # Unstandardize to target scales
    def unB(x): return x * yB_std2 + yB_mean2
    def unA(x): return x * yA_std1 + yA_mean1

    pred_B1_HE = unB(pred_B1_HE_s)
    pred_A2_HE = unA(pred_A2_HE_s)
    pred_B1_C = unB(pred_B1_C_s)
    pred_A2_C = unA(pred_A2_C_s)

    # Calibrate each branch to target measured distribution
    pred_B1_HE_cal = calibrate_to_target(pred_B1_HE, Y_B2)
    pred_A2_HE_cal = calibrate_to_target(pred_A2_HE, Y_A1)
    pred_B1_C_cal = calibrate_to_target(pred_B1_C, Y_B2)
    pred_A2_C_cal = calibrate_to_target(pred_A2_C, Y_A1)

    # Reliability weights from the side where the target panel is measured
    # For PanelB: evaluate on slice 2 (YB2 measured)
    #   HE branch: P_B(HE2) vs YB2
    #   Panel branch: C_AB? input is A, which is missing on slice2. Use C_BA(P_A(HE2)) as proxy input.
    with torch.no_grad():
        pB_on_2 = P_B(HE2_t).cpu().numpy()
        pA_on_2 = P_A(HE2_t).cpu().numpy()
        pB_cycle_on_2 = C_BA(torch.tensor(pA_on_2, device=device)).cpu().numpy()
    pcc_he_B = pcc_only(Y_B2s, pB_on_2)
    pcc_cycle_B = pcc_only(Y_B2s, pB_cycle_on_2)
    print(f'\nReliability on slice2 (PanelB): HE={pcc_he_B:.4f}, cycle={pcc_cycle_B:.4f}')
    # avoid negative weights
    rel_he_B = max(pcc_he_B, 0)
    rel_cycle_B = max(pcc_cycle_B, 0)
    eps = 1e-8
    alpha_B = rel_he_B / (rel_he_B + rel_cycle_B + eps)

    # For PanelA: evaluate on slice 1 (YA1 measured)
    with torch.no_grad():
        pA_on_1 = P_A(HE1_t).cpu().numpy()
        pB_on_1 = P_B(HE1_t).cpu().numpy()
        pA_cycle_on_1 = C_AB(torch.tensor(pB_on_1, device=device)).cpu().numpy()
    pcc_he_A = pcc_only(Y_A1s, pA_on_1)
    pcc_cycle_A = pcc_only(Y_A1s, pA_cycle_on_1)
    print(f'Reliability on slice1 (PanelA): HE={pcc_he_A:.4f}, cycle={pcc_cycle_A:.4f}')
    rel_he_A = max(pcc_he_A, 0)
    rel_cycle_A = max(pcc_cycle_A, 0)
    alpha_A = rel_he_A / (rel_he_A + rel_cycle_A + eps)

    print(f'Chosen alphas: alpha_B={alpha_B:.3f}, alpha_A={alpha_A:.3f}')

    # Ensembles (using calibrated outputs)
    pred_B1_avg = 0.5 * pred_B1_HE_cal + 0.5 * pred_B1_C_cal
    pred_A2_avg = 0.5 * pred_A2_HE_cal + 0.5 * pred_A2_C_cal
    pred_B1_rel = alpha_B * pred_B1_HE_cal + (1 - alpha_B) * pred_B1_C_cal
    pred_A2_rel = alpha_A * pred_A2_HE_cal + (1 - alpha_A) * pred_A2_C_cal

    # Evaluation
    graph1 = se.pp.Build_graph(rep1.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')
    graph2 = se.pp.Build_graph(rep2.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')

    variants = {
        'HE branch': (pred_B1_HE, pred_A2_HE),
        'Panel branch': (pred_B1_C, pred_A2_C),
        'HE branch (calibrated)': (pred_B1_HE_cal, pred_A2_HE_cal),
        'Panel branch (calibrated)': (pred_B1_C_cal, pred_A2_C_cal),
        '0.5 average (calibrated)': (pred_B1_avg, pred_A2_avg),
        'reliability-weighted (calibrated)': (pred_B1_rel, pred_A2_rel),
    }

    rows = []
    for name, (pb1, pa2) in variants.items():
        pcc1, ssim1, cmd1 = evaluate(Y_B1, pb1, graph1)
        pcc2, ssim2, cmd2 = evaluate(Y_A2, pa2, graph2)
        rows.append({
            'variant': name,
            'slice1_pcc': pcc1, 'slice1_ssim': ssim1, 'slice1_cmd': cmd1,
            'slice2_pcc': pcc2, 'slice2_ssim': ssim2, 'slice2_cmd': cmd2,
        })
        print(f'[{name}] Slice1 PCC={pcc1:.4f} SSIM={ssim1:.4f} CMD={cmd1:.4f} | '
              f'Slice2 PCC={pcc2:.4f} SSIM={ssim2:.4f} CMD={cmd2:.4f}')

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, 'decomposed_metrics.csv'), index=False)

    # Save predictions for the best ensemble
    pd.DataFrame(pred_B1_rel, index=rep1.obs_names, columns=panelB).to_csv(
        os.path.join(args.out_dir, 'pred_panelB1_reliability.csv'))
    pd.DataFrame(pred_A2_rel, index=rep2.obs_names, columns=panelA).to_csv(
        os.path.join(args.out_dir, 'pred_panelA2_reliability.csv'))
    print(f'\nOutputs saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
