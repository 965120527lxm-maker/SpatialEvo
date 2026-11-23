#!/usr/bin/env python3
"""
Ablation: H&E-only Step 2 pseudo-label vs B-panel Step 2 pseudo-label.

Strict Fig.3 protocol. Step 1 always uses H&E MNN for pseudo-YB1.
- Baseline:       Step 2 uses B-panel MNN (YB2 <-> pseudo_YB1 -> YA1).
- Ablation (ind): Step 2 uses a second independent H&E MNN (HE2 <-> HE1 -> YA1).
- Ablation (bid): One shared H&E MNN is used for both pseudo-YB1 and pseudo-YA2.

Never uses held-out Y_B1 or Y_A2.
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
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'fig3'))
import run_fig3_mnn_pseudo as mnn
import run_fig3_latent_mnn as latent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    parser.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    parser.add_argument('--panel_csv', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'panel_split_official.csv'))
    parser.add_argument('--panelA_size', type=int, default=150)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--mnn_k', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default=output_dir('he_only_step2_ablation'))
    return parser.parse_args()


def evaluate(gt, pred, graph):
    import SpatialEx as se
    pcc, pcc_reduce = se.utils.Compute_metrics(gt, pred, metric='pcc')
    ssim, ssim_reduce = se.utils.Compute_metrics(gt, pred, metric='ssim', graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt, pred, metric='cmd')
    return float(pcc_reduce), float(ssim_reduce), float(cmd_reduce)


def build_bidirectional_he_pseudo(he1, he2, yb2, ya1, k=5, mnn_k=20, batch_size=4096, device='cuda'):
    """One shared H&E MNN. Transfer yb2 -> slice1 and ya1 -> slice2 via the same pairs."""
    fwd = mnn.batched_topk_indices(he1, he2, k=mnn_k, batch_size=batch_size, device=device)
    rev = mnn.batched_topk_indices(he2, he1, k=mnn_k, batch_size=batch_size, device=device)
    rev_sets = [set(rev[j].tolist()) for j in range(he2.shape[0])]

    n1 = he1.shape[0]
    n2 = he2.shape[0]
    selected_1to2 = []  # for each slice1 cell, list of matched slice2 cells
    for i in range(n1):
        mnn_idx = [j for j in fwd[i].tolist() if i in rev_sets[j]]
        if len(mnn_idx) == 0:
            mnn_idx = fwd[i, :k].tolist()
        selected_1to2.append(mnn_idx)

    # pseudo-YB1: for each slice1 cell, average YB2 of its matched slice2 cells
    pseudo_yb1 = np.stack([yb2[idx].mean(axis=0) for idx in selected_1to2]).astype(np.float32)

    # Build reverse map: for each slice2 cell, which slice1 cells matched to it
    selected_2to1 = [[] for _ in range(n2)]
    for i, js in enumerate(selected_1to2):
        for j in js:
            selected_2to1[j].append(i)

    # pseudo-YA2: for each slice2 cell, average YA1 of slice1 cells that matched to it
    pseudo_ya2 = np.zeros((n2, ya1.shape[1]), dtype=np.float32)
    for j in range(n2):
        if len(selected_2to1[j]) > 0:
            pseudo_ya2[j] = ya1[selected_2to1[j]].mean(axis=0)
        else:
            # fall back to raw kNN in HE space from slice2 to slice1
            knn_idx = mnn.batched_topk_indices(he2[j:j+1], he1, k=k, batch_size=batch_size, device=device)[0]
            pseudo_ya2[j] = ya1[knn_idx].mean(axis=0)

    return pseudo_yb1, pseudo_ya2


def train_panel_mlps(Y_A1, pseudo_B1, Y_B2, pseudo_A2, hidden_dim, epochs, lr, device):
    model1 = mnn.PanelMLP(Y_A1.shape[1], hidden_dim, pseudo_B1.shape[1]).to(device)
    model2 = mnn.PanelMLP(Y_B2.shape[1], hidden_dim, pseudo_A2.shape[1]).to(device)
    opt = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=lr)
    crit = nn.MSELoss()
    X1 = torch.tensor(mnn.zscore(Y_A1), dtype=torch.float32, device=device)
    X2 = torch.tensor(mnn.zscore(Y_B2), dtype=torch.float32, device=device)
    T1 = torch.tensor(mnn.zscore(pseudo_B1), dtype=torch.float32, device=device)
    T2 = torch.tensor(mnn.zscore(pseudo_A2), dtype=torch.float32, device=device)
    for _ in tqdm(range(epochs), desc='train MLP', leave=False):
        opt.zero_grad()
        loss = crit(model1(X1), T1) + crit(model2(X2), T2)
        loss.backward()
        opt.step()
    return model1, model2


def predict(model, x, y_mean, y_std, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(mnn.zscore(x), dtype=torch.float32, device=device)).cpu().numpy()
    return pred * y_std + y_mean


def run_variant(name, pseudo_B1, pseudo_A2, Y_A1, Y_B1, Y_A2, Y_B2,
                graph1, graph2, args, device):
    print(f'\n=== {name} ===')
    direct1 = evaluate(Y_B1, pseudo_B1, graph1)
    direct2 = evaluate(Y_A2, pseudo_A2, graph2)

    model1, model2 = train_panel_mlps(
        Y_A1, pseudo_B1, Y_B2, pseudo_A2,
        args.hidden_dim, args.epochs, args.lr, device)

    yB_mean, yB_std = Y_B2.mean(axis=0), Y_B2.std(axis=0) + 1e-8
    yA_mean, yA_std = Y_A1.mean(axis=0), Y_A1.std(axis=0) + 1e-8

    pred_B1 = predict(model1, Y_A1, yB_mean, yB_std, device)
    pred_A2 = predict(model2, Y_B2, yA_mean, yA_std, device)
    learned1 = evaluate(Y_B1, pred_B1, graph1)
    learned2 = evaluate(Y_A2, pred_A2, graph2)

    return {
        'variant': name,
        'slice1_direct_pcc': direct1[0], 'slice1_direct_ssim': direct1[1], 'slice1_direct_cmd': direct1[2],
        'slice2_direct_pcc': direct2[0], 'slice2_direct_ssim': direct2[1], 'slice2_direct_cmd': direct2[2],
        'slice1_learned_pcc': learned1[0], 'slice1_learned_ssim': learned1[1], 'slice1_learned_cmd': learned1[2],
        'slice2_learned_pcc': learned2[0], 'slice2_learned_ssim': learned2[1], 'slice2_learned_cmd': learned2[2],
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rep1 = mnn.load_slice(args.rep1, args.data_root)
    rep2 = mnn.load_slice(args.rep2, args.data_root)

    if args.panel_csv and os.path.exists(args.panel_csv):
        panel_df = pd.read_csv(args.panel_csv)
        panelA = panel_df[panel_df['panel'] == 'panelA']['gene'].astype(str).tolist()
        panelB = panel_df[panel_df['panel'] == 'panelB']['gene'].astype(str).tolist()
        print(f'Loaded panel split from {args.panel_csv}: A={len(panelA)}, B={len(panelB)}')
    else:
        genes = rep1.var_names.values
        np.random.seed(args.seed)
        np.random.shuffle(genes)
        panelA = genes[:args.panelA_size].tolist()
        panelB = genes[args.panelA_size:].tolist()
        print(f'Using random split seed={args.seed}: A={len(panelA)}, B={len(panelB)}')

    Y_A1 = mnn.get_X(rep1, panelA).astype(np.float32)
    Y_B1 = mnn.get_X(rep1, panelB).astype(np.float32)
    Y_A2 = mnn.get_X(rep2, panelA).astype(np.float32)
    Y_B2 = mnn.get_X(rep2, panelB).astype(np.float32)
    HE1 = np.asarray(rep1.obsm['he'], dtype=np.float32)
    HE2 = np.asarray(rep2.obsm['he'], dtype=np.float32)

    import SpatialEx as se
    graph1 = se.pp.Build_graph(rep1.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')
    graph2 = se.pp.Build_graph(rep2.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')

    # Baseline + independent H&E: shared Step 1 H&E MNN for pseudo-YB1
    print('Building shared Step 1 pseudo-YB1 via H&E MNN...')
    pseudo_B1 = mnn.build_mnn_pseudo(HE1, HE2, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)

    # Baseline Step 2: B-panel MNN
    print('Building baseline Step 2 pseudo-YA2 via B-panel MNN...')
    pseudo_A2_bpanel = mnn.build_mnn_pseudo(Y_B2, pseudo_B1, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)

    # Ablation Step 2: independent H&E MNN
    print('Building ablation Step 2 pseudo-YA2 via independent H&E MNN...')
    pseudo_A2_he_ind = mnn.build_mnn_pseudo(HE2, HE1, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)

    # Ablation Step 2: single shared H&E MNN for both directions
    print('Building ablation pseudo-labels via single shared H&E MNN...')
    pseudo_B1_bid, pseudo_A2_he_bid = build_bidirectional_he_pseudo(
        HE1, HE2, Y_B2, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)

    rows = []
    rows.append(run_variant(
        'Step2=B-panel MNN (baseline)', pseudo_B1, pseudo_A2_bpanel,
        Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device))
    rows.append(run_variant(
        'Step2=H&E MNN independent (ablation)', pseudo_B1, pseudo_A2_he_ind,
        Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device))
    rows.append(run_variant(
        'Step2=H&E MNN single shared (ablation)', pseudo_B1_bid, pseudo_A2_he_bid,
        Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, 'he_step2_ablation_results.csv')
    df.to_csv(out_csv, index=False)
    print('\n=== H&E Step 2 ablation results (3 variants) ===')
    print(df.to_string(index=False))
    print(f'\nSaved to: {out_csv}')


if __name__ == '__main__':
    main()
