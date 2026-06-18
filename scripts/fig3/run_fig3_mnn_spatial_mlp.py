#!/usr/bin/env python3
"""
Strict MNN pseudo-label MLP with optional within-slice spatial context.

Cross-slice supervision is unchanged (strict Fig.3 MNN bridges).
Within-slice information is injected into MLP inputs via spatial KNN graph.

Modes:
  none    : baseline MLP on measured panel only (same as run_fig3_mnn_pseudo)
  concat  : [self, neighbor-mean panel]
  blend   : (1-w)*self + w*neighbor-mean panel
  delta   : MLP predicts residual; output = neighbor-mean + MLP(x_ctx)
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

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'fig3'))
import run_fig3_mnn_pseudo as mnn


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
    parser.add_argument('--panel_csv', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'panel_split_official.csv'))
    parser.add_argument('--spatial_mode', type=str, default='concat',
                        choices=['none', 'concat', 'blend', 'delta'])
    parser.add_argument('--spatial_weight', type=float, default=0.5,
                        help='Blend weight for neighbor mean (blend mode) or unused for concat')
    parser.add_argument('--num_neighbors', type=int, default=7,
                        help='Within-slice spatial KNN for context features')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--mnn_k', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'conditional',
                                            'fig3_mnn_spatial_mlp_official'))
    return parser.parse_args()


def neighbor_mean(graph, X):
    """Row-normalized spatial graph @ X -> neighbor-averaged features."""
    if scipy.sparse.issparse(graph):
        adj = graph.tocsr()
    else:
        adj = scipy.sparse.csr_matrix(graph)
    return np.asarray(adj @ X, dtype=np.float32)


def build_spatial_graph(adata, num_neighbors):
    return se.pp.Build_graph(
        adata.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo',
        num_neighbors=num_neighbors,
    )


def spatialize_input(X, graph, mode, spatial_weight):
    nb = neighbor_mean(graph, X)
    if mode == 'none':
        return X.astype(np.float32)
    if mode == 'concat':
        return np.concatenate([X, nb], axis=1).astype(np.float32)
    if mode == 'blend':
        w = spatial_weight
        return ((1.0 - w) * X + w * nb).astype(np.float32)
    if mode == 'delta':
        return np.concatenate([X, nb], axis=1).astype(np.float32)
    raise ValueError(mode)


def train_spatial_mlp(X1, graph1, Y1_pseudo, X2, graph2, Y2_pseudo,
                      out_dim1, out_dim2, spatial_mode, spatial_weight,
                      hidden_dim, epochs, lr, device):
    in_ctx1 = spatialize_input(X1, graph1, spatial_mode, spatial_weight)
    in_ctx2 = spatialize_input(X2, graph2, spatial_mode, spatial_weight)
    base1 = neighbor_mean(graph1, Y1_pseudo)
    base2 = neighbor_mean(graph2, Y2_pseudo)

    model1 = PanelMLP(in_ctx1.shape[1], hidden_dim, out_dim1).to(device)
    model2 = PanelMLP(in_ctx2.shape[1], hidden_dim, out_dim2).to(device)
    opt = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=lr)
    crit = nn.MSELoss()

    X1_t = torch.tensor(mnn.zscore(in_ctx1), dtype=torch.float32, device=device)
    X2_t = torch.tensor(mnn.zscore(in_ctx2), dtype=torch.float32, device=device)
    Y1_t = torch.tensor(mnn.zscore(Y1_pseudo), dtype=torch.float32, device=device)
    Y2_t = torch.tensor(mnn.zscore(Y2_pseudo), dtype=torch.float32, device=device)
    B1_t = torch.tensor(mnn.zscore(base1), dtype=torch.float32, device=device)
    B2_t = torch.tensor(mnn.zscore(base2), dtype=torch.float32, device=device)

    for epoch in tqdm(range(epochs), desc=f'train spatial MLP ({spatial_mode})'):
        model1.train()
        model2.train()
        opt.zero_grad()
        out1 = model1(X1_t)
        out2 = model2(X2_t)
        if spatial_mode == 'delta':
            out1 = out1 + B1_t
            out2 = out2 + B2_t
        loss = crit(out1, Y1_t) + crit(out2, Y2_t)
        loss.backward()
        opt.step()

    ctx = {'base1': base1, 'base2': base2}
    return model1, model2, ctx


def predict_spatial(model, X, graph, spatial_mode, spatial_weight, ctx_base,
                    y_mean, y_std, device):
    in_ctx = spatialize_input(X, graph, spatial_mode, spatial_weight)
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(mnn.zscore(in_ctx), dtype=torch.float32, device=device)
        pred = model(x_t).cpu().numpy()
        if spatial_mode == 'delta':
            pred = pred + mnn.zscore(ctx_base)
    pred = pred * y_std + y_mean
    return pred


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rep1 = mnn.load_slice(args.rep1, args.data_root)
    rep2 = mnn.load_slice(args.rep2, args.data_root)

    panel_df = pd.read_csv(args.panel_csv)
    panelA = panel_df[panel_df.panel == 'panelA'].gene.astype(str).tolist()
    panelB = panel_df[panel_df.panel == 'panelB'].gene.astype(str).tolist()
    print(f'Panel A={len(panelA)}, B={len(panelB)}, spatial_mode={args.spatial_mode}')

    Y_A1 = mnn.get_X(rep1, panelA).astype(np.float32)
    Y_B1 = mnn.get_X(rep1, panelB).astype(np.float32)
    Y_A2 = mnn.get_X(rep2, panelA).astype(np.float32)
    Y_B2 = mnn.get_X(rep2, panelB).astype(np.float32)
    HE1 = np.asarray(rep1.obsm['he'], dtype=np.float32)
    HE2 = np.asarray(rep2.obsm['he'], dtype=np.float32)

    graph1 = build_spatial_graph(rep1, args.num_neighbors)
    graph2 = build_spatial_graph(rep2, args.num_neighbors)

    print('=== Strict MNN pseudo labels ===')
    pseudo_B1 = mnn.build_mnn_pseudo(HE1, HE2, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)
    pseudo_A2 = mnn.build_mnn_pseudo(Y_B2, pseudo_B1, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)

    model_B1, model_A2, ctx = train_spatial_mlp(
        Y_A1, graph1, pseudo_B1, Y_B2, graph2, pseudo_A2,
        Y_B2.shape[1], Y_A1.shape[1],
        args.spatial_mode, args.spatial_weight,
        args.hidden_dim, args.epochs, args.lr, device,
    )

    yB_mean, yB_std = Y_B2.mean(0), Y_B2.std(0) + 1e-8
    yA_mean, yA_std = Y_A1.mean(0), Y_A1.std(0) + 1e-8

    base_B1 = ctx['base1']
    base_A2 = ctx['base2']

    pred_B1 = predict_spatial(
        model_B1, Y_A1, graph1, args.spatial_mode, args.spatial_weight,
        base_B1, yB_mean, yB_std, device,
    )
    pred_A2 = predict_spatial(
        model_A2, Y_B2, graph2, args.spatial_mode, args.spatial_weight,
        base_A2, yA_mean, yA_std, device,
    )

    m1 = mnn.evaluate(Y_B1, pred_B1, graph1)
    m2 = mnn.evaluate(Y_A2, pred_A2, graph2)
    print(f'\nSlice1 PanelB: PCC={m1[0]:.4f}, SSIM={m1[1]:.4f}, CMD={m1[2]:.4f}')
    print(f'Slice2 PanelA: PCC={m2[0]:.4f}, SSIM={m2[1]:.4f}, CMD={m2[2]:.4f}')

    summary = pd.DataFrame([{
        'model': f'mnn_spatial_mlp_{args.spatial_mode}',
        'spatial_mode': args.spatial_mode,
        'spatial_weight': args.spatial_weight,
        'num_neighbors': args.num_neighbors,
        'slice1_pcc': m1[0], 'slice1_ssim': m1[1], 'slice1_cmd': m1[2],
        'slice2_pcc': m2[0], 'slice2_ssim': m2[1], 'slice2_cmd': m2[2],
    }])
    summary.to_csv(os.path.join(args.out_dir, 'metrics_spatial_mlp.csv'), index=False)
    pd.DataFrame(pred_B1, index=rep1.obs_names, columns=panelB).to_csv(
        os.path.join(args.out_dir, 'pred_panelB1.csv'))
    pd.DataFrame(pred_A2, index=rep2.obs_names, columns=panelA).to_csv(
        os.path.join(args.out_dir, 'pred_panelA2.csv'))
    print(f'Saved to {args.out_dir}')


if __name__ == '__main__':
    main()
