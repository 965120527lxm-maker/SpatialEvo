#!/usr/bin/env python3
"""Per-gene PCC analysis for raw kNN vs MNN pseudo-label / MLP predictions."""

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
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'conditional', 'fig3_per_gene_pcc'))
    return parser.parse_args()


def per_gene_pcc(y_true, y_pred):
    pccs = []
    for g in range(y_true.shape[1]):
        a = y_true[:, g]
        b = y_pred[:, g]
        if a.std() < 1e-8 or b.std() < 1e-8:
            pccs.append(np.nan)
        else:
            pccs.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(pccs)


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
    HE1 = np.asarray(rep1.obsm['he'], dtype=np.float32)
    HE2 = np.asarray(rep2.obsm['he'], dtype=np.float32)

    # Raw kNN (strict H&E bridge)
    print('Building raw kNN pseudo-labels...')
    raw_B1 = mnn.build_raw_pseudo(HE1, HE2, Y_B2, k=args.k, device=device)
    raw_A2 = mnn.build_raw_pseudo(Y_B2, raw_B1, Y_A1, k=args.k, device=device)
    model_raw1, model_raw2 = mnn.train_panel_mlp(
        Y_A1, raw_B1, Y_B2, raw_A2,
        Y_A1.shape[1], Y_B2.shape[1], Y_A1.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    yB_mean, yB_std = Y_B2.mean(axis=0), Y_B2.std(axis=0) + 1e-8
    yA_mean, yA_std = Y_A1.mean(axis=0), Y_A1.std(axis=0) + 1e-8
    def _predict(model, x, y_mean, y_std):
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(mnn.zscore(x), dtype=torch.float32, device=device)).cpu().numpy()
        return pred * y_std + y_mean

    pred_raw_B1 = _predict(model_raw1, Y_A1, yB_mean, yB_std)
    pred_raw_A2 = _predict(model_raw2, Y_B2, yA_mean, yA_std)

    # MNN kNN
    print('Building MNN pseudo-labels...')
    mnn_B1 = mnn.build_mnn_pseudo(HE1, HE2, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)
    mnn_A2 = mnn.build_mnn_pseudo(Y_B2, mnn_B1, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)
    model_mnn1, model_mnn2 = mnn.train_panel_mlp(
        Y_A1, mnn_B1, Y_B2, mnn_A2,
        Y_A1.shape[1], Y_B2.shape[1], Y_A1.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)
    pred_mnn_B1 = _predict(model_mnn1, Y_A1, yB_mean, yB_std)
    pred_mnn_A2 = _predict(model_mnn2, Y_B2, yA_mean, yA_std)

    # Per-gene PCC
    results = {
        'gene': panelB + panelA,
        'slice': ['Slice1'] * len(panelB) + ['Slice2'] * len(panelA),
        'raw_direct_pcc': np.concatenate([per_gene_pcc(Y_B1, raw_B1), per_gene_pcc(Y_A2, raw_A2)]),
        'raw_learned_pcc': np.concatenate([per_gene_pcc(Y_B1, pred_raw_B1), per_gene_pcc(Y_A2, pred_raw_A2)]),
        'mnn_direct_pcc': np.concatenate([per_gene_pcc(Y_B1, mnn_B1), per_gene_pcc(Y_A2, mnn_A2)]),
        'mnn_learned_pcc': np.concatenate([per_gene_pcc(Y_B1, pred_mnn_B1), per_gene_pcc(Y_A2, pred_mnn_A2)]),
    }
    df = pd.DataFrame(results)
    df['mnn_gain'] = df['mnn_learned_pcc'] - df['raw_learned_pcc']
    df.to_csv(os.path.join(args.out_dir, 'per_gene_pcc.csv'), index=False)

    print('\n=== Per-gene PCC summary ===')
    print(df.groupby('slice')[['raw_direct_pcc', 'raw_learned_pcc', 'mnn_direct_pcc', 'mnn_learned_pcc']].mean())

    print('\n=== Top MNN gains (Slice2) ===')
    print(df[df['slice'] == 'Slice2'].nlargest(10, 'mnn_gain')[['gene', 'raw_learned_pcc', 'mnn_learned_pcc', 'mnn_gain']].to_string(index=False))

    print('\n=== Top MNN gains (Slice1) ===')
    print(df[df['slice'] == 'Slice1'].nlargest(10, 'mnn_gain')[['gene', 'raw_learned_pcc', 'mnn_learned_pcc', 'mnn_gain']].to_string(index=False))

    print(f'\nOutputs saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
