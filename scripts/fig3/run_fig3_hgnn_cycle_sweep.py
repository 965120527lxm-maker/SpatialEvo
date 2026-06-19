#!/usr/bin/env python3
"""
HGNN + Cycle (SpatialExP) hyperparameter sweep on official Fig.3 split.

Tunable axes:
  - num_neighbors: spatial graph k (hypergraph construction)
  - hidden_dim / translator_hidden_dim: model capacity
  - lambda_recon / lambda_map / lambda_cycle: loss term weights

Default grid (phase 1): num_neighbors in {5, 7, 10, 15}
  with hidden_dim=512, all lambdas=1.0, epochs=500
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'fig3'))
from exp_paths import output_dir

import argparse
import contextlib
import gc
import io
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import SpatialEx as se


DEFAULT_NUM_NEIGHBORS = [5, 7, 10, 15]
DEFAULT_HIDDEN_DIMS = [512]
DEFAULT_LAMBDA_CYCLES = [1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    p.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    p.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    p.add_argument('--panel_csv', type=str,
                   default=os.path.join(PROJECT_ROOT, 'data', 'panel_split_official.csv'))
    p.add_argument('--num_neighborss', type=int, nargs='+', default=DEFAULT_NUM_NEIGHBORS)
    p.add_argument('--hidden_dims', type=int, nargs='+', default=DEFAULT_HIDDEN_DIMS)
    p.add_argument('--translator_hidden_dims', type=int, nargs='+', default=None,
                   help='Defaults to matching hidden_dim per config if unset')
    p.add_argument('--lambda_recons', type=float, nargs='+', default=[1.0])
    p.add_argument('--lambda_maps', type=float, nargs='+', default=[1.0])
    p.add_argument('--lambda_cycles', type=float, nargs='+', default=DEFAULT_LAMBDA_CYCLES)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--out_dir', type=str, default=output_dir('hgnn_cycle_sweep_official'))
    return p.parse_args()


def load_slice(path, data_root):
    adata = sc.read_h5ad(os.path.join(data_root, path))
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata


def load_panels(panel_csv, rep1, seed):
    panel_df = pd.read_csv(panel_csv)
    panelA = panel_df[panel_df['panel'] == 'panelA']['gene'].astype(str).tolist()
    panelB = panel_df[panel_df['panel'] == 'panelB']['gene'].astype(str).tolist()
    print(f'Panel split: A={len(panelA)}, B={len(panelB)}', flush=True)
    return panelA, panelB


def evaluate(gt_adata, pred, graph):
    gt = gt_adata.X.toarray() if hasattr(gt_adata.X, 'toarray') else np.asarray(gt_adata.X)
    pcc, pcc_r = se.utils.Compute_metrics(gt, pred, metric='pcc')
    ssim, ssim_r = se.utils.Compute_metrics(gt, pred, metric='ssim', graph=graph)
    cmd, cmd_r = se.utils.Compute_metrics(gt, pred, metric='cmd')
    return float(pcc_r), float(ssim_r), float(cmd_r)


def config_key(num_neighbors, hidden_dim, translator_hidden_dim,
               lambda_recon, lambda_map, lambda_cycle):
    return (num_neighbors, hidden_dim, translator_hidden_dim,
            lambda_recon, lambda_map, lambda_cycle)


def load_done(out_dir):
    path = os.path.join(out_dir, 'hgnn_cycle_sweep.csv')
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    keys = set()
    for r in df.to_dict('records'):
        keys.add(config_key(
            int(r['num_neighbors']), int(r['hidden_dim']), int(r['translator_hidden_dim']),
            float(r['lambda_recon']), float(r['lambda_map']), float(r['lambda_cycle'])))
    return keys


def append_row(row, out_dir):
    path = os.path.join(out_dir, 'hgnn_cycle_sweep.csv')
    chunk = pd.DataFrame([row])
    if os.path.exists(path):
        pd.concat([pd.read_csv(path), chunk], ignore_index=True).to_csv(path, index=False)
    else:
        chunk.to_csv(path, index=False)
    print(f'Appended nn={row["num_neighbors"]}, h={row["hidden_dim"]}, '
          f'lc={row["lambda_cycle"]} -> {path}', flush=True)


def free_mem(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def build_configs(args):
    thd_list = args.translator_hidden_dims
    configs = []
    for nn, hd, lr, lm, lc in itertools.product(
            args.num_neighborss, args.hidden_dims,
            args.lambda_recons, args.lambda_maps, args.lambda_cycles):
        thd = hd if thd_list is None else thd_list[0]
        if thd_list is not None and len(thd_list) == len(args.hidden_dims):
            idx = args.hidden_dims.index(hd)
            thd = thd_list[idx]
        configs.append((nn, hd, thd, lr, lm, lc))
    return configs


def run_config(cfg, rep1_A, rep2_B, rep1_B, rep2_A, args, device):
    nn, hd, thd, lam_r, lam_m, lam_c = cfg
    print(f'\n=== num_neighbors={nn}, hidden_dim={hd}, translator={thd}, '
          f'lambda_recon={lam_r}, lambda_map={lam_m}, lambda_cycle={lam_c} ===', flush=True)

    graph1 = se.pp.Build_hypergraph_spatial_and_HE(
        rep1_A, nn, graph_kind='spatial', return_type='csr')
    graph2 = se.pp.Build_hypergraph_spatial_and_HE(
        rep2_B, nn, graph_kind='spatial', return_type='csr')

    eval_graph1 = se.pp.Build_graph(
        rep1_A.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo')
    eval_graph2 = se.pp.Build_graph(
        rep2_B.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo')

    trainer = se.SpatialExP(
        rep1_A, rep2_B, graph1, graph2,
        device=device, epochs=args.epochs, lr=args.lr,
        hidden_dim=hd, translator_hidden_dim=thd,
        lambda_recon=lam_r, lambda_map=lam_m, lambda_cycle=lam_c,
        seed=args.seed,
    )

    ctx = contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext()
    with ctx:
        trainer.train()

    pred_B1 = trainer.inference_indirect(rep1_A.obsm['he'], graph1, panel='panelB')
    pred_A2 = trainer.inference_indirect(rep2_B.obsm['he'], graph2, panel='panelA')

    gt_B1 = rep1_B[rep1_A.obs_names]
    gt_A2 = rep2_A[rep2_B.obs_names]
    m1 = evaluate(gt_B1, pred_B1, eval_graph1)
    m2 = evaluate(gt_A2, pred_A2, eval_graph2)

    del trainer, graph1, graph2, pred_B1, pred_A2
    free_mem(device)

    return {
        'num_neighbors': nn,
        'hidden_dim': hd,
        'translator_hidden_dim': thd,
        'lambda_recon': lam_r,
        'lambda_map': lam_m,
        'lambda_cycle': lam_c,
        'epochs': args.epochs,
        'lr': args.lr,
        'slice1_pcc': m1[0],
        'slice1_ssim': m1[1],
        'slice1_cmd': m1[2],
        'slice2_pcc': m2[0],
        'slice2_ssim': m2[1],
        'slice2_cmd': m2[2],
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rep1 = load_slice(args.rep1, args.data_root)
    rep2 = load_slice(args.rep2, args.data_root)
    panelA, panelB = load_panels(args.panel_csv, rep1, args.seed)

    rep1_A = rep1[:, panelA].copy()
    rep1_B = rep1[:, panelB].copy()
    rep2_A = rep2[:, panelA].copy()
    rep2_B = rep2[:, panelB].copy()

    all_configs = build_configs(args)
    done = load_done(args.out_dir) if args.resume else set()
    pending = [c for c in all_configs if config_key(*c) not in done]
    print(f'Configs: {len(all_configs)} total, {len(done)} done, {len(pending)} pending', flush=True)

    for cfg in pending:
        row = run_config(cfg, rep1_A, rep2_B, rep1_B, rep2_A, args, device)
        append_row(row, args.out_dir)

    path = os.path.join(args.out_dir, 'hgnn_cycle_sweep.csv')
    if not os.path.exists(path):
        print('No results.', flush=True)
        return

    df = pd.read_csv(path).sort_values(['slice1_pcc', 'slice2_pcc'], ascending=False)
    cols = ['num_neighbors', 'hidden_dim', 'lambda_cycle',
            'slice1_pcc', 'slice1_ssim', 'slice2_pcc', 'slice2_ssim']
    print('\n=== HGNN+Cycle sweep (official split) ===', flush=True)
    print(df[cols].to_string(index=False), flush=True)

    best = df.loc[df['slice2_pcc'].idxmax()]
    print(f'\nBest Slice2 PCC: nn={int(best.num_neighbors)}, h={int(best.hidden_dim)}, '
          f'lc={best.lambda_cycle}, PCC={best.slice2_pcc:.4f}', flush=True)


if __name__ == '__main__':
    main()
