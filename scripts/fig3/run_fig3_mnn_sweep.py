#!/usr/bin/env python3
"""
Strict Fig.3 MNN pseudo-label parameter sensitivity sweep.

Protocol (same as run_fig3_mnn_pseudo / alignment benchmark):
  Step1: HE1 <-> HE2 MNN -> pseudo YB1
  Step2: YB2 <-> pseudo YB1 MNN -> pseudo YA2
  Downstream: dual Panel MLP on pseudo labels

Default grid: k in {3, 5, 10, 20}, mnn_k in {10, 20, 50, 100}
Default panel: data/panel_split_official.csv
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG3_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'fig3')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FIG3_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'fig3'))
from exp_paths import output_dir

import argparse
import contextlib
import gc
import io
import numpy as np
import pandas as pd
import torch

import SpatialEx as se
import run_fig3_mnn_pseudo as mnn


DEFAULT_KS = [3, 5, 10, 20]
DEFAULT_MNN_KS = [10, 20, 50, 100]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    parser.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    parser.add_argument('--panel_csv', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'panel_split_official.csv'))
    parser.add_argument('--panelA_size', type=int, default=150,
                        help='Used only when panel_csv is missing (random split fallback)')
    parser.add_argument('--ks', type=int, nargs='+', default=DEFAULT_KS,
                        help='Fallback neighbor count when no MNN match')
    parser.add_argument('--mnn_ks', type=int, nargs='+', default=DEFAULT_MNN_KS,
                        help='Forward/reverse kNN pool size for MNN filtering')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='kNN mini-batch size (lower if GPU OOM)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true',
                        help='Skip (k, mnn_k) configs already in mnn_sweep.csv')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose kNN progress prints')
    parser.add_argument('--out_dir', type=str, default=output_dir('mnn_sweep_official'))
    return parser.parse_args()


def load_panels(args, rep1):
    if args.panel_csv and os.path.exists(args.panel_csv):
        panel_df = pd.read_csv(args.panel_csv)
        panelA = panel_df[panel_df['panel'] == 'panelA']['gene'].astype(str).tolist()
        panelB = panel_df[panel_df['panel'] == 'panelB']['gene'].astype(str).tolist()
        print(f'Loaded panel split from {args.panel_csv}: A={len(panelA)}, B={len(panelB)}')
        return panelA, panelB

    genes = rep1.var_names.values
    np.random.seed(args.seed)
    np.random.shuffle(genes)
    panelA = genes[:args.panelA_size].tolist()
    panelB = genes[args.panelA_size:].tolist()
    print(f'Random panel split: A={len(panelA)}, B={len(panelB)}')
    return panelA, panelB


def free_device_memory(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def compute_mnn_match_rate(query, ref, mnn_k, device, batch_size):
    fwd = mnn.batched_topk_indices(query, ref, k=mnn_k, batch_size=batch_size, device=device)
    rev = mnn.batched_topk_indices(ref, query, k=mnn_k, batch_size=batch_size, device=device)
    rev_sets = [set(rev[j].tolist()) for j in range(ref.shape[0])]
    matched = sum(
        1 for i in range(query.shape[0])
        if any(i in rev_sets[j] for j in fwd[i].tolist())
    )
    return matched / max(query.shape[0], 1)


def predict(model, x, y_mean, y_std, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(mnn.zscore(x), dtype=torch.float32, device=device)).cpu().numpy()
    return pred * y_std + y_mean


def load_done_configs(out_dir):
    out_csv = os.path.join(out_dir, 'mnn_sweep.csv')
    if not os.path.exists(out_csv):
        return set()
    df = pd.read_csv(out_csv)
    return {(int(r.k), int(r.mnn_k)) for r in df.itertuples()}


def build_mnn_pseudo_quiet(*args, quiet=False, **kwargs):
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        return mnn.build_mnn_pseudo(*args, **kwargs)


def run_config(k, mnn_k, HE1, HE2, Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device):
    print(f'\n=== k={k}, mnn_k={mnn_k} ===', flush=True)
    bs = args.batch_size
    pseudo_B1 = build_mnn_pseudo_quiet(
        HE1, HE2, Y_B2, k=k, mnn_k=mnn_k, batch_size=bs, device=device, quiet=args.quiet)
    free_device_memory(device)
    pseudo_A2 = build_mnn_pseudo_quiet(
        Y_B2, pseudo_B1, Y_A1, k=k, mnn_k=mnn_k, batch_size=bs, device=device, quiet=args.quiet)
    free_device_memory(device)

    direct1 = mnn.evaluate(Y_B1, pseudo_B1, graph1)
    direct2 = mnn.evaluate(Y_A2, pseudo_A2, graph2)

    model1, model2 = mnn.train_panel_mlp(
        Y_A1, pseudo_B1, Y_B2, pseudo_A2,
        Y_A1.shape[1], Y_B2.shape[1], Y_A1.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    yB_mean, yB_std = Y_B2.mean(axis=0), Y_B2.std(axis=0) + 1e-8
    yA_mean, yA_std = Y_A1.mean(axis=0), Y_A1.std(axis=0) + 1e-8
    pred_B1 = predict(model1, Y_A1, yB_mean, yB_std, device)
    pred_A2 = predict(model2, Y_B2, yA_mean, yA_std, device)
    learned1 = mnn.evaluate(Y_B1, pred_B1, graph1)
    learned2 = mnn.evaluate(Y_A2, pred_A2, graph2)

    step1_rate = compute_mnn_match_rate(HE1, HE2, mnn_k, device, bs)
    free_device_memory(device)
    step2_rate = compute_mnn_match_rate(Y_B2, pseudo_B1, mnn_k, device, bs)
    free_device_memory(device)

    del model1, model2, pseudo_B1, pseudo_A2, pred_B1, pred_A2

    return {
        'k': k,
        'mnn_k': mnn_k,
        'step1_mnn_match_rate': step1_rate,
        'step2_mnn_match_rate': step2_rate,
        'slice1_direct_pcc': direct1[0],
        'slice1_direct_ssim': direct1[1],
        'slice1_direct_cmd': direct1[2],
        'slice1_learned_pcc': learned1[0],
        'slice1_learned_ssim': learned1[1],
        'slice1_learned_cmd': learned1[2],
        'slice2_direct_pcc': direct2[0],
        'slice2_direct_ssim': direct2[1],
        'slice2_direct_cmd': direct2[2],
        'slice2_learned_pcc': learned2[0],
        'slice2_learned_ssim': learned2[1],
        'slice2_learned_cmd': learned2[2],
    }


def append_row(row, out_dir):
    out_csv = os.path.join(out_dir, 'mnn_sweep.csv')
    chunk = pd.DataFrame([row])
    if os.path.exists(out_csv):
        pd.concat([pd.read_csv(out_csv), chunk], ignore_index=True).to_csv(out_csv, index=False)
    else:
        chunk.to_csv(out_csv, index=False)
    print(f'Appended k={row["k"]}, mnn_k={row["mnn_k"]} -> {out_csv}')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rep1 = mnn.load_slice(args.rep1, args.data_root)
    rep2 = mnn.load_slice(args.rep2, args.data_root)
    panelA, panelB = load_panels(args, rep1)

    Y_A1 = mnn.get_X(rep1, panelA).astype(np.float32)
    Y_B1 = mnn.get_X(rep1, panelB).astype(np.float32)
    Y_A2 = mnn.get_X(rep2, panelA).astype(np.float32)
    Y_B2 = mnn.get_X(rep2, panelB).astype(np.float32)
    HE1 = np.asarray(rep1.obsm['he'], dtype=np.float32)
    HE2 = np.asarray(rep2.obsm['he'], dtype=np.float32)

    graph1 = se.pp.Build_graph(rep1.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')
    graph2 = se.pp.Build_graph(rep2.obsm['spatial'], graph_type='knn', weighted='gaussian',
                               apply_normalize='row', return_type='coo')

    configs = [(k, mnn_k) for k in args.ks for mnn_k in args.mnn_ks if k <= mnn_k]
    done = load_done_configs(args.out_dir) if args.resume else set()
    pending = [(k, mk) for k, mk in configs if (k, mk) not in done]
    print(f'Sweep configs: {len(configs)} total, {len(done)} done, {len(pending)} pending')
    if pending:
        print('Pending:', pending)

    rows = []
    for k, mnn_k in pending:
        row = run_config(k, mnn_k, HE1, HE2, Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device)
        rows.append(row)
        append_row(row, args.out_dir)

    out_csv = os.path.join(args.out_dir, 'mnn_sweep.csv')
    if not os.path.exists(out_csv):
        print('No results produced.')
        return
    df = pd.read_csv(out_csv)
    print('\n=== MNN sweep (official split) ===')
    print(df.sort_values(['slice1_learned_pcc', 'slice2_learned_pcc'], ascending=False).to_string(index=False))

    best_s1 = df.loc[df['slice1_learned_pcc'].idxmax()]
    best_s2 = df.loc[df['slice2_learned_pcc'].idxmax()]
    print(f'\nBest Slice1 learned PCC: k={int(best_s1.k)}, mnn_k={int(best_s1.mnn_k)}, '
          f'PCC={best_s1.slice1_learned_pcc:.4f}')
    print(f'Best Slice2 learned PCC: k={int(best_s2.k)}, mnn_k={int(best_s2.mnn_k)}, '
          f'PCC={best_s2.slice2_learned_pcc:.4f}')


if __name__ == '__main__':
    main()
