#!/usr/bin/env python3
"""
Strict Fig.3 cross-slice alignment benchmark (OT focus).

Two-hop pseudo-label protocol:
  Step1: H&E cross-slice matching -> pseudo YB1
  Step2: B-panel cross-slice matching -> pseudo YA2
  Downstream: Panel MLP trained on pseudo labels

Step1 methods (H&E space): raw_knn, mnn_he, pca50_mnn, coral_mnn, landmark_ot, localk_ot
Step2 methods (B-panel space): mnn_bpanel, pca50_mnn_bpanel, landmark_ot, localk_ot, raw_knn
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
import numpy as np
import pandas as pd
import torch

import SpatialEx as se
import run_fig3_mnn_pseudo as mnn
import run_fig3_latent_mnn as latent
import alignment_ot as ot


STEP1_METHODS = ['raw_knn', 'mnn_he', 'pca50_mnn', 'coral_mnn', 'landmark_ot', 'localk_ot']
STEP2_METHODS = ['mnn_bpanel', 'pca50_mnn_bpanel', 'landmark_ot', 'localk_ot', 'raw_knn']
# Back-compat: single-method runs use step2=mnn_bpanel
ALL_METHODS = STEP1_METHODS

DEFAULT_PIPELINES = [
    'mnn_he+mnn_bpanel',           # current SOTA baseline
    'mnn_he+landmark_ot',          # H&E MNN + B-panel OT
    'landmark_ot+landmark_ot',     # full OT pipeline
    'pca50_mnn+pca50_mnn_bpanel',  # PCA MNN both steps
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'data'))
    parser.add_argument('--rep1', type=str, default='Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad')
    parser.add_argument('--rep2', type=str, default='Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad')
    parser.add_argument('--panel_csv', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'panel_split_official.csv'))
    parser.add_argument('--panelA_size', type=int, default=150)
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        choices=STEP1_METHODS,
                        help='Legacy: Step1-only sweep with Step2=mnn_bpanel')
    parser.add_argument('--pipelines', type=str, nargs='+', default=None,
                        help='Step1+Step2 pairs, e.g. mnn_he+landmark_ot')
    parser.add_argument('--step1', type=str, default=None, choices=STEP1_METHODS)
    parser.add_argument('--step2', type=str, default='mnn_bpanel', choices=STEP2_METHODS)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--mnn_k', type=int, default=20)
    parser.add_argument('--pca_dim', type=int, default=50)
    parser.add_argument('--landmarks', type=int, default=2048)
    parser.add_argument('--ot_reg', type=float, default=0.05)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default=output_dir('alignment_benchmark_official'))
    return parser.parse_args()


def compute_mnn_match_rate(query, ref, mnn_k=20, batch_size=4096, device='cuda'):
    fwd = mnn.batched_topk_indices(query, ref, k=mnn_k, batch_size=batch_size, device=device)
    rev = mnn.batched_topk_indices(ref, query, k=mnn_k, batch_size=batch_size, device=device)
    rev_sets = [set(rev[j].tolist()) for j in range(ref.shape[0])]
    matched = 0
    for i in range(query.shape[0]):
        if any(i in rev_sets[j] for j in fwd[i].tolist()):
            matched += 1
    return matched / max(query.shape[0], 1)


def _match_features_step1(method, HE1, HE2, args):
    if method == 'pca50_mnn':
        return latent.fit_pca(HE1, HE2, n_components=args.pca_dim)
    if method == 'coral_mnn':
        return HE1, latent.coral_alignment(HE2, HE1)
    return HE1, HE2


def _match_features_step2(method, Y_B2, pseudo_B1, args):
    if method == 'pca50_mnn_bpanel':
        return ot.fit_pca(Y_B2, pseudo_B1, n_components=args.pca_dim, seed=args.seed)
    return Y_B2, pseudo_B1


def build_pseudo_yb1(method, HE1, HE2, Y_B2, args, device):
    q, r = _match_features_step1(method, HE1, HE2, args)
    if method == 'raw_knn':
        return mnn.build_raw_pseudo(q, r, Y_B2, k=args.k, device=device)
    if method in ('mnn_he', 'pca50_mnn', 'coral_mnn'):
        return mnn.build_mnn_pseudo(q, r, Y_B2, k=args.k, mnn_k=args.mnn_k, device=device)
    if method == 'landmark_ot':
        return ot.build_landmark_ot_pseudo(
            HE1, HE2, Y_B2, k=args.k, pca_dim=args.pca_dim,
            n_landmarks=args.landmarks, reg=args.ot_reg, seed=args.seed, device=device)
    if method == 'localk_ot':
        return ot.build_localk_ot_pseudo(
            HE1, HE2, Y_B2, k=args.mnn_k, pca_dim=args.pca_dim,
            reg=args.ot_reg, seed=args.seed, device=device)
    raise ValueError(f'Unknown step1 method: {method}')


def build_pseudo_ya2(method, Y_B2, pseudo_B1, Y_A1, args, device):
    q, r = _match_features_step2(method, Y_B2, pseudo_B1, args)
    if method == 'raw_knn':
        return mnn.build_raw_pseudo(q, r, Y_A1, k=args.k, device=device)
    if method in ('mnn_bpanel', 'pca50_mnn_bpanel'):
        return mnn.build_mnn_pseudo(q, r, Y_A1, k=args.k, mnn_k=args.mnn_k, device=device)
    if method == 'landmark_ot':
        return ot.build_landmark_ot_pseudo(
            Y_B2, pseudo_B1, Y_A1, k=args.k, pca_dim=args.pca_dim,
            n_landmarks=args.landmarks, reg=args.ot_reg, seed=args.seed, device=device)
    if method == 'localk_ot':
        return ot.build_localk_ot_pseudo(
            Y_B2, pseudo_B1, Y_A1, k=args.mnn_k, pca_dim=args.pca_dim,
            reg=args.ot_reg, seed=args.seed, device=device)
    raise ValueError(f'Unknown step2 method: {method}')


def alignment_diagnostics(method, HE1, HE2, args, device):
    if method in ('mnn_he', 'pca50_mnn', 'coral_mnn'):
        if method == 'pca50_mnn':
            q, r = latent.fit_pca(HE1, HE2, n_components=args.pca_dim)
        elif method == 'coral_mnn':
            q, r = HE1, latent.coral_alignment(HE2, HE1)
        else:
            q, r = HE1, HE2
        return {'mnn_match_rate': compute_mnn_match_rate(q, r, args.mnn_k, device=device)}
    if method in ('landmark_ot', 'localk_ot'):
        return {'mean_nn_dist_pca': ot.mean_nn_dist_pca(
            HE1, HE2, k=1, pca_dim=args.pca_dim, seed=args.seed, device=device)}
    return {}


def predict(model, x, y_mean, y_std, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(mnn.zscore(x), dtype=torch.float32, device=device)).cpu().numpy()
    return pred * y_std + y_mean


def parse_pipelines(args):
    if args.pipelines:
        out = []
        for p in args.pipelines:
            if '+' not in p:
                raise ValueError(f'Pipeline must be step1+step2, got: {p}')
            s1, s2 = p.split('+', 1)
            out.append((s1.strip(), s2.strip()))
        return out
    if args.step1:
        return [(args.step1, args.step2)]
    if args.methods:
        return [(m, 'mnn_bpanel') for m in args.methods]
    return [tuple(x.split('+', 1)) for x in DEFAULT_PIPELINES]


def run_pipeline(step1, step2, HE1, HE2, Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device):
    label = f'{step1}+{step2}'
    print(f'\n{"=" * 60}\nPipeline: {label}\n{"=" * 60}')
    pseudo_B1 = build_pseudo_yb1(step1, HE1, HE2, Y_B2, args, device)
    pseudo_A2 = build_pseudo_ya2(step2, Y_B2, pseudo_B1, Y_A1, args, device)

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

    diag = alignment_diagnostics(step1, HE1, HE2, args, device)
    rows = []
    for what, m1, m2 in [('direct', direct1, direct2), ('learned', learned1, learned2)]:
        row = {
            'pipeline': label,
            'step1': step1,
            'step2': step2,
            'what': what,
            'slice1_pcc': m1[0], 'slice1_ssim': m1[1], 'slice1_cmd': m1[2],
            'slice2_pcc': m2[0], 'slice2_ssim': m2[1], 'slice2_cmd': m2[2],
        }
        row.update(diag)
        rows.append(row)
    return rows


def append_pipeline_rows(rows, out_dir):
    """Append one pipeline result immediately (crash-safe)."""
    out_csv = os.path.join(out_dir, 'alignment_results.csv')
    chunk = pd.DataFrame(rows)
    if os.path.exists(out_csv):
        pd.concat([pd.read_csv(out_csv), chunk], ignore_index=True).to_csv(out_csv, index=False)
    else:
        chunk.to_csv(out_csv, index=False)
    print(f'Appended {rows[0]["pipeline"]} -> {out_csv}')


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

    pipelines = parse_pipelines(args)
    print('Pipelines:', pipelines)

    all_rows = []
    for step1, step2 in pipelines:
        if step1 not in STEP1_METHODS:
            raise ValueError(f'Unknown step1: {step1}')
        if step2 not in STEP2_METHODS:
            raise ValueError(f'Unknown step2: {step2}')
        rows = run_pipeline(
            step1, step2, HE1, HE2, Y_A1, Y_B1, Y_A2, Y_B2, graph1, graph2, args, device)
        all_rows.extend(rows)
        append_pipeline_rows(rows, args.out_dir)

    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    print('\n=== Final alignment benchmark results ===')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
