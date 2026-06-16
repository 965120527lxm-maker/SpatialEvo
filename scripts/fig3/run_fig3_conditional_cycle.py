#!/usr/bin/env python3
"""
Fig. 3 conditional cycle-completion baseline.

Trains two translators:
    F_{B<-A}(X, Y_A) -> Y_B
    F_{A<-B}(X, Y_B) -> Y_A

using only the measured panels on each slice (Y_A^1, Y_B^2) plus H&E.
The held-out panels (Y_B^1, Y_A^2) are never used during training.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import SpatialEx as se


def parse_args():
    parser = argparse.ArgumentParser(description="Fig.3 conditional cycle MLP")
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--rep1", type=str,
                        default="Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad")
    parser.add_argument("--rep2", type=str,
                        default="Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad")
    parser.add_argument("--panelA_size", type=int, default=150)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--lambda_dist", type=float, default=1.0,
                        help="Weight for marginal distribution matching loss.")
    parser.add_argument("--lambda_cycle", type=float, default=1.0,
                        help="Weight for cycle-consistency loss.")
    parser.add_argument("--lambda_he", type=float, default=1.0,
                        help="Weight for H&E-only anchor loss.")
    parser.add_argument("--use_he", action="store_true", default=True,
                        help="Use H&E in the conditional input and anchors.")
    parser.add_argument("--no_use_he", action="store_true",
                        help="Disable H&E; use measured panel only.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "conditional", "fig3_conditional_cycle"))
    return parser.parse_args()


def load_slice(path, data_root):
    full_path = os.path.join(data_root, path)
    print(f"Loading {full_path}")
    adata = sc.read_h5ad(full_path)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata


def split_panels(adata, panelA_genes, panelB_genes):
    return adata[:, panelA_genes].copy(), adata[:, panelB_genes].copy()


def evaluate(adata_gt, pred, label="Slice"):
    graph = se.pp.Build_graph(
        adata_gt.obsm["spatial"], graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo"
    )
    gt_X = adata_gt.X.toarray() if hasattr(adata_gt.X, "toarray") else np.array(adata_gt.X.copy())
    pred_X = np.array(pred.copy())

    pcc, pcc_reduce = se.utils.Compute_metrics(gt_X, pred_X, metric="pcc")
    ssim, ssim_reduce = se.utils.Compute_metrics(gt_X, pred_X, metric="ssim", graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt_X, pred_X, metric="cmd")

    print(f"[{label}] gene-level PCC: {pcc_reduce:.6f}, SSIM: {ssim_reduce:.6f}, CMD: {cmd_reduce:.6f}")
    return {
        "pcc_per_gene": pcc,
        "pcc": float(pcc_reduce),
        "ssim": float(ssim_reduce),
        "cmd": float(cmd_reduce),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load full slices
    rep1_full = load_slice(args.rep1, args.data_root)
    rep2_full = load_slice(args.rep2, args.data_root)
    print(f"Rep1: {rep1_full.shape}, Rep2: {rep2_full.shape}")

    # Split panel A/B
    genes = rep1_full.var_names.values
    np.random.seed(args.seed)
    np.random.shuffle(genes)
    panelA_genes = genes[:args.panelA_size].tolist()
    panelB_genes = genes[args.panelA_size:].tolist()
    print(f"Panel A: {len(panelA_genes)} genes, Panel B: {len(panelB_genes)} genes")

    rep1_A, rep1_B = split_panels(rep1_full, panelA_genes, panelB_genes)
    rep2_A, rep2_B = split_panels(rep2_full, panelA_genes, panelB_genes)

    # Train
    print("\n=== Training conditional_cycle ===")
    trainer = se.SpatialExP_ConditionalCycleMLP(
        rep1_A, rep2_B,
        use_he=(args.use_he and not args.no_use_he),
        hidden_dim=args.hidden_dim,
        lambda_dist=args.lambda_dist,
        lambda_cycle=args.lambda_cycle,
        lambda_he=args.lambda_he,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        device=device,
        seed=args.seed,
    )
    trainer.train()

    # Inference
    print("\n=== Inference ===")
    pred_B1 = trainer.predict_panelB_on_slice1(
        rep1_A.obsm["he"],
        np.asarray(rep1_A.X.toarray() if hasattr(rep1_A.X, "toarray") else rep1_A.X)
    )
    pred_A2 = trainer.predict_panelA_on_slice2(
        rep2_B.obsm["he"],
        np.asarray(rep2_B.X.toarray() if hasattr(rep2_B.X, "toarray") else rep2_B.X)
    )

    pred_B1_df = pd.DataFrame(pred_B1, index=rep1_A.obs_names, columns=panelB_genes)
    pred_A2_df = pd.DataFrame(pred_A2, index=rep2_B.obs_names, columns=panelA_genes)
    pred_B1_df.to_csv(os.path.join(args.out_dir, "pred_panelB1_conditional_cycle.csv"))
    pred_A2_df.to_csv(os.path.join(args.out_dir, "pred_panelA2_conditional_cycle.csv"))

    # Evaluation
    print("\n=== Evaluation ===")
    gt_B1 = rep1_B[rep1_A.obs_names, panelB_genes]
    gt_A2 = rep2_A[rep2_B.obs_names, panelA_genes]

    metrics_B1 = evaluate(gt_B1, pred_B1_df.values, label="Slice1 PanelB prediction (conditional_cycle)")
    metrics_A2 = evaluate(gt_A2, pred_A2_df.values, label="Slice2 PanelA prediction (conditional_cycle)")

    summary = {
        "model": "conditional_cycle",
        "lambda_dist": args.lambda_dist,
        "lambda_cycle": args.lambda_cycle,
        "lambda_he": args.lambda_he,
        "use_he": (args.use_he and not args.no_use_he),
        "hidden_dim": args.hidden_dim,
        "slice1_pcc": metrics_B1["pcc"],
        "slice1_ssim": metrics_B1["ssim"],
        "slice1_cmd": metrics_B1["cmd"],
        "slice2_pcc": metrics_A2["pcc"],
        "slice2_ssim": metrics_A2["ssim"],
        "slice2_cmd": metrics_A2["cmd"],
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, "metrics_conditional_cycle.csv"), index=False)
    print(f"\nOutputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
