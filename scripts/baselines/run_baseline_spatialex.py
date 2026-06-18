#!/usr/bin/env python3
"""
Baseline: train plain SpatialEx on Rep2 RNA only, then predict Rep1 RNA.

This answers the question: "What if we just used SpatialEx instead of SpatialExP?"
Since SpatialEx expects two slices with the same panel, we pass Rep2 RNA as both
slices (training two identical networks on the same data), then use module_HB to
predict RNA on Rep1.
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
    parser = argparse.ArgumentParser(description="SpatialEx baseline: Rep2 RNA -> Rep1 RNA")
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"),
                        help="Root folder containing the h5ad files.")
    parser.add_argument("--num_neighbors", type=int, default=7,
                        help="K for spatial KNN hypergraph.")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension of the SpatialEx backbone.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--device", type=str, default=None,
                        help="PyTorch device; defaults to cuda:0 if available.")
    parser.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "baseline_spatialex"),
                        help="Directory to save predictions and metrics.")
    return parser.parse_args()


def load_rna_slice(data_root, name):
    path = os.path.join(data_root, f"{name}_uni_resolution64_full.h5ad")
    print(f"[RNA] Loading {path}")
    adata = sc.read_h5ad(path)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    print(f"[RNA] Shape: {adata.shape}")
    return adata


def evaluate_rna_prediction(adata_gt, pred, label="Slice 1"):
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

    # Load Rep1 RNA (for evaluation ground truth) and Rep2 RNA (for training)
    adata_rep1 = load_rna_slice(args.data_root, "Human_Breast_Cancer_Rep1")
    adata_rep2 = load_rna_slice(args.data_root, "Human_Breast_Cancer_Rep2")

    # Build graphs
    graph_rep1 = se.pp.Build_hypergraph_spatial_and_HE(
        adata_rep1, args.num_neighbors, graph_kind="spatial", return_type="csr"
    )
    graph_rep2 = se.pp.Build_hypergraph_spatial_and_HE(
        adata_rep2, args.num_neighbors, graph_kind="spatial", return_type="csr"
    )

    # Train SpatialEx with Rep2 as both slices (equivalent to training on Rep2 only)
    print("\n=== Training SpatialEx on Rep2 RNA ===")
    spatialex = se.SpatialEx(
        adata_rep2, adata_rep2, graph_rep2, graph_rep2,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
    )
    spatialex.train()

    # Predict Rep1 RNA using module_HB (trained on Rep2 RNA)
    print("\n=== Predicting Rep1 RNA ===")
    pred = spatialex.inference(adata_rep1.obsm["he"], graph_rep1, panel="panelB")

    pred_df = pd.DataFrame(pred, index=adata_rep1.obs_names, columns=adata_rep1.var_names)
    pred_df.to_csv(os.path.join(args.out_dir, "predicted_rna_on_rep1.csv"))

    # Evaluate
    print("\n=== Evaluation ===")
    # Use NicheTrans ground truth if available; otherwise use repo Rep1 RNA (slight overfit if used in train, but here we didn't train on it)
    niche_path = os.path.join(args.data_root, "10x_breast_cancer", "human_breast_cancer.h5ad")
    if os.path.exists(niche_path):
        niche = sc.read_h5ad(niche_path)
        niche.obs_names = niche.obs_names.astype(str)
        common_cells = list(set(niche.obs_names).intersection(set(adata_rep1.obs_names)))
        gt = niche[common_cells].copy()
        gt = gt[:, adata_rep1.var_names].copy()
        gt.obsm["spatial"] = adata_rep1[gt.obs_names].obsm["spatial"].copy()
        pred_aligned = pred_df.loc[gt.obs_names].values
    else:
        gt = adata_rep1
        pred_aligned = pred_df.values

    metrics = evaluate_rna_prediction(gt, pred_aligned, label="Rep2->Rep1 RNA")
    pd.DataFrame([metrics]).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)

    print(f"\nOutputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
