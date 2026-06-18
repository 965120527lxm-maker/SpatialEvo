#!/usr/bin/env python3
"""
Run SpatialEx+ omics-diagonal integration for breast cancer transcriptomics ↔ proteomics.

This script is adapted to the NicheTrans Zenodo release for 10x breast cancer, which
ships a single preprocessed h5ad file containing:
  - 167,780 cells x 313 genes (transcriptomics, in .X)
  - matched protein intensities for CD20 and HER2 (in .obs)

The existing SpatialEx repo already provides Rep1/Rep2 Xenium h5ad files with UNI
embeddings and spatial coordinates, so we combine:
  - protein expression from the NicheTrans h5ad
  - H&E embeddings and spatial coordinates from the repo h5ad files
  - Rep2 transcriptomics from the repo h5ad file

Expected data layout (under --data_root):
    Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad
    Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad
    10x_breast_cancer/human_breast_cancer.h5ad

Usage:
    conda activate spatialex
    python scripts/multi_omics/run_multi_omics.py --data_root ./data --epochs 500
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
    parser = argparse.ArgumentParser(description="SpatialEx+ multi-omics (transcriptomics-proteomics)")
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"),
                        help="Root folder containing the h5ad files.")
    parser.add_argument("--num_neighbors", type=int, default=7,
                        help="K for spatial KNN hypergraph.")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension of the SpatialEx+ backbone.")
    parser.add_argument("--translator_hidden_dim", type=int, default=512,
                        help="Hidden dimension of the cross-omics translators.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--device", type=str, default=None,
                        help="PyTorch device; defaults to cuda:0 if available.")
    parser.add_argument("--protein_features", type=str, default="cell_CD20_mean,cell_HER2_mean",
                        help="Comma-separated obs columns to use as protein panel.")
    parser.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "multi_omics", "run_multi_omics"),
                        help="Directory to save predictions and metrics.")
    parser.add_argument("--model", type=str, default="spatialexp",
                        choices=["spatialexp", "spatialexp_small", "spatialexp_gt"],
                        help="Which SpatialEx+ variant to train.")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads (only used for spatialexp_gt).")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (only used for spatialexp_gt).")
    return parser.parse_args()


def load_protein_slice(data_root, protein_cols):
    """Build the protein-panel AnnData for Rep1.

    Protein expression comes from the NicheTrans h5ad; H&E embeddings and spatial
    coordinates come from the existing Rep1 h5ad shipped with the repo.
    """
    niche_path = os.path.join(data_root, "10x_breast_cancer", "human_breast_cancer.h5ad")
    he_path = os.path.join(data_root, "Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad")

    print(f"[Protein] Loading protein data from {niche_path}")
    niche = sc.read_h5ad(niche_path)
    niche.obs_names = niche.obs_names.astype(str)

    # Validate requested protein columns
    missing = [c for c in protein_cols if c not in niche.obs.columns]
    if missing:
        raise ValueError(f"Protein columns not found in obs: {missing}. Available: {niche.obs.columns.tolist()}")

    print(f"[Protein] Loading H&E/spatial from {he_path}")
    he_adata = sc.read_h5ad(he_path)
    he_adata.obs_names = he_adata.obs_names.astype(str)

    # Keep only cells present in both datasets
    common_cells = list(set(niche.obs_names).intersection(set(he_adata.obs_names)))
    print(f"[Protein] Common cells between NicheTrans and Rep1 h5ad: {len(common_cells)}")
    if len(common_cells) == 0:
        raise ValueError("No overlapping cell IDs between NicheTrans protein data and Rep1 H&E data.")

    niche = niche[common_cells].copy()
    he_adata = he_adata[common_cells].copy()

    # Build protein AnnData
    protein_X = niche.obs[protein_cols].values.astype(np.float32)
    adata = sc.AnnData(X=protein_X, obs=niche.obs.copy())
    adata.var_names = pd.Index(protein_cols)
    adata.obsm["spatial"] = he_adata.obsm["spatial"].copy()
    adata.obsm["he"] = he_adata.obsm["he"].copy()

    # Generate_pseudo_spot expects spatial coords in obs columns
    adata.obs["x_centroid"] = he_adata.obs["x_centroid"].values
    adata.obs["y_centroid"] = he_adata.obs["y_centroid"].values

    # Standardize protein intensities per feature (matches Tutorial 4 preprocessing)
    sc.pp.scale(adata)
    return adata


def load_rna_slice(data_root):
    """Load the transcriptomics-panel AnnData for Rep2 (existing repo h5ad)."""
    rna_path = os.path.join(data_root, "Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad")
    print(f"[RNA] Loading Rep2 transcriptomics from {rna_path}")
    adata = sc.read_h5ad(rna_path)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    print(f"[RNA] Slice shape: {adata.shape}")
    return adata


def load_groundtruth_rna_rep1(data_root, protein_adata):
    """Load Rep1 ground-truth RNA from the NicheTrans h5ad for evaluation."""
    niche_path = os.path.join(data_root, "10x_breast_cancer", "human_breast_cancer.h5ad")
    niche = sc.read_h5ad(niche_path)
    niche.obs_names = niche.obs_names.astype(str)

    common_cells = list(set(niche.obs_names).intersection(set(protein_adata.obs_names)))
    gt = niche[common_cells].copy()
    # Spatial coordinates are required by graph-aware metrics; copy from protein_adata
    gt.obsm["spatial"] = protein_adata[gt.obs_names].obsm["spatial"].copy()
    return gt


def evaluate_rna_prediction(adata_gt, pred, label="Slice 1"):
    """Evaluate predicted transcriptomics against ground truth using PCC, SSIM, CMD."""
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

    protein_cols = [c.strip() for c in args.protein_features.split(",")]

    # 1. Load slices
    print("\n=== Loading protein slice (Rep1) ===")
    adata_protein = load_protein_slice(args.data_root, protein_cols)
    print(f"Protein slice: {adata_protein.n_obs} cells x {adata_protein.n_vars} proteins")

    print("\n=== Loading transcriptomics slice (Rep2) ===")
    adata_rna = load_rna_slice(args.data_root)
    print(f"RNA slice: {adata_rna.n_obs} cells x {adata_rna.n_vars} genes")

    # 2. Build hypergraphs
    print("\n=== Building hypergraphs ===")
    graph1 = se.pp.Build_hypergraph_spatial_and_HE(
        adata_protein, args.num_neighbors, graph_kind="spatial", return_type="csr"
    )
    graph2 = se.pp.Build_hypergraph_spatial_and_HE(
        adata_rna, args.num_neighbors, graph_kind="spatial", return_type="csr"
    )

    # 3. Train SpatialEx+
    print(f"\n=== Training {args.model} ===")
    if args.model == "spatialexp":
        spatialexp = se.SpatialExP(
            adata_protein, adata_rna, graph1, graph2,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            translator_hidden_dim=args.translator_hidden_dim,
        )
    elif args.model == "spatialexp_small":
        spatialexp = se.SpatialExP_Small(
            adata_protein, adata_rna, graph1, graph2,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            translator_hidden_dim=args.translator_hidden_dim,
        )
    elif args.model == "spatialexp_gt":
        spatialexp = se.SpatialExP_GT(
            adata_protein, adata_rna, graph1, graph2,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            translator_hidden_dim=args.translator_hidden_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    spatialexp.train()

    # 4. In-area inference
    print("\n=== In-area inference ===")
    panelB1_arr = spatialexp.inference_direct(adata_protein.obsm["he"], graph1, panel="panelB")
    panelA2_arr = spatialexp.inference_indirect(adata_rna.obsm["he"], graph2, panel="panelA")

    panelB1 = pd.DataFrame(
        panelB1_arr,
        index=adata_protein.obs_names,
        columns=adata_rna.var_names
    )
    panelA2 = pd.DataFrame(
        panelA2_arr,
        index=adata_rna.obs_names,
        columns=adata_protein.var_names
    )

    panelB1.to_csv(os.path.join(args.out_dir, "panelB1_predicted_transcriptomics_on_protein_slice.csv"))
    panelA2.to_csv(os.path.join(args.out_dir, "panelA2_predicted_proteins_on_rna_slice.csv"))

    # 5. Evaluation: predicted transcriptomics on protein slice vs ground truth
    print("\n=== Evaluation ===")
    adata1_gt = load_groundtruth_rna_rep1(args.data_root, adata_protein)
    # Align ground truth to predicted cell order and gene order
    adata1_gt = adata1_gt[panelB1.index].copy()
    adata1_gt = adata1_gt[:, panelB1.columns].copy()

    metrics = evaluate_rna_prediction(adata1_gt, panelB1.values, label="Slice 1 protein->RNA")
    pd.DataFrame([metrics]).to_csv(os.path.join(args.out_dir, "metrics_slice1.csv"), index=False)

    print(f"\nOutputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
