#!/usr/bin/env python3
"""
Smoke test for the translator_hidden_dim fix.

Uses the existing Xenium breast-cancer .h5ad files but splits the 313-gene panel
into two disjoint subsets.  This exercises SpatialExP when adata1.n_vars != adata2.n_vars
and verifies that the translator hidden size is no longer forced to equal the
target panel size.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import scanpy as sc
import torch

import SpatialEx as se


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = os.path.join(PROJECT_ROOT, "data")
    adata_full = sc.read_h5ad(os.path.join(data_root, "Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad"))
    print(f"Loaded full data: {adata_full.n_obs} cells x {adata_full.n_vars} genes")

    # Split into two non-overlapping panels
    genes = adata_full.var_names.values
    np.random.seed(0)
    np.random.shuffle(genes)
    panelA_genes = genes[:150]
    panelB_genes = genes[150:]

    adata_A = adata_full[:, panelA_genes].copy()
    adata_B = adata_full[:, panelB_genes].copy()
    print(f"Panel A: {adata_A.n_vars} genes, Panel B: {adata_B.n_vars} genes")

    # Use the same graph for both (smoke test only)
    graph = se.pp.Build_hypergraph_spatial_and_HE(
        adata_A, num_neighbors=7, graph_kind="spatial", return_type="csr"
    )

    # Train with a custom translator hidden dim
    translator_hidden_dim = 128
    spatialexp = se.SpatialExP(
        adata_A, adata_B, graph, graph,
        device=device,
        epochs=2,
        hidden_dim=128,
        translator_hidden_dim=translator_hidden_dim,
        prune=5000,
    )

    # Verify translator hidden dims
    assert spatialexp.rm_AB.mlp[0].out_features == translator_hidden_dim, \
        f"rm_AB hidden dim should be {translator_hidden_dim}, got {spatialexp.rm_AB.mlp[0].out_features}"
    assert spatialexp.rm_BA.mlp[0].out_features == translator_hidden_dim, \
        f"rm_BA hidden dim should be {translator_hidden_dim}, got {spatialexp.rm_BA.mlp[0].out_features}"
    print(f"Translator hidden dims verified: {translator_hidden_dim}")

    # Run a few training steps
    print("Running 2-epoch smoke test training...")
    spatialexp.train()

    # Inference
    print("Running inference...")
    pred_B = spatialexp.inference_direct(adata_A.obsm["he"], graph, panel="panelB")
    pred_A = spatialexp.inference_indirect(adata_B.obsm["he"], graph, panel="panelA")

    print(f"Predicted panel B shape: {pred_B.shape}")
    print(f"Predicted panel A shape: {pred_A.shape}")
    assert pred_B.shape[1] == adata_B.n_vars
    assert pred_A.shape[1] == adata_A.n_vars

    print("\nSmoke test passed!")


if __name__ == "__main__":
    main()
