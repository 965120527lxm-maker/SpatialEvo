#!/usr/bin/env python3
"""
Smoke test for SpatialExP_Small.

Verifies that the capacity-matched HGNN baseline can be constructed with
hidden_dim=128 / translator_hidden_dim=128 / use_dgi=False, and that it
successfully trains for a few epochs on synthetic data.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import scanpy as sc
import torch
from sklearn.neighbors import kneighbors_graph

from SpatialEx import SpatialExP, SpatialExP_Small, SpatialExP_GT


def make_synthetic_data(n_cells=200, in_dim=64, out_dim1=20, out_dim2=15):
    he = np.random.randn(n_cells, in_dim).astype(np.float32)
    x1 = np.abs(np.random.randn(n_cells, out_dim1).astype(np.float32))
    x2 = np.abs(np.random.randn(n_cells, out_dim2).astype(np.float32))

    adata1 = sc.AnnData(X=x1)
    adata1.obsm["he"] = he
    adata1.obs["x_centroid"] = np.random.rand(n_cells) * 100
    adata1.obs["y_centroid"] = np.random.rand(n_cells) * 100

    adata2 = sc.AnnData(X=x2)
    adata2.obsm["he"] = he
    adata2.obs["x_centroid"] = np.random.rand(n_cells) * 100
    adata2.obs["y_centroid"] = np.random.rand(n_cells) * 100

    graph = kneighbors_graph(he, n_neighbors=5, mode="connectivity")
    return adata1, adata2, graph


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cpu")
    adata1, adata2, graph = make_synthetic_data()

    # 1. SpatialExP_Small uses the intended reduced defaults.
    small = SpatialExP_Small(
        adata1, adata2, graph, graph,
        epochs=2, prune=50, device=device
    )
    assert small.hidden_dim == 128
    assert small.translator_hidden_dim == 128
    assert small.use_dgi is False
    assert small.module_HA.use_dgi is False
    assert small.module_HB.use_dgi is False
    print(f"SpatialExP_Small backbone params: {count_params(small.module_HA):,}")

    # 2. Explicit SpatialExP small config matches SpatialExP_Small.
    explicit = SpatialExP(
        adata1, adata2, graph, graph,
        hidden_dim=128, translator_hidden_dim=128, use_dgi=False,
        epochs=2, prune=50, device=device
    )
    assert explicit.hidden_dim == small.hidden_dim
    assert explicit.translator_hidden_dim == small.translator_hidden_dim
    assert explicit.use_dgi == small.use_dgi
    assert count_params(explicit.module_HA) == count_params(small.module_HA)

    # 3. Default SpatialExP remains unchanged (backward compatibility).
    default = SpatialExP(
        adata1, adata2, graph, graph,
        epochs=2, prune=50, device=device
    )
    assert default.hidden_dim == 512
    assert default.translator_hidden_dim == 512
    assert default.use_dgi is True

    # 4. Training loop runs.
    small.train()
    print("SpatialExP_Small training completed for 2 epochs.")

    # 5. Inference runs.
    pred_b = small.inference_direct(adata1.obsm["he"], graph, panel="panelB")
    pred_a = small.inference_indirect(adata2.obsm["he"], graph, panel="panelA")
    assert pred_b.shape == (adata1.n_obs, adata2.n_vars)
    assert pred_a.shape == (adata2.n_obs, adata1.n_vars)

    print("\nSpatialExP_Small smoke test passed!")


if __name__ == "__main__":
    main()
