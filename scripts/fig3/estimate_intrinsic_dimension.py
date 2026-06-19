#!/usr/bin/env python3
"""Estimate intrinsic dimension with Participation Ratio (PR) and TwoNN."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'fig3'))
from exp_paths import output_dir

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse

try:
    import skdim
except ImportError:
    skdim = None


def parse_args():
    parser = argparse.ArgumentParser(description="Intrinsic dimension: PR + TwoNN")
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--rep1", type=str, default="Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad")
    parser.add_argument("--rep2", type=str, default="Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad")
    parser.add_argument("--panel_csv", type=str, default=os.path.join(PROJECT_ROOT, "data", "panel_split_official.csv"))
    parser.add_argument("--twonn_n", type=int, default=8000, help="Subsample size for TwoNN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=output_dir('intrinsic_dimension'))
    return parser.parse_args()


def load_matrix(adata, genes):
    X = adata[:, genes].X
    return X.toarray() if scipy.sparse.issparse(X) else np.asarray(X, dtype=np.float64)


def zscore_rows(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def participation_ratio(X):
    """PR from feature covariance eigenvalues."""
    X = zscore_rows(X)
    # cell x gene -> covariance over genes (features)
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = np.clip(eigvals, 0, None)
    num = eigvals.sum() ** 2
    den = (eigvals ** 2).sum() + 1e-12
    pr = float(num / den)
    return pr, eigvals


def pca_elbow_k(eigvals, var_threshold=0.95):
    total = eigvals.sum() + 1e-12
    cum = np.cumsum(eigvals) / total
    k95 = int(np.searchsorted(cum, var_threshold) + 1)
    return k95


def twonn_id(X, seed=42):
    if skdim is None:
        raise ImportError("scikit-dimension required for TwoNN")
    X = zscore_rows(X).astype(np.float64)
    twonn = skdim.id.TwoNN()
    return float(twonn.fit_transform(X))


def subsample(X, n, seed):
    rng = np.random.default_rng(seed)
    n = min(n, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], idx


def analyze_block(name, X, args):
    pr, eigvals = participation_ratio(X)
    k95 = pca_elbow_k(eigvals, 0.95)
    X_sub, _ = subsample(X, args.twonn_n, args.seed)
    twonn = twonn_id(X_sub, args.seed)
    return {
        "object": name,
        "n_cells": X.shape[0],
        "n_features": X.shape[1],
        "participation_ratio": pr,
        "pca_k95": k95,
        "twonn_subsample_n": X_sub.shape[0],
        "twonn_id": twonn,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    panel_df = pd.read_csv(args.panel_csv)
    panelA = panel_df[panel_df.panel == "panelA"].gene.astype(str).tolist()
    panelB = panel_df[panel_df.panel == "panelB"].gene.astype(str).tolist()

    rep1 = sc.read_h5ad(os.path.join(args.data_root, args.rep1))
    rep2 = sc.read_h5ad(os.path.join(args.data_root, args.rep2))
    rep1.var_names = rep1.var_names.astype(str)
    rep2.var_names = rep2.var_names.astype(str)

    YA1 = load_matrix(rep1, panelA)
    YB1 = load_matrix(rep1, panelB)
    YA2 = load_matrix(rep2, panelA)
    YB2 = load_matrix(rep2, panelB)
    HE1 = np.asarray(rep1.obsm["he"], dtype=np.float64)
    HE2 = np.asarray(rep2.obsm["he"], dtype=np.float64)

    rows = []
    blocks = [
        ("Rep1 panelA", YA1),
        ("Rep1 panelB", YB1),
        ("Rep2 panelA", YA2),
        ("Rep2 panelB", YB2),
        ("Rep1 full panel", load_matrix(rep1, panelA + panelB)),
        ("Rep2 full panel", load_matrix(rep2, panelA + panelB)),
        ("Rep1 H&E", HE1),
        ("Rep2 H&E", HE2),
        ("panelA both slices", np.vstack([YA1, YA2])),
        ("panelB both slices", np.vstack([YB1, YB2])),
        ("full panel both slices", np.vstack([
            load_matrix(rep1, panelA + panelB),
            load_matrix(rep2, panelA + panelB),
        ])),
    ]

    print("=== Intrinsic dimension estimates (PR + TwoNN) ===")
    for name, X in blocks:
        row = analyze_block(name, X, args)
        rows.append(row)
        print(f"{name:28s}  features={row['n_features']:4d}  "
              f"PR={row['participation_ratio']:6.1f}  PCA95={row['pca_k95']:3d}  "
              f"TwoNN={row['twonn_id']:6.2f}  (n={row['twonn_subsample_n']})")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "intrinsic_dimension_pr_twonN.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
