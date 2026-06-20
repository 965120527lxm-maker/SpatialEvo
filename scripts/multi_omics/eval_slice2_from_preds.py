#!/usr/bin/env python3
"""Evaluate saved SpatialEx+ slice2 protein predictions without retraining."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import pandas as pd

from run_multi_omics import (
    load_protein_slice,
    load_groundtruth_protein_rep2,
    evaluate_protein_prediction,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--pred_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--protein_features", default="cell_CD20_mean,cell_HER2_mean")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    protein_cols = [c.strip() for c in args.protein_features.split(",")]

    adata_protein = load_protein_slice(args.data_root, protein_cols)
    panelA2 = pd.read_csv(args.pred_csv, index_col=0)
    panelA2.index = panelA2.index.astype(str)
    adata2_gt = load_groundtruth_protein_rep2(
        args.data_root, adata_protein, protein_cols, panelA2.index)

    metrics = evaluate_protein_prediction(
        adata2_gt, panelA2.values, label="Slice 2 RNA->protein (SpatialEx+)")
    row = {k: v for k, v in metrics.items() if k != "pcc_per_feature"}
    pd.DataFrame([row]).to_csv(os.path.join(args.out_dir, "metrics_slice2.csv"), index=False)
    print(f"Saved {args.out_dir}/metrics_slice2.csv")


if __name__ == "__main__":
    main()
