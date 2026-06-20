#!/usr/bin/env python3
"""
Fig.4: SpatialEx+ scalability on million-cell big slices (Tutorial 3 protocol).

Train SpatialExP_Big on adjacent Human_Breast_IDC_Big1/2 with 280-gene
panel split (Big_by_name.csv): slice1 panel A, slice2 panel B.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "fig4"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "fig4"))
from exp_paths import output_dir

import argparse
import json

import numpy as np
import pandas as pd
import torch

import SpatialEx as se
from load_big_data import load_big_slice, load_panel_genes, subset_panel, default_gt_h5ad
from graph_cache import get_or_build_graphs, graph_cache_meta


def parse_args():
    p = argparse.ArgumentParser(description="Fig.4 SpatialExP_Big panel diagonal")
    p.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--panel_csv", type=str,
                   default=os.path.join(PROJECT_ROOT, "data", "panel_selection", "Big_by_name.csv"))
    p.add_argument("--num_neighbors", type=int, default=7)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_num", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_dir", type=str, default=output_dir("spatialexp_big"))
    p.add_argument("--gt_big1", type=str, default=None,
                   help="Optional full-slice h5ad for slice1 panel-B ground truth eval")
    p.add_argument("--gt_big2", type=str, default=None,
                   help="Optional full-slice h5ad for slice2 panel-A ground truth eval")
    p.add_argument("--graph_cache_dir", type=str, default=None,
                   help="Directory to save/load spatial hypergraphs (default: <out_dir>/graphs)")
    p.add_argument("--rebuild_graphs", action="store_true",
                   help="Force rebuild hypergraphs even if cache exists")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; evaluate saved predictions in out_dir")
    return p.parse_args()


def eval_panel(gt, pred, graph, label):
    pcc, pcc_reduce = se.utils.Compute_metrics(gt, pred, metric="pcc")
    ssim, ssim_reduce = se.utils.Compute_metrics(gt, pred, metric="ssim", graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt, pred, metric="cmd")
    print(f"[{label}] PCC={pcc_reduce:.4f} SSIM={ssim_reduce:.4f} CMD={cmd_reduce:.4f}")
    return {
        "label": label,
        "pcc_mean": float(pcc_reduce),
        "ssim_mean": float(ssim_reduce),
        "cmd_mean": float(cmd_reduce),
    }


def maybe_eval_gt(gt_path, obs_names, genes, pred, spatial, num_neighbors, label):
    if gt_path is None or not os.path.isfile(gt_path):
        return None
    import scanpy as sc
    gt_adata = sc.read_h5ad(gt_path)
    gt_adata = gt_adata[obs_names, genes].copy()
    gt_x = gt_adata.X.toarray() if hasattr(gt_adata.X, "toarray") else np.asarray(gt_adata.X)
    graph = se.pp.Build_graph(
        spatial, graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="csr", num_neighbors=num_neighbors)
    return eval_panel(gt_x, pred, graph, label)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    panel_a, panel_b = load_panel_genes(args.panel_csv)
    log(f"Panel split: A={len(panel_a)} genes, B={len(panel_b)} genes")

    log("Loading big slices ...")
    adata1_full = load_big_slice(args.data_root, 1)
    adata2_full = load_big_slice(args.data_root, 2)
    log(f"Loaded big1: {adata1_full.n_obs} cells x {adata1_full.n_vars} vars")
    log(f"Loaded big2: {adata2_full.n_obs} cells x {adata2_full.n_vars} vars")

    adata1 = subset_panel(adata1_full, panel_a, "big1 panelA")
    adata2 = subset_panel(adata2_full, panel_b, "big2 panelB")

    if args.eval_only:
        pred_b1 = np.load(os.path.join(args.out_dir, "pred_panelB_slice1.npy"))
        pred_a2 = np.load(os.path.join(args.out_dir, "pred_panelA_slice2.npy"))
        rows = []
        gt1 = args.gt_big1 or default_gt_h5ad(args.data_root, 1)
        gt2 = args.gt_big2 or default_gt_h5ad(args.data_root, 2)
        m1 = maybe_eval_gt(
            gt1, adata1.obs_names, panel_b, pred_b1,
            adata1.obsm["spatial"], args.num_neighbors, "Slice1 panelB")
        if m1:
            rows.append(m1)
        m2 = maybe_eval_gt(
            gt2, adata2.obs_names, panel_a, pred_a2,
            adata2.obsm["spatial"], args.num_neighbors, "Slice2 panelA")
        if m2:
            rows.append(m2)
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metrics_spatialexp_big.csv"), index=False)
        else:
            print("No ground-truth h5ad provided; nothing to evaluate.")
        log(f"\nEval-only done. Saved to {args.out_dir}")
        return

    meta = {
        "big1_cells": int(adata1.n_obs),
        "big2_cells": int(adata2.n_obs),
        "panel_a_genes": len(panel_a),
        "panel_b_genes": len(panel_b),
        "he_dim1": int(adata1.obsm["he"].shape[1]),
        "he_dim2": int(adata2.obsm["he"].shape[1]),
        "epochs": args.epochs,
        "batch_num": args.batch_num,
        "hidden_dim": args.hidden_dim,
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    cache_dir = args.graph_cache_dir or os.path.join(args.out_dir, "graphs")
    cache_meta = graph_cache_meta(adata1.n_obs, adata2.n_obs, args.num_neighbors)

    def _build():
        log("Building spatial hypergraphs (BallTree KNN; may take 10-30 min per slice) ...")
        g1 = se.pp.Build_hypergraph_spatial_and_HE(
            adata1, args.num_neighbors, graph_kind="spatial", return_type="csr")
        log("Slice1 graph done; building slice2 ...")
        g2 = se.pp.Build_hypergraph_spatial_and_HE(
            adata2, args.num_neighbors, graph_kind="spatial", return_type="csr")
        log("Slice2 graph done.")
        return g1, g2

    graph1, graph2 = get_or_build_graphs(
        _build, cache_dir, cache_meta, rebuild=args.rebuild_graphs, return_type="csr")

    log("Initializing SpatialExP_Big (pseudo-spot aggregation; may take several more minutes) ...")
    log("Training SpatialExP_Big ...")
    model = se.SpatialExP_Big(
        adata1, adata2, graph1, graph2,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_num=args.batch_num,
        lr=args.lr,
        seed=args.seed,
        device=device,
        num_neighbors=args.num_neighbors,
        save_path=os.path.join(args.out_dir, "checkpoints") + os.sep,
    )
    model.train()
    pred_b1, pred_a2 = model.auto_inference()

    np.save(os.path.join(args.out_dir, "pred_panelB_slice1.npy"), pred_b1)
    np.save(os.path.join(args.out_dir, "pred_panelA_slice2.npy"), pred_a2)

    rows = []
    gt1 = args.gt_big1 or default_gt_h5ad(args.data_root, 1)
    gt2 = args.gt_big2 or default_gt_h5ad(args.data_root, 2)
    m1 = maybe_eval_gt(
        gt1, adata1.obs_names, panel_b, pred_b1,
        adata1.obsm["spatial"], args.num_neighbors, "Slice1 panelB")
    if m1:
        rows.append(m1)
    m2 = maybe_eval_gt(
        gt2, adata2.obs_names, panel_a, pred_a2,
        adata2.obsm["spatial"], args.num_neighbors, "Slice2 panelA")
    if m2:
        rows.append(m2)

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "metrics_spatialexp_big.csv"), index=False)
    else:
        print("No ground-truth h5ad provided; saved predictions only.")

    log(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
