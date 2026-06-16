#!/usr/bin/env python3
"""
Reproduce and extend SpatialEx+ Fig. 3: panel diagonal integration.

The 10x Xenium Human Breast Cancer 313-gene panel is split into two disjoint
panels A/B. Slice 1 is assigned panel A, slice 2 panel B. The held-out panels
(slice 1 panel B, slice 2 panel A) serve as ground truth for evaluation.

Models compared:
- spatialexp       : original SpatialEx+ (H&E only -> missing panel via cycle)
- spatialexp_small : reduced HGNN baseline (h=128)
- spatialexp_gt    : Graph Transformer baseline (h=128)
- conditional      : measured-panel-conditioned completion (X, Y_measured -> Y_missing)
- conditional_mlp  : lightweight conditional MLP using measured-panel pseudo-labels
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
    parser = argparse.ArgumentParser(description="SpatialEx+ Fig.3 panel split")
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--rep1", type=str,
                        default="Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad")
    parser.add_argument("--rep2", type=str,
                        default="Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad")
    parser.add_argument("--model", type=str, default="conditional",
                        choices=["spatialexp", "spatialexp_small", "spatialexp_gt", "conditional", "conditional_mlp"])
    parser.add_argument("--panelA_size", type=int, default=150)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--translator_hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pseudo_k", type=int, default=5,
                        help="k for cross-slice H&E nearest neighbor pseudo-labels / matched pairs (conditional only)")
    parser.add_argument("--mlp_mode", type=str, default="measured_pseudo",
                        choices=["he_conditional", "panel_translator", "measured_pseudo"],
                        help="ConditionalMLP variant (conditional_mlp only)")
    parser.add_argument("--mlp_use_he", action="store_true",
                        help="Concatenate H&E to the measured panel input in measured_pseudo mode")
    parser.add_argument("--num_neighbors", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "conditional", "fig3_panel_split"))
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

    # 1. Load full slices
    rep1_full = load_slice(args.rep1, args.data_root)
    rep2_full = load_slice(args.rep2, args.data_root)
    print(f"Rep1: {rep1_full.shape}, Rep2: {rep2_full.shape}")

    # 2. Split panel A/B (same split for both slices)
    genes = rep1_full.var_names.values
    np.random.seed(args.seed)
    np.random.shuffle(genes)
    panelA_genes = genes[:args.panelA_size].tolist()
    panelB_genes = genes[args.panelA_size:].tolist()
    print(f"Panel A: {len(panelA_genes)} genes, Panel B: {len(panelB_genes)} genes")

    rep1_A, rep1_B = split_panels(rep1_full, panelA_genes, panelB_genes)
    rep2_A, rep2_B = split_panels(rep2_full, panelA_genes, panelB_genes)

    # 3. Build hypergraphs (spatial only, based on original coords)
    print("\n=== Building hypergraphs ===")
    graph1 = se.pp.Build_hypergraph_spatial_and_HE(
        rep1_A, args.num_neighbors, graph_kind="spatial", return_type="csr"
    )
    graph2 = se.pp.Build_hypergraph_spatial_and_HE(
        rep2_B, args.num_neighbors, graph_kind="spatial", return_type="csr"
    )

    # 4. Train model
    print(f"\n=== Training {args.model} ===")
    if args.model == "spatialexp":
        trainer = se.SpatialExP(
            rep1_A, rep2_B, graph1, graph2,
            device=device, epochs=args.epochs, lr=args.lr,
            hidden_dim=args.hidden_dim, translator_hidden_dim=args.translator_hidden_dim,
        )
    elif args.model == "spatialexp_small":
        trainer = se.SpatialExP_Small(
            rep1_A, rep2_B, graph1, graph2,
            device=device, epochs=args.epochs, lr=args.lr,
            hidden_dim=args.hidden_dim, translator_hidden_dim=args.translator_hidden_dim,
        )
    elif args.model == "spatialexp_gt":
        trainer = se.SpatialExP_GT(
            rep1_A, rep2_B, graph1, graph2,
            device=device, epochs=args.epochs, lr=args.lr,
            hidden_dim=args.hidden_dim, translator_hidden_dim=args.translator_hidden_dim,
        )
    elif args.model == "conditional":
        trainer = se.SpatialExP_Conditional(
            rep1_A, rep2_B, graph1, graph2,
            pseudo_k=args.pseudo_k,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
    elif args.model == "conditional_mlp":
        kwargs = dict(
            mode=args.mlp_mode,
            pseudo_k=args.pseudo_k,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        if args.mlp_mode == "measured_pseudo":
            kwargs["measured_A2"] = np.asarray(rep2_A.X.toarray() if hasattr(rep2_A.X, 'toarray') else rep2_A.X)
            kwargs["measured_B1"] = np.asarray(rep1_B.X.toarray() if hasattr(rep1_B.X, 'toarray') else rep1_B.X)
            kwargs["use_he"] = args.mlp_use_he
        trainer = se.SpatialExP_ConditionalMLP(rep1_A, rep2_B, **kwargs)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    trainer.train()

    # 5. Inference
    print("\n=== Inference ===")
    if args.model in ("spatialexp", "spatialexp_small", "spatialexp_gt"):
        pred_B1 = trainer.inference_indirect(rep1_A.obsm["he"], graph1, panel="panelB")
        pred_A2 = trainer.inference_indirect(rep2_B.obsm["he"], graph2, panel="panelA")
    else:
        pred_B1 = trainer.predict_panelB_on_slice1(
            rep1_A.obsm["he"], np.asarray(rep1_A.X.toarray() if hasattr(rep1_A.X, 'toarray') else rep1_A.X),
            graph1
        )
        pred_A2 = trainer.predict_panelA_on_slice2(
            rep2_B.obsm["he"], np.asarray(rep2_B.X.toarray() if hasattr(rep2_B.X, 'toarray') else rep2_B.X),
            graph2
        )

    pred_B1_df = pd.DataFrame(pred_B1, index=rep1_A.obs_names, columns=panelB_genes)
    pred_A2_df = pd.DataFrame(pred_A2, index=rep2_B.obs_names, columns=panelA_genes)
    pred_B1_df.to_csv(os.path.join(args.out_dir, f"pred_panelB1_{args.model}.csv"))
    pred_A2_df.to_csv(os.path.join(args.out_dir, f"pred_panelA2_{args.model}.csv"))

    # 6. Evaluation
    print("\n=== Evaluation ===")
    gt_B1 = rep1_B[rep1_A.obs_names, panelB_genes]
    gt_A2 = rep2_A[rep2_B.obs_names, panelA_genes]

    metrics_B1 = evaluate(gt_B1, pred_B1_df.values, label=f"Slice1 PanelB prediction ({args.model})")
    metrics_A2 = evaluate(gt_A2, pred_A2_df.values, label=f"Slice2 PanelA prediction ({args.model})")

    summary = {
        "model": args.model,
        "slice1_pcc": metrics_B1["pcc"],
        "slice1_ssim": metrics_B1["ssim"],
        "slice1_cmd": metrics_B1["cmd"],
        "slice2_pcc": metrics_A2["pcc"],
        "slice2_ssim": metrics_A2["ssim"],
        "slice2_cmd": metrics_A2["cmd"],
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, f"metrics_{args.model}.csv"), index=False)
    print(f"\nOutputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
