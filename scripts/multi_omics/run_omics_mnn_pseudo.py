#!/usr/bin/env python3
"""
Omics diagonal integration (transcriptomics ↔ proteomics) with strict MNN pseudo-labels.

Protocol (Tutorial 4 / Fig. omics diagonal):
  Slice1 (Rep1): measured protein P1, held-out RNA R1 (eval only)
  Slice2 (Rep2): measured RNA R2, held-out protein P2 (eval only)

Strict MNN bridges (analogous to Fig.3 panel diagonal):
  Step 1: H&E cross-slice MNN (HE1 ↔ HE2) → pseudo R1 from R2
  Step 2: RNA cross-slice MNN (R2 ↔ pseudo R1) → pseudo P2 from P1

Train two MLPs:
  protein → pseudo RNA on slice1
  RNA → pseudo protein on slice2
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "fig3"))

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import SpatialEx as se
import run_fig3_mnn_pseudo as mnn


def parse_args():
    p = argparse.ArgumentParser(description="MLP + Strict MNN for omics diagonal integration")
    p.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--rep1", type=str, default="Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad")
    p.add_argument("--rep2", type=str, default="Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad")
    p.add_argument("--niche_h5ad", type=str, default=os.path.join("10x_breast_cancer", "human_breast_cancer.h5ad"))
    p.add_argument("--protein_features", type=str, default="cell_CD20_mean,cell_HER2_mean")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mnn_k", type=int, default=20)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "experiments", "multi_omics", "mnn_pseudo", "outputs"))
    return p.parse_args()


def load_protein_slice(data_root, rep1_path, niche_path, protein_cols):
    niche = sc.read_h5ad(os.path.join(data_root, niche_path))
    niche.obs_names = niche.obs_names.astype(str)
    rep1 = sc.read_h5ad(os.path.join(data_root, rep1_path))
    rep1.obs_names = rep1.obs_names.astype(str)

    common = list(set(niche.obs_names).intersection(rep1.obs_names))
    niche = niche[common].copy()
    rep1 = rep1[common].copy()

    protein_x = niche.obs[protein_cols].values.astype(np.float32)
    adata = sc.AnnData(X=protein_x, obs=niche.obs.copy())
    adata.var_names = pd.Index(protein_cols)
    adata.obsm["spatial"] = rep1.obsm["spatial"].copy()
    adata.obsm["he"] = rep1.obsm["he"].copy()
    adata.obs["x_centroid"] = rep1.obs["x_centroid"].values
    adata.obs["y_centroid"] = rep1.obs["y_centroid"].values
    sc.pp.scale(adata)
    return adata, niche


def load_rna_slice(data_root, rep2_path):
    adata = sc.read_h5ad(os.path.join(data_root, rep2_path))
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values
    return adata


def get_dense(adata):
    x = adata.X
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    protein_cols = [c.strip() for c in args.protein_features.split(",")]
    adata_prot, niche_rep1 = load_protein_slice(
        args.data_root, args.rep1, args.niche_h5ad, protein_cols)
    adata_rna = load_rna_slice(args.data_root, args.rep2)

    niche = sc.read_h5ad(os.path.join(args.data_root, args.niche_h5ad))
    niche.obs_names = niche.obs_names.astype(str)
    niche_rep2 = niche[adata_rna.obs_names].copy()

    P1 = get_dense(adata_prot).astype(np.float32)
    R2 = get_dense(adata_rna).astype(np.float32)
    R1_gt = get_dense(niche_rep1).astype(np.float32)
    P2_gt = niche_rep2.obs[protein_cols].values.astype(np.float32)
    HE1 = np.asarray(adata_prot.obsm["he"], dtype=np.float32)
    HE2 = np.asarray(adata_rna.obsm["he"], dtype=np.float32)

    gene_names = adata_rna.var_names.tolist()
    print(f"Slice1 protein: {P1.shape[0]} cells x {P1.shape[1]} proteins")
    print(f"Slice2 RNA: {R2.shape[0]} cells x {R2.shape[1]} genes")

    graph1 = se.pp.Build_graph(
        adata_prot.obsm["spatial"], graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo")
    graph2 = se.pp.Build_graph(
        adata_rna.obsm["spatial"], graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo")

    print("=== Strict MNN pseudo labels (omics diagonal) ===")
    pseudo_R1 = mnn.build_mnn_pseudo(HE1, HE2, R2, k=args.k, mnn_k=args.mnn_k, device=device)
    pseudo_P2 = mnn.build_mnn_pseudo(R2, pseudo_R1, P1, k=args.k, mnn_k=args.mnn_k, device=device)

    ceil_r1 = mnn.evaluate(R1_gt, pseudo_R1, graph1)
    ceil_p2 = mnn.evaluate(P2_gt, pseudo_P2, graph2)
    print(f"Pseudo ceiling slice1 RNA: PCC={ceil_r1[0]:.4f} SSIM={ceil_r1[1]:.4f} CMD={ceil_r1[2]:.4f}")
    print(f"Pseudo ceiling slice2 protein: PCC={ceil_p2[0]:.4f} SSIM={ceil_p2[1]:.4f} CMD={ceil_p2[2]:.4f}")

    print("=== Training MLP on strict MNN pseudo labels ===")
    model_r1, model_p2 = mnn.train_panel_mlp(
        P1, pseudo_R1, R2, pseudo_P2,
        P1.shape[1], R2.shape[1], P1.shape[1],
        args.hidden_dim, args.epochs, args.lr, device)

    def predict(model, x, y_ref):
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(mnn.zscore(x), dtype=torch.float32, device=device)
            pred = model(x_t).cpu().numpy()
        pred = pred * y_ref.std(axis=0) + y_ref.mean(axis=0)
        return pred

    pred_R1 = predict(model_r1, P1, R2)
    pred_P2 = predict(model_p2, R2, P1)

    learn_r1 = mnn.evaluate(R1_gt, pred_R1, graph1)
    learn_p2 = mnn.evaluate(P2_gt, pred_P2, graph2)
    print(f"Learned slice1 RNA: PCC={learn_r1[0]:.4f} SSIM={learn_r1[1]:.4f} CMD={learn_r1[2]:.4f}")
    print(f"Learned slice2 protein: PCC={learn_p2[0]:.4f} SSIM={learn_p2[1]:.4f} CMD={learn_p2[2]:.4f}")

    rows = [
        {"direction": "slice1_protein_to_RNA", "what": "pseudo_ceiling",
         "pcc": ceil_r1[0], "ssim": ceil_r1[1], "cmd": ceil_r1[2]},
        {"direction": "slice1_protein_to_RNA", "what": "mlp_mnn",
         "pcc": learn_r1[0], "ssim": learn_r1[1], "cmd": learn_r1[2]},
        {"direction": "slice2_RNA_to_protein", "what": "pseudo_ceiling",
         "pcc": ceil_p2[0], "ssim": ceil_p2[1], "cmd": ceil_p2[2]},
        {"direction": "slice2_RNA_to_protein", "what": "mlp_mnn",
         "pcc": learn_p2[0], "ssim": learn_p2[1], "cmd": learn_p2[2]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "omics_mnn_metrics.csv"), index=False)

    pd.DataFrame(pred_R1, index=adata_prot.obs_names, columns=gene_names).to_csv(
        os.path.join(args.out_dir, "pred_RNA_slice1.csv"))
    pd.DataFrame(pred_P2, index=adata_rna.obs_names, columns=protein_cols).to_csv(
        os.path.join(args.out_dir, "pred_protein_slice2.csv"))
    print(f"\nSaved to {args.out_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
