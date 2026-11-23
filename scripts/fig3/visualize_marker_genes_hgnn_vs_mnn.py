#!/usr/bin/env python3
"""Spatial heatmaps: Ground truth vs HGNN+Cycle vs MLP+MNN (Slice2 PanelA)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "fig3"))

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

DEFAULT_HGNN_PRED = os.path.join(
    PROJECT_ROOT, "experiments/fig3/spatialexp_official/outputs/pred_panelA2_spatialexp.csv"
)
DEFAULT_MNN_PRED = os.path.join(
    PROJECT_ROOT, "experiments/fig3/mnn_pseudo_strict_official/outputs/pred_panelA2_mnn.csv"
)
DEFAULT_GAIN_CSV = os.path.join(
    PROJECT_ROOT, "experiments/fig3/top_marker_gains_hgnn_vs_mnn.csv"
)
DEFAULT_PANEL_CSV = os.path.join(PROJECT_ROOT, "data/panel_split_official.csv")
DEFAULT_OUT = os.path.join(
    PROJECT_ROOT, "docs/image/fig3_diagnosis/11_marker_genes_slice2.jpg"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--rep2", default="Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad")
    p.add_argument("--panel_csv", default=DEFAULT_PANEL_CSV)
    p.add_argument("--hgnn_pred", default=DEFAULT_HGNN_PRED)
    p.add_argument("--mnn_pred", default=DEFAULT_MNN_PRED)
    p.add_argument("--gain_csv", default=DEFAULT_GAIN_CSV)
    p.add_argument("--genes", nargs="+", default=None,
                   help="Marker genes to plot; default = top 4 by ΔPCC from gain_csv")
    p.add_argument("--top_n", type=int, default=4)
    p.add_argument("--point_size", type=float, default=0.35)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args()


def load_rep2(path, data_root):
    adata = sc.read_h5ad(os.path.join(data_root, path))
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values
    return adata


def per_gene_pcc(y_true, y_pred, gene_names):
    out = {}
    for i, gene in enumerate(gene_names):
        a = y_true[:, i]
        b = y_pred[:, i]
        if a.std() < 1e-8 or b.std() < 1e-8:
            out[gene] = float("nan")
        else:
            out[gene] = float(np.corrcoef(a, b)[0, 1])
    return out


def pick_genes(args, panel_genes):
    if args.genes:
        genes = [g for g in args.genes if g in panel_genes]
        missing = set(args.genes) - set(genes)
        for g in missing:
            print(f"Warning: {g} not in PanelA, skipping.")
        return genes
    df = pd.read_csv(args.gain_csv)
    top = df.nlargest(args.top_n, "gain")["gene"].tolist()
    return [g for g in top if g in panel_genes][: args.top_n]


def plot_spatial(ax, coords, values, title, cmap="viridis", s=0.35, alpha=0.85):
    vmax = np.percentile(values, 99)
    vmin = np.percentile(values, 1)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    scat = ax.scatter(
        coords[:, 0], coords[:, 1], c=values, cmap=cmap,
        vmin=vmin, vmax=vmax, s=s, alpha=alpha, edgecolors="none", rasterized=True,
    )
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.01)
    return scat


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    panel_df = pd.read_csv(args.panel_csv)
    panelA = panel_df[panel_df["panel"] == "panelA"]["gene"].astype(str).tolist()

    rep2 = load_rep2(args.rep2, args.data_root)
    gt_X = rep2[:, panelA].X
    if hasattr(gt_X, "toarray"):
        gt_X = gt_X.toarray()
    coords = np.asarray(rep2.obsm["spatial"], dtype=np.float32)

    hgnn_df = pd.read_csv(args.hgnn_pred)
    mnn_df = pd.read_csv(args.mnn_pred)
    if not (hgnn_df["cell_id"].astype(str).values == rep2.obs_names.astype(str).values).all():
        raise ValueError("HGNN prediction cell_id order does not match Rep2 obs_names")
    if not (mnn_df["cell_id"].astype(str).values == rep2.obs_names.astype(str).values).all():
        raise ValueError("MNN prediction cell_id order does not match Rep2 obs_names")

    genes = pick_genes(args, panelA)
    if not genes:
        raise ValueError("No valid genes to plot")

    hgnn_mat = hgnn_df[genes].values.astype(np.float32)
    mnn_mat = mnn_df[genes].values.astype(np.float32)
    gt_sub = gt_X[:, [panelA.index(g) for g in genes]]

    pcc_hgnn = per_gene_pcc(gt_sub, hgnn_mat, genes)
    pcc_mnn = per_gene_pcc(gt_sub, mnn_mat, genes)

    n = len(genes)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.1 * n), constrained_layout=False)
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Ground truth", "HGNN+Cycle", "MLP+MNN"]
    for i, gene in enumerate(genes):
        plot_spatial(axes[i, 0], coords, gt_sub[:, i], f"{gene}\n{col_titles[0]}",
                     s=args.point_size)
        plot_spatial(
            axes[i, 1], coords, hgnn_mat[:, i],
            f"{gene}\n{col_titles[1]}\nPCC={pcc_hgnn[gene]:.3f}",
            s=args.point_size,
        )
        scat = plot_spatial(
            axes[i, 2], coords, mnn_mat[:, i],
            f"{gene}\n{col_titles[2]}\nPCC={pcc_mnn[gene]:.3f}",
            s=args.point_size,
        )
        # Dedicated colorbar axes to avoid overlap with the rightmost panel
        divider = make_axes_locatable(axes[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.12)
        cbar = fig.colorbar(scat, cax=cax)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle("PanelA (Slice2) marker gene predictions", fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.03, hspace=0.38, wspace=0.12)

    ext = os.path.splitext(args.out)[1].lower()
    if ext in (".jpg", ".jpeg"):
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    else:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"Saved: {args.out}")
    print("Genes:", genes)
    for g in genes:
        print(f"  {g}: HGNN+Cycle PCC={pcc_hgnn[g]:.3f}, MLP+MNN PCC={pcc_mnn[g]:.3f}, "
              f"Δ={pcc_mnn[g]-pcc_hgnn[g]:+.3f}")


if __name__ == "__main__":
    main()
