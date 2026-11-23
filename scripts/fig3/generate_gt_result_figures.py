#!/usr/bin/env python3
"""Generate result figures for 改进一: GT vs HGNN (Fig.3 + Fig.2)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import pearsonr

NOTO_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(NOTO_PATH)
CJK_FONT = font_manager.FontProperties(fname=NOTO_PATH)
plt.rcParams["font.sans-serif"] = [CJK_FONT.get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "image", "fig3_diagnosis")
os.makedirs(OUT_DIR, exist_ok=True)

C_HGNN = "#4e79a7"
C_GT = "#59a14f"
C_DEEPPT = "#e15759"
C_MNN = "#f28e2b"
C_NAVY = "#1A237E"


def save(fig, name):
    from figure_io import save_figure
    save_figure(fig, OUT_DIR, name)


def _annotate_bars(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, fmt.format(h),
                ha="center", va="bottom", fontsize=8)


def plot_fig3_hgnn_vs_gt_bar():
    """14: HGNN vs GT on Fig.3 official split (PCC + SSIM)."""
    summary = pd.read_csv(
        os.path.join(PROJECT_ROOT, "experiments/fig3/ssim_comparison/outputs/ssim_pcc_summary.csv")
    )
    sub = summary[summary["method"].isin(["HGNN+Cycle", "GT+Cycle"])].copy()
    sub["method"] = sub["method"].map({"HGNN+Cycle": "HGNN-512\n+ Cycle", "GT+Cycle": "GT-128\n+ Cycle"})

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    slices = ["Slice1 PanelB", "Slice2 PanelA"]
    colors = [C_HGNN, C_GT]
    for ax, metric, ylab, title in zip(
        axes,
        ["pcc_mean", "ssim_mean"],
        ["Gene-level PCC", "Gene-level SSIM"],
        ["Fig.3 PanelB / PanelA: PCC", "Fig.3 PanelB / PanelA: SSIM"],
    ):
        x = np.arange(len(slices))
        width = 0.32
        for i, method in enumerate(["HGNN-512\n+ Cycle", "GT-128\n+ Cycle"]):
            vals = []
            for sl in slices:
                row = sub[(sub["method"] == method) & (sub["slice"] == sl)]
                vals.append(row[metric].iloc[0] if len(row) else np.nan)
            bars = ax.bar(x + (i - 0.5) * width, vals, width, label=method.replace("\n", " "),
                          color=colors[i], alpha=0.92)
            _annotate_bars(ax, bars)
        ax.set_xticks(x)
        ax.set_xticklabels(["Slice1\n(predict PanelB)", "Slice2\n(predict PanelA)"], fontproperties=CJK_FONT)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontproperties=CJK_FONT, fontsize=11)
        ax.set_ylim(0, max(0.45, ax.get_ylim()[1] * 1.08))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("改进一 · Fig.3 Official Split: GT vs HGNN (Cycle 监督)",
                 fontproperties=CJK_FONT, fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "14_gt_vs_hgnn_fig3_bar.svg")


def plot_fig3_cycle_methods_bar():
    """17: all Cycle-family methods on Fig.3 (from ssim comparison summary)."""
    summary = pd.read_csv(
        os.path.join(PROJECT_ROOT, "experiments/fig3/ssim_comparison/outputs/ssim_pcc_summary.csv")
    )
    methods = summary["method"].unique()
    color_map = {"DeepPT": C_DEEPPT, "HGNN+Cycle": C_HGNN, "GT+Cycle": C_GT, "MLP+Strict MNN": C_MNN}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    slices = ["Slice1 PanelB", "Slice2 PanelA"]
    for ax, metric, title in zip(axes, ["pcc_mean", "ssim_mean"], ["Mean PCC", "Mean SSIM"]):
        x = np.arange(len(methods))
        width = 0.35
        for i, sl in enumerate(slices):
            vals = []
            for m in methods:
                row = summary[(summary["method"] == m) & (summary["slice"] == sl)]
                vals.append(row[metric].iloc[0] if len(row) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width, label=sl, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=18, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save(fig, "17_fig3_cycle_methods_bar.svg")


def plot_fig2_encoder_ablation():
    """15: Fig.2 encoder ablation (HGNN vs GT variants vs DeepPT)."""
    rows = [
        ("HGNN SpatialEx\n(512-d)", "experiments/fig2/spatialex_breast_cancer/outputs/metrics_spatialex.csv", C_HGNN),
        ("GT-128 + MFP", "experiments/fig2/gt_breast_cancer/outputs/metrics_gt.csv", C_GT),
        ("GT-512 + MFP", "experiments/fig2/gt512_breast_cancer/outputs/metrics_gt.csv", "#76b7b2"),
        ("GT-128 + DGI", "experiments/fig2/gt_dgi_breast_cancer/outputs/metrics_gt_dgi.csv", "#b07aa1"),
        ("DeepPT", "experiments/fig2/deeppt_breast_cancer/outputs/metrics_deeppt.csv", C_DEEPPT),
    ]
    labels, pcc_s1, pcc_s2, ssim_s1, ssim_s2, bar_colors = [], [], [], [], [], []
    for label, path, color in rows:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, path))
        s1 = df[df["slice"].str.contains("Slice1", na=False)].iloc[0]
        s2 = df[df["slice"].str.contains("Slice2", na=False)].iloc[0]
        labels.append(label)
        pcc_s1.append(s1["pcc_mean"])
        pcc_s2.append(s2["pcc_mean"])
        ssim_s1.append(s1["ssim_mean"])
        ssim_s2.append(s2["ssim_mean"])
        bar_colors.append(color)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(labels))
    width = 0.35
    for ax, s1_vals, s2_vals, ylab, title in zip(
        axes,
        [pcc_s1, ssim_s1],
        [pcc_s2, ssim_s2],
        ["Gene-level PCC", "Gene-level SSIM"],
        ["Fig.2 H&E→Transcriptome: PCC", "Fig.2 H&E→Transcriptome: SSIM"],
    ):
        b1 = ax.bar(x - width / 2, s1_vals, width, label="Slice1", color=bar_colors, alpha=0.75, edgecolor="white")
        b2 = ax.bar(x + width / 2, s2_vals, width, label="Slice2", color=bar_colors, alpha=1.0, edgecolor="white")
        _annotate_bars(ax, b1)
        _annotate_bars(ax, b2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontproperties=CJK_FONT, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("改进一 · Fig.2 对照: HGNN 仍优于 GT（即使 GT-512）",
                 fontproperties=CJK_FONT, fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "15_fig2_encoder_ablation.svg")


def per_gene_pcc(pred, gt):
    out = []
    for i in range(pred.shape[1]):
        p, g = pred[:, i], gt[:, i]
        if np.std(p) < 1e-8 or np.std(g) < 1e-8:
            out.append(np.nan)
        else:
            out.append(pearsonr(p, g)[0])
    return np.array(out)


def plot_gt_vs_hgnn_pergene_scatter():
    """16: per-gene PCC scatter HGNN vs GT on Fig.3."""
    panel = pd.read_csv(os.path.join(PROJECT_ROOT, "data/panel_split_official.csv"))
    pa = panel[panel.panel == "panelA"].gene.astype(str).tolist()
    pb = panel[panel.panel == "panelB"].gene.astype(str).tolist()
    rep1 = sc.read_h5ad(os.path.join(PROJECT_ROOT, "data/Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad"))
    rep2 = sc.read_h5ad(os.path.join(PROJECT_ROOT, "data/Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad"))
    rep1.var_names = rep1.var_names.astype(str)
    rep2.var_names = rep2.var_names.astype(str)
    gt_b1 = rep1[:, pb].X.toarray() if hasattr(rep1[:, pb].X, "toarray") else np.asarray(rep1[:, pb].X)
    gt_a2 = rep2[:, pa].X.toarray() if hasattr(rep2[:, pa].X, "toarray") else np.asarray(rep2[:, pa].X)

    hg_b1 = pd.read_csv(os.path.join(PROJECT_ROOT, "experiments/fig3/spatialexp_official/outputs/pred_panelB1_spatialexp.csv"), index_col=0).values
    hg_a2 = pd.read_csv(os.path.join(PROJECT_ROOT, "experiments/fig3/spatialexp_official/outputs/pred_panelA2_spatialexp.csv"), index_col=0).values
    gt_b1p = pd.read_csv(os.path.join(PROJECT_ROOT, "experiments/fig3/spatialexp_gt_official/outputs/pred_panelB1_spatialexp_gt.csv"), index_col=0).values
    gt_a2p = pd.read_csv(os.path.join(PROJECT_ROOT, "experiments/fig3/spatialexp_gt_official/outputs/pred_panelA2_spatialexp_gt.csv"), index_col=0).values

    pairs = [
        ("Slice1 PanelB", per_gene_pcc(hg_b1, gt_b1), per_gene_pcc(gt_b1p, gt_b1), len(pb)),
        ("Slice2 PanelA", per_gene_pcc(hg_a2, gt_a2), per_gene_pcc(gt_a2p, gt_a2), len(pa)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, (title, hg, gt, n_genes) in zip(axes, pairs):
        mask = np.isfinite(hg) & np.isfinite(gt)
        hg_m, gt_m = hg[mask], gt[mask]
        wins = (gt_m > hg_m).sum()
        ax.scatter(hg_m, gt_m, c=C_GT, s=22, alpha=0.65, edgecolors="none")
        lim = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])
        lo, hi = min(lim[0], -0.05), max(lim[1], 0.55)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("HGNN-512 + Cycle PCC")
        ax.set_ylabel("GT-128 + Cycle PCC")
        ax.set_title(f"{title} (n={n_genes})", fontproperties=CJK_FONT)
        ax.text(0.05, 0.95, f"GT wins: {wins}/{len(hg_m)}\nmean Δ={ (gt_m-hg_m).mean():+.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.9))
        ax.grid(alpha=0.25)
    fig.suptitle("改进一 · Fig.3 Per-Gene: GT 未系统性超越 HGNN",
                 fontproperties=CJK_FONT, fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "16_gt_vs_hgnn_pergene_scatter.svg")


def main():
    plot_fig3_hgnn_vs_gt_bar()
    plot_fig3_cycle_methods_bar()
    plot_fig2_encoder_ablation()
    plot_gt_vs_hgnn_pergene_scatter()
    print(f"\nAll GT result figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
