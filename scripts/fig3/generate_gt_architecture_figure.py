#!/usr/bin/env python3
"""Minimal SpatialEx-GT architecture diagram (structure only, no prose)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "image", "fig3_diagnosis")
os.makedirs(OUT_DIR, exist_ok=True)

C_NAVY = "#1A237E"
C_ORANGE = "#E85D04"
C_BLUE = "#4C78A8"
C_PALE = "#EFF2FF"
C_GREEN = "#59A14F"
C_GRAY = "#666666"


def _box(ax, xy, wh, text, fc=C_PALE, ec=C_NAVY, fs=10, lw=1.5):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#222")


def _arrow(ax, a, b, color=C_GRAY, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=color,
        shrinkA=2, shrinkB=2,
    ))


def plot_gt_architecture():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    _box(ax, (0.2, 2.55), (1.5, 0.9), r"$X$", fc="#DCE6F7", ec=C_BLUE)
    _box(ax, (0.2, 1.35), (1.5, 0.9), r"$(x,y)$", fc="#DCE6F7", ec=C_ORANGE)
    _box(ax, (2.1, 1.95), (1.4, 0.9), "k-NN", fc="#FFF3E6", ec=C_ORANGE)

    enc = FancyBboxPatch(
        (4.0, 1.2), 2.8, 2.0,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        linewidth=2.0, edgecolor=C_NAVY, facecolor="#FFFFFF",
    )
    ax.add_patch(enc)
    ax.text(5.4, 2.55, "GT Encoder", ha="center", fontsize=11, weight="bold", color=C_NAVY)
    ax.text(5.4, 2.05, r"sparse attn", ha="center", fontsize=9, color="#333")
    ax.text(5.4, 1.55, r"$h_i$", ha="center", fontsize=10, color="#333")

    _box(ax, (7.4, 2.65), (1.55, 0.75), r"$\hat{Y}$", fc="#E8F5E9", ec=C_GREEN, fs=9)
    _box(ax, (7.4, 1.75), (1.55, 0.75), "MFP", fc="#FFF8E1", ec=C_ORANGE, fs=9)
    _box(ax, (7.4, 0.85), (1.55, 0.75), r"$C$", fc="#FCE4EC", ec="#AD1457", fs=10)

    _arrow(ax, (1.7, 3.0), (2.1, 2.4), C_BLUE)
    _arrow(ax, (1.7, 1.8), (2.1, 2.4), C_ORANGE)
    _arrow(ax, (3.5, 2.4), (4.0, 2.4))
    _arrow(ax, (6.8, 2.7), (7.4, 3.02), C_GREEN)
    _arrow(ax, (6.8, 2.2), (7.4, 2.12), C_ORANGE)
    _arrow(ax, (6.8, 1.7), (7.4, 1.22), "#AD1457")

    ax.text(8.35, 3.02, r"$\mathcal{L}_{\mathrm{sup}}$", fontsize=9, color=C_GREEN)
    ax.text(8.35, 2.12, r"$\mathcal{L}_{\mathrm{MFP}}$", fontsize=9, color=C_ORANGE)
    ax.text(8.35, 1.22, r"$\mathcal{L}_{\mathrm{cycle}}$", fontsize=9, color="#AD1457")

    out_path = os.path.join(OUT_DIR, "12_gt_architecture.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    plot_gt_architecture()
