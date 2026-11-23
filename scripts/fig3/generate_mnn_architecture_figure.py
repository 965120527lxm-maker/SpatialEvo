#!/usr/bin/env python3
"""Minimal Strict MNN pipeline diagram (structure only)."""

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

C_BLUE = "#4C78A8"
C_ORANGE = "#E85D04"
C_GREEN = "#59A14F"
C_NAVY = "#1A237E"
C_RED = "#C62828"


def _box(ax, xy, wh, text, fc="#EFF2FF", ec=C_NAVY, fs=10):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=ec, facecolor=fc,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#222")


def _arrow(ax, a, b, color=C_NAVY, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=11, linewidth=lw, color=color,
        shrinkA=2, shrinkB=2,
    ))


def plot_mnn_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    # Slice inputs
    _box(ax, (0.3, 3.35), (1.2, 0.7), r"$X_1$", fc="#DCE6F7", ec=C_BLUE, fs=11)
    _box(ax, (0.3, 2.45), (1.2, 0.7), r"$Y_A^1$", fc="#E8F5E9", ec=C_GREEN, fs=10)
    _box(ax, (2.0, 3.35), (1.2, 0.7), r"$X_2$", fc="#FFF3E0", ec=C_ORANGE, fs=11)
    _box(ax, (2.0, 2.45), (1.2, 0.7), r"$Y_B^2$", fc="#FFF3E0", ec=C_ORANGE, fs=10)

    # Step 1
    s1 = FancyBboxPatch(
        (3.6, 2.85), 2.6, 1.35,
        boxstyle="round,pad=0.03,rounding_size=0.1", linewidth=1.8, edgecolor=C_BLUE, facecolor="#FFFFFF",
    )
    ax.add_patch(s1)
    ax.text(4.9, 3.75, "Step 1", ha="center", fontsize=9, weight="bold", color=C_BLUE)
    ax.text(4.9, 3.45, "MNN", ha="center", fontsize=10, weight="bold", color=C_BLUE)
    ax.text(4.9, 3.25, r"$X_1\leftrightarrow X_2$", ha="center", fontsize=9)
    ax.text(4.9, 2.95, r"$\tilde{Y}_B^1$", ha="center", fontsize=10, color=C_GREEN)

    # Step 2
    s2 = FancyBboxPatch(
        (3.6, 1.0), 2.6, 1.35,
        boxstyle="round,pad=0.03,rounding_size=0.1", linewidth=1.8, edgecolor=C_ORANGE, facecolor="#FFFFFF",
    )
    ax.add_patch(s2)
    ax.text(4.9, 1.9, "Step 2", ha="center", fontsize=9, weight="bold", color=C_ORANGE)
    ax.text(4.9, 1.55, "MNN", ha="center", fontsize=10, weight="bold", color=C_ORANGE)
    ax.text(4.9, 1.4, r"$Y_B^2\leftrightarrow\tilde{Y}_B^1$", ha="center", fontsize=8.5)
    ax.text(4.9, 1.1, r"$\tilde{Y}_A^2$", ha="center", fontsize=10, color=C_GREEN)

    # MLP
    mlp = FancyBboxPatch(
        (7.0, 1.55), 1.8, 1.5,
        boxstyle="round,pad=0.03,rounding_size=0.1", linewidth=1.8, edgecolor=C_NAVY, facecolor=C_NAVY,
    )
    ax.add_patch(mlp)
    ax.text(7.9, 2.3, "MLP", ha="center", fontsize=12, weight="bold", color="white")
    ax.text(7.9, 1.85, r"$\hat{Y}$", ha="center", fontsize=11, color="white")

    _arrow(ax, (0.9, 3.7), (3.6, 3.5), C_BLUE)
    _arrow(ax, (2.6, 3.7), (3.6, 3.5), C_ORANGE)
    _arrow(ax, (4.9, 2.85), (4.9, 2.35), C_GREEN)
    _arrow(ax, (2.6, 2.8), (3.6, 1.65), C_ORANGE)
    _arrow(ax, (4.9, 1.0), (4.9, 0.55), C_GREEN)
    _arrow(ax, (6.2, 1.65), (7.0, 2.1))
    _arrow(ax, (6.2, 3.2), (7.0, 2.5), C_GREEN)

    _box(ax, (8.9, 2.65), (1.5, 0.65), r"$\tilde{Y}_B^1$", fc="#E8F5E9", ec=C_GREEN, fs=9)
    _box(ax, (8.9, 1.55), (1.5, 0.65), r"$\tilde{Y}_A^2$", fc="#E8F5E9", ec=C_GREEN, fs=9)

    out_path = os.path.join(OUT_DIR, "13_mnn_architecture.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    plot_mnn_architecture()
