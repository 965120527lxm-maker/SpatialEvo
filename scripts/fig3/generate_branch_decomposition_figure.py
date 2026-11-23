#!/usr/bin/env python3
"""Minimal Branch Decomposition design diagram (structure only)."""

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
C_GREEN = "#59A14F"
C_ORANGE = "#E85D04"
C_PURPLE = "#7B1FA2"
C_HE = "#DCE6F7"
C_PANEL = "#E8F5E9"


def _box(ax, xy, wh, text, fc, ec, fs=10):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=ec, facecolor=fc,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#222")


def _arrow(ax, a, b, color="#666", lw=1.4):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=11, linewidth=lw, color=color,
        shrinkA=2, shrinkB=2,
    ))


def plot_branch_decomposition_design():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    y_he, y_pan = 2.55, 1.35
    # Slice1 row
    _box(ax, (0.4, y_he), (0.9, 0.6), r"$X_1$", C_HE, C_BLUE)
    _box(ax, (1.5, y_he), (1.0, 0.6), r"$P_B$", "#EFF2FF", C_BLUE, fs=9)
    _box(ax, (2.7, y_he), (0.9, 0.6), r"$\hat{Y}_B^1$", "#FFF3E0", C_ORANGE)

    _box(ax, (0.4, y_pan), (0.9, 0.6), r"$Y_A^1$", C_PANEL, C_GREEN)
    _box(ax, (1.5, y_pan), (1.0, 0.6), r"$C_{A\to B}$", "#EFF2FF", C_GREEN, fs=9)
    _box(ax, (2.7, y_pan), (0.9, 0.6), r"$\hat{Y}_B^1$", "#FFF3E0", C_ORANGE)

    _arrow(ax, (1.3, y_he + 0.3), (1.5, y_he + 0.3), C_BLUE)
    _arrow(ax, (2.5, y_he + 0.3), (2.7, y_he + 0.3), C_BLUE)
    _arrow(ax, (1.3, y_pan + 0.3), (1.5, y_pan + 0.3), C_GREEN)
    _arrow(ax, (2.5, y_pan + 0.3), (2.7, y_pan + 0.3), C_GREEN)

    ax.text(0.15, y_he + 0.3, "H&E", fontsize=9, weight="bold", color=C_BLUE, va="center")
    ax.text(0.05, y_pan + 0.3, "Panel", fontsize=9, weight="bold", color=C_GREEN, va="center")

    # Slice2 row
    _box(ax, (4.2, y_he), (0.9, 0.6), r"$X_2$", C_HE, C_BLUE)
    _box(ax, (5.3, y_he), (1.0, 0.6), r"$P_A$", "#EFF2FF", C_BLUE, fs=9)
    _box(ax, (6.5, y_he), (0.9, 0.6), r"$\hat{Y}_A^2$", "#FFF3E0", C_ORANGE)

    _box(ax, (4.2, y_pan), (0.9, 0.6), r"$Y_B^2$", C_PANEL, C_GREEN)
    _box(ax, (5.3, y_pan), (1.0, 0.6), r"$C_{B\to A}$", "#EFF2FF", C_GREEN, fs=9)
    _box(ax, (6.5, y_pan), (0.9, 0.6), r"$\hat{Y}_A^2$", "#FFF3E0", C_ORANGE)

    _arrow(ax, (5.1, y_he + 0.3), (5.3, y_he + 0.3), C_BLUE)
    _arrow(ax, (6.3, y_he + 0.3), (6.5, y_he + 0.3), C_BLUE)
    _arrow(ax, (5.1, y_pan + 0.3), (5.3, y_pan + 0.3), C_GREEN)
    _arrow(ax, (6.3, y_pan + 0.3), (6.5, y_pan + 0.3), C_GREEN)

    # Fusion
    _box(ax, (2.0, 0.25), (2.2, 0.65), "Fusion", "#F3E5F5", C_PURPLE, fs=10)
    _arrow(ax, (3.15, y_pan), (3.1, 0.92), C_PURPLE, lw=1.2)
    _arrow(ax, (3.15, y_he), (3.1, 0.92), C_PURPLE, lw=1.0)
    _arrow(ax, (5.4, y_pan), (3.5, 0.55), C_PURPLE, lw=1.0)
    _arrow(ax, (5.4, y_he), (3.5, 0.55), C_PURPLE, lw=1.0)

    out_path = os.path.join(OUT_DIR, "18_branch_decomposition_design.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    plot_branch_decomposition_design()
