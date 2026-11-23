#!/usr/bin/env python3
"""Minimal MNN construction diagrams (structure only)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "image", "fig3_diagnosis")
os.makedirs(OUT_DIR, exist_ok=True)

C_BLUE = "#4C78A8"
C_ORANGE = "#E85D04"
C_GREEN = "#59A14F"
C_NAVY = "#1A237E"
C_PURPLE = "#7B1FA2"


def _arrow(ax, a, b, color="#666", lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=11, linewidth=lw, color=color,
        shrinkA=2, shrinkB=2,
    ))


def _region(ax, xy, wh, fc, ec, alpha=0.35, lw=1.2):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, zorder=0,
    ))


def _cell(ax, xy, color, ec, r=0.13, lw=1.6, zorder=2, alpha=1.0):
    ax.add_patch(Circle(xy, r, facecolor=color, edgecolor=ec, lw=lw, zorder=zorder, alpha=alpha))


def plot_mnn_filter_algorithm():
    """Visualize build_mnn_pseudo: forward candidates, reverse filter, mutual mean."""
    fig, ax = plt.subplots(figsize=(10.0, 4.9))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    # Panel backgrounds (extra bottom margin for pipeline + labels)
    _region(ax, (0.12, 0.72), (2.05, 3.55), "#EEF4FB", C_BLUE, alpha=0.55)
    _region(ax, (2.35, 0.72), (4.55, 3.55), "#FAFAFA", "#BBBBBB", alpha=0.45)
    _region(ax, (7.05, 0.72), (2.55, 3.55), "#EEF8EE", C_GREEN, alpha=0.55)

    ax.text(1.15, 4.05, r"$Q$", ha="center", fontsize=12, weight="bold", color=C_BLUE)
    ax.text(4.55, 4.05, r"$R$", ha="center", fontsize=12, weight="bold", color=C_ORANGE)
    ax.text(8.32, 4.05, r"$\tilde{y}_i$", ha="center", fontsize=12, weight="bold", color=C_GREEN)

    # Query cloud (cell i highlighted)
    i_star = (1.35, 2.55)
    q_bg = [(0.75, 3.15), (1.05, 1.95), (1.85, 2.95), (1.65, 1.75)]
    for x, y in q_bg:
        _cell(ax, (x, y), "#D8E4F2", C_BLUE, r=0.1, lw=1.0, alpha=0.85)
    _cell(ax, i_star, C_BLUE, C_NAVY, r=0.15, lw=2.2)
    ax.text(i_star[0], i_star[1] - 0.38, r"$i$", ha="center", fontsize=10, weight="bold", color=C_NAVY)

    # Reference cloud (keep away from label bands)
    r_bg = [(3.05, 3.35), (3.35, 1.95), (5.35, 1.55), (6.45, 1.85)]
    for x, y in r_bg:
        _cell(ax, (x, y), "#FFE8CC", "#FFCC80", r=0.1, lw=1.0, alpha=0.8)

    mutual = {
        "j1": (3.55, 3.05),
        "j2": (4.55, 2.35),
    }
    reject = (6.05, 3.05)

    for key, (x, y) in mutual.items():
        _cell(ax, (x, y), C_ORANGE, C_ORANGE, r=0.14, lw=2.0)
        ax.text(x, y - 0.36, f"${key.replace('j', 'j_')}$", ha="center", fontsize=9, color=C_ORANGE)

    _cell(ax, reject, "#FFE0B2", "#FFAB91", r=0.13, lw=1.4, alpha=0.95)
    ax.text(reject[0], reject[1] - 0.36, r"$j_3$", ha="center", fontsize=9, color="#BF360C")

    # Forward candidates (top-mnn_k)
    for tgt in list(mutual.values()) + [reject]:
        _arrow(ax, (i_star[0] + 0.12, i_star[1] + 0.08), (tgt[0] - 0.14, tgt[1] + 0.02),
               C_BLUE, lw=2.0 if tgt in mutual.values() else 1.2)
    ax.text(4.55, 3.62, r"$\mathcal{F}(i)$", ha="center", fontsize=10, color=C_BLUE, weight="bold")

    # Reverse mutual links (only j1, j2)
    for tgt in mutual.values():
        _arrow(ax, (tgt[0] + 0.08, tgt[1] - 0.14), (i_star[0] + 0.05, i_star[1] - 0.14),
               C_GREEN, lw=1.5, style="-|>")
    ax.text(2.95, 2.05, r"$\mathcal{R}(j)\ni i$", ha="center", fontsize=9.5, color=C_GREEN)

    # Rejected forward-only match
    ax.plot([reject[0] - 0.15, reject[0] + 0.15], [reject[1] + 0.15, reject[1] - 0.15],
            color="#C62828", lw=1.8, zorder=4)
    ax.text(reject[0] + 0.55, reject[1] + 0.02, "no Rev", ha="left", va="center",
            fontsize=8, color="#C62828")

    # Mutual set label (bottom of R panel, clear of point labels)
    ax.add_patch(FancyBboxPatch(
        (3.05, 0.95), 2.45, 0.52, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=C_PURPLE, facecolor="#F3E5F5", zorder=1,
    ))
    ax.text(4.27, 1.21, r"$\mathcal{N}^{\mathrm{MNN}}(i)=\{j_1,j_2\}$",
            ha="center", va="center", fontsize=9.0, color=C_PURPLE, weight="bold")

    # Output: mean of y_ref over mutual neighbors
    box_x, box_w, box_h = 7.30, 1.05, 0.48
    y1_top, y2_top = 3.05, 2.05
    merge_x, merge_y = 8.55, 2.55
    out_top = 1.67  # top of final y_tilde box

    out_pairs = [
        ("j1", mutual["j1"], box_x, y1_top, r"$y^{ref}_{j_1}$"),
        ("j2", mutual["j2"], box_x, y2_top, r"$y^{ref}_{j_2}$"),
    ]
    for _key, src, x, y, lab in out_pairs:
        ax.add_patch(FancyBboxPatch(
            (x, y), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.3, edgecolor=C_ORANGE, facecolor="#FFF3E0", zorder=2,
        ))
        ax.text(x + box_w / 2, y + box_h / 2, lab, ha="center", va="center",
                fontsize=9, color="#333")
        _arrow(ax, (src[0] + 0.15, src[1]), (x, y + box_h / 2), C_ORANGE, lw=1.2)

    # Funnel: both y_ref boxes → merge node → y_tilde
    box_right = box_x + box_w
    box1_mid_y = y1_top + box_h / 2
    box2_mid_y = y2_top + box_h / 2
    _arrow(ax, (box_right, box1_mid_y), (merge_x, merge_y + 0.08), C_GREEN, lw=1.5)
    _arrow(ax, (box_right, box2_mid_y), (merge_x, merge_y - 0.08), C_GREEN, lw=1.5)
    _cell(ax, (merge_x, merge_y), "#FFFFFF", C_GREEN, r=0.07, lw=1.4, zorder=3)
    ax.text(merge_x + 0.38, merge_y, r"$\mathrm{mean}$", ha="left", va="center",
            fontsize=10, color=C_GREEN, weight="bold")
    _arrow(ax, (merge_x, merge_y - 0.10), (8.32, out_top), C_GREEN, lw=1.6)

    ax.add_patch(FancyBboxPatch(
        (7.55, 1.05), 1.55, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.8, edgecolor=C_GREEN, facecolor="#E8F5E9", zorder=2,
    ))
    ax.text(8.32, 1.36, r"$\tilde{y}_i$", ha="center", va="center", fontsize=12,
            color=C_GREEN, weight="bold")

    # Bottom pipeline strip
    steps = [
        (0.35, r"L2 norm"),
        (2.05, r"Fwd"),
        (3.85, r"Rev"),
        (5.65, r"Mutual"),
        (7.45, r"mean"),
    ]
    for idx, (x, lab) in enumerate(steps):
        ax.add_patch(FancyBboxPatch(
            (x, 0.08), 1.35, 0.42, boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.0, edgecolor="#888888", facecolor="#FFFFFF", zorder=1,
        ))
        ax.text(x + 0.675, 0.29, lab, ha="center", va="center", fontsize=8.5, color="#444")
        if idx < len(steps) - 1:
            _arrow(ax, (x + 1.35, 0.29), (steps[idx + 1][0], 0.29), "#888888", lw=1.0)

    out_path = os.path.join(OUT_DIR, "19_mnn_filter_algorithm.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


def plot_strict_mnn_two_step_bridge():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    def box(x, y, t, fc, ec):
        ax.add_patch(FancyBboxPatch(
            (x, y), 0.85, 0.55, boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2, edgecolor=ec, facecolor=fc,
        ))
        ax.text(x + 0.425, y + 0.275, t, ha="center", va="center", fontsize=9)

    box(0.3, 2.35, r"$X_1$", "#DCE6F7", C_BLUE)
    box(1.3, 2.35, r"$X_2$", "#FFF3E0", C_ORANGE)
    box(0.3, 1.55, r"$Y_B^2$", "#FFF3E0", C_ORANGE)
    box(2.5, 1.55, r"$\tilde{Y}_B^1$", "#E8F5E9", C_GREEN)

    s1 = FancyBboxPatch(
        (2.2, 2.05), 2.4, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=C_BLUE, facecolor="#FFFFFF",
    )
    ax.add_patch(s1)
    ax.text(3.4, 2.78, "Step 1", ha="center", fontsize=8, weight="bold", color=C_BLUE)
    ax.text(3.4, 2.52, r"MNN$(X_1,X_2)$", ha="center", fontsize=9, color=C_BLUE)
    _arrow(ax, (0.72, 2.62), (2.2, 2.55), C_BLUE)
    _arrow(ax, (1.72, 2.62), (2.2, 2.55), C_ORANGE)
    _arrow(ax, (1.72, 2.35), (1.72, 2.55), C_ORANGE)
    _arrow(ax, (3.4, 2.05), (2.92, 1.82), C_GREEN)

    box(5.0, 2.35, r"$Y_B^2$", "#FFF3E0", C_ORANGE)
    box(6.0, 2.35, r"$\tilde{Y}_B^1$", "#E8F5E9", C_GREEN)
    box(5.0, 1.55, r"$Y_A^1$", "#E8F5E9", C_GREEN)
    box(7.0, 1.55, r"$\tilde{Y}_A^2$", "#E8F5E9", C_GREEN)

    s2 = FancyBboxPatch(
        (5.5, 1.95), 2.0, 0.85, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=C_ORANGE, facecolor="#FFFFFF",
    )
    ax.add_patch(s2)
    ax.text(6.5, 2.58, "Step 2", ha="center", fontsize=8, weight="bold", color=C_ORANGE)
    ax.text(6.5, 2.32, r"MNN$(Y_B^2,\tilde{Y}_B^1)$", ha="center", fontsize=8.5, color=C_ORANGE)
    _arrow(ax, (5.42, 2.62), (5.5, 2.45), C_ORANGE)
    _arrow(ax, (6.42, 2.62), (6.5, 2.45), C_GREEN)
    _arrow(ax, (5.42, 1.82), (6.5, 1.95), C_GREEN)
    _arrow(ax, (6.5, 1.95), (7.0, 1.82), C_GREEN)

    out_path = os.path.join(OUT_DIR, "20_strict_mnn_two_step_bridge.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    plot_mnn_filter_algorithm()
    plot_strict_mnn_two_step_bridge()
