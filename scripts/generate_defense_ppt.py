"""
Generate a 58-page expanded defense presentation PPTX for SpatialEx/SpatialEx+.
- Unified academic style: dark-blue primary + orange accents
- Title underlines, section accent bars, card backgrounds, highlight blocks
- Chapter divider slides with large white title cards + subtitles
- Renders LaTeX formulas and matplotlib figures as SVG vector assets

Output: docs/SpatialEx_Defense.pptx
Existing output is backed up with a timestamp if docs/SpatialEx_Defense_27p.pptx already exists.
"""

import os
import sys
import hashlib
import tempfile
import shutil
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LXM_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
sys.path.insert(0, LXM_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from tools.ppt import (  # noqa: E402
    apply_pptx_svg_support,
    set_palette,
    set_formula_dir,
    set_slide_size,
    add_title_shape,
    add_section_title,
    add_bullet_box,
    add_two_column_bullets,
    add_highlight_block,
    add_caption,
    add_divider_line,
    add_table_slide,
    add_section_divider,
    add_formula as _add_formula_toolkit,
    render_latex,
    svg_aspect_ratio as _svg_aspect_ratio,
    _svg_viewport_px_size,
    SLIDE_WIDTH,
    SLIDE_HEIGHT,
    DARK_BLUE,
    ACCENT_ORANGE,
    LIGHT_GRAY,
    PALE_BLUE,
    DARK_GRAY,
    MEDIUM_GRAY,
    WHITE,
    BLACK,
    Inches,
    Pt,
    PP_ALIGN,
    MSO_SHAPE,
)

set_palette("defense", sync_module=sys.modules[__name__])
apply_pptx_svg_support()

# Configuration
OUTPUT_PATH = "docs/SpatialEx_Defense.pptx"
BACKUP_PATH = "docs/SpatialEx_Defense_27p.pptx"
IMG_DIR = "docs/image/fig3_diagnosis"
FORMULA_DIR = os.path.join(PROJECT_ROOT, "docs", "image", "formulas")
PAPER_IMG_DIR = "docs/image/from_paper"
OTHER_IMG_DIR = "docs/image"

set_formula_dir(FORMULA_DIR)

TITLE_FONT_SIZE = Pt(32)
SUBTITLE_FONT_SIZE = Pt(24)
BODY_FONT_SIZE = Pt(18)
SMALL_FONT_SIZE = Pt(15)
CAPTION_FONT_SIZE = Pt(14)

LINE_SPACING_TIGHT = 1.05
LINE_SPACING_NORMAL = 1.15
LINE_SPACING_LOOSE = 1.25

SLIDE_MARGIN_LEFT = Inches(0.6)
SLIDE_MARGIN_RIGHT = Inches(0.6)
SLIDE_MARGIN_TOP = Inches(0.35)
SLIDE_MARGIN_BOTTOM = Inches(0.35)


def backup_existing(output_path, backup_path):
    if not os.path.exists(output_path):
        return
    # Avoid overwriting an existing backup (e.g. the original 27-page version);
    # use a timestamped name if the target already exists.
    if os.path.exists(backup_path):
        from datetime import datetime
        stem, ext = os.path.splitext(backup_path)
        timestamped = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        shutil.copy(output_path, timestamped)
        print(f"Backed up existing PPT to: {timestamped}")
    else:
        shutil.copy(output_path, backup_path)
        print(f"Backed up existing PPT to: {backup_path}")


def add_formula(slide, latex_str, left=Inches(0.8), top=Inches(1.5),
                height=Inches(0.6), fontsize=20):
    """Defense scripts expect a picture handle only."""
    pic, _, _ = _add_formula_toolkit(
        slide, latex_str, left=left, top=top, height=height, fontsize=fontsize,
        max_width=None,
    )
    return pic



def get_image_pixel_size(img_path):
    """Return (width, height) in pixels for SVG or raster images."""
    if img_path.lower().endswith(".svg"):
        from xml.etree import ElementTree
        root = ElementTree.parse(img_path).getroot()
        w, h = _svg_viewport_px_size(root)
        return int(w), int(h)
    from PIL import Image
    with Image.open(img_path) as img:
        return img.size


def diag_fig(stem):
    """Path to fig3_diagnosis figure; prefer SVG, fall back to PNG."""
    svg = os.path.join(IMG_DIR, f"{stem}.svg")
    if os.path.exists(svg):
        return svg
    png = os.path.join(IMG_DIR, f"{stem}.png")
    return png if os.path.exists(png) else svg


def add_figure_on_slide(slide, fig_stem, left=Inches(0.35), top=Inches(1.15),
                        max_width=Inches(7.6), max_height=Inches(5.5)):
    """Place architecture figure; return (pic_w, pic_h) or None if missing."""
    img_path = diag_fig(fig_stem)
    if not os.path.exists(img_path):
        return None
    img_w, img_h = get_image_pixel_size(img_path)
    scale = min(max_width / Inches(img_w / 150), max_height / Inches(img_h / 150))
    pic_w = Inches(img_w / 150) * scale
    pic_h = Inches(img_h / 150) * scale
    slide.shapes.add_picture(img_path, left, top, width=pic_w, height=pic_h)
    return pic_w, pic_h


def add_architecture_slide(slide, title, fig_stem, sidebar_builder,
                           fig_max_width=Inches(7.5), sidebar_left=Inches(8.1)):
    """Architecture slide: figure left, explanatory content right."""
    add_title_shape(slide, title)
    placed = add_figure_on_slide(slide, fig_stem, max_width=fig_max_width)
    if placed is None:
        add_bullet_box(slide, [f"Figure not found: {fig_stem}"], top=Inches(2.0), font_size=16)
    sidebar_builder(slide, sidebar_left)


def add_formula(slide, latex_str, left=Inches(0.8), top=Inches(1.5),
                height=Inches(0.6), fontsize=20):
    """Render LaTeX to SVG and add as vector picture."""
    svg_path = render_latex(latex_str, fontsize=fontsize)
    aspect = _svg_aspect_ratio(svg_path)
    width = height * aspect
    pic = slide.shapes.add_picture(svg_path, left, top, width=width, height=height)
    return pic


def render_reproduction_summary_chart():
    """Generate a bar chart summarizing SpatialEx reproduction results (SVG)."""
    os.makedirs(OTHER_IMG_DIR, exist_ok=True)
    chart_path = os.path.join(OTHER_IMG_DIR, "reproduction_summary.svg")
    if os.path.exists(chart_path):
        return chart_path

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
    }):
        fig, ax = plt.subplots(figsize=(5.5, 4.8))

        categories = [
            "Fig.2 S1 PCC", "Fig.2 S2 PCC",
            "Fig.2 S1 SSIM", "Fig.2 S2 SSIM",
            "Fig.2 S1 CMD", "Fig.2 S2 CMD",
            "Fig.3 S1 PCC", "Fig.3 S2 PCC",
            "Fig.3 S1 SSIM", "Fig.3 S2 SSIM",
            "Fig.3 S1 CMD", "Fig.3 S2 CMD",
            "Fig.4 S1 PCC", "Fig.4 S2 PCC",
            "Fig.4 S1 SSIM", "Fig.4 S2 SSIM",
            "Fig.4 S1 CMD", "Fig.4 S2 CMD",
            "Fig.5 p→RNA PCC", "Fig.5 RNA→p PCC",
            "Fig.5 p→RNA SSIM", "Fig.5 RNA→p SSIM",
            "Fig.5 p→RNA CMD", "Fig.5 RNA→p CMD"
        ]
        values = [
            0.257, 0.273,
            0.419, 0.425,
            0.205, 0.207,
            0.275, 0.301,
            0.308, 0.332,
            0.334, 0.267,
            0.008, 0.013,
            0.439, 0.433,
            0.328, 0.383,
            0.196, -0.042,
            0.367, 0.187,
            0.222, 0.003
        ]
        colors = [
            DARK_BLUE, DARK_BLUE,
            ACCENT_ORANGE, ACCENT_ORANGE,
            MEDIUM_GRAY, MEDIUM_GRAY,
            DARK_BLUE, DARK_BLUE,
            ACCENT_ORANGE, ACCENT_ORANGE,
            MEDIUM_GRAY, MEDIUM_GRAY,
            DARK_BLUE, DARK_BLUE,
            ACCENT_ORANGE, ACCENT_ORANGE,
            MEDIUM_GRAY, MEDIUM_GRAY,
            DARK_BLUE, DARK_BLUE,
            ACCENT_ORANGE, ACCENT_ORANGE,
            MEDIUM_GRAY, MEDIUM_GRAY
        ]

        def rgbcolor_to_tuple(c):
            r, g, b = c
            return (r / 255.0, g / 255.0, b / 255.0)

        y_pos = range(len(categories))
        bars = ax.barh(y_pos, values, color=[rgbcolor_to_tuple(c) for c in colors], height=0.6)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(categories, fontsize=9)
        ax.set_xlim(-0.1, 0.55)
        ax.set_xlabel("Metric value", fontsize=11)
        ax.set_title("SpatialEx/SpatialEx+ Reproduction Results", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(x=0, color="black", linewidth=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            offset = 0.01 if val >= 0 else -0.03
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=rgbcolor_to_tuple(DARK_BLUE), label="PCC"),
            Patch(facecolor=rgbcolor_to_tuple(ACCENT_ORANGE), label="SSIM"),
            Patch(facecolor=rgbcolor_to_tuple(MEDIUM_GRAY), label="CMD")
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig(chart_path, format="svg", bbox_inches="tight", pad_inches=0.15,
                    transparent=False, facecolor="white")
        plt.close(fig)
    return chart_path


def render_mnn_bridge_flowchart():
    """Generate a simple top-down flowchart for Strict MNN two-step bridge."""
    from matplotlib.patches import FancyBboxPatch

    os.makedirs(OTHER_IMG_DIR, exist_ok=True)
    chart_path = os.path.join(OTHER_IMG_DIR, "mnn_bridge_flowchart.svg")
    if os.path.exists(chart_path):
        return chart_path

    def rgb(c):
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)

    dark_blue = rgb(DARK_BLUE)
    accent_orange = rgb(ACCENT_ORANGE)
    medium_gray = rgb(MEDIUM_GRAY)

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK JP", "DejaVu Sans"],
    }):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

        def box(x, y, w, h, text, facecolor, edgecolor, fontsize=11, text_color="black"):
            rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
                                   facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fontsize, color=text_color, weight="bold", wrap=True)

        def arrow(x1, y1, x2, y2, color="black", style="->", lw=2):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                       connectionstyle="arc3,rad=0"))

        # Title
        ax.text(5, 5.75, "Strict MNN 两步桥接", ha="center", va="center",
                fontsize=18, weight="bold", color=dark_blue)
        ax.text(5, 5.4, r"$\tilde{Y}_B^1$ 与 $\tilde{Y}_A^2$ 均由 build_mnn_pseudo 构造，全程不用 held-out",
                ha="center", va="center", fontsize=11, color=medium_gray)

        # Input slices
        box(0.8, 4.35, 2.8, 0.7, "Slice 1 (Rep1)\nH&E X1 + Panel A Y_A^1", "#EFF2FF", dark_blue, 10)
        box(6.4, 4.35, 2.8, 0.7, "Slice 2 (Rep2)\nH&E X2 + Panel B Y_B^2", "#FFF4E6", accent_orange, 10)

        # Held-out labels
        ax.text(2.2, 4.15, "held-out: Y_B^1", ha="center", va="center", fontsize=9, color="red")
        ax.text(7.8, 4.15, "held-out: Y_A^2", ha="center", va="center", fontsize=9, color="red")

        # Step 1
        box(2.6, 3.0, 5.0, 0.8, "Step 1 · H&E Bridge\nMNN(X1, X2, Y_B^2)  →  Ỹ_B^1", "#DBE4FF", dark_blue, 11)
        arrow(2.2, 4.35, 3.6, 3.8, color=dark_blue)
        arrow(7.8, 4.35, 6.6, 3.8, color=dark_blue)

        # Step 2
        box(2.6, 1.6, 5.0, 0.8, "Step 2 · Panel B Bridge\nMNN(Y_B^2, Ỹ_B^1, Y_A^1)  →  Ỹ_A^2", "#FFE5CC", accent_orange, 11)
        arrow(5.0, 3.0, 5.0, 2.4, color=accent_orange, lw=2.5)
        # Side input for Step 2
        ax.text(8.0, 2.3, "Y_B^2 也作为\nquery 进入 Step 2", ha="left", va="center", fontsize=9, color=accent_orange)
        arrow(7.8, 4.15, 7.5, 2.45, color=accent_orange, style="->")

        # MLP training
        box(1.8, 0.15, 6.6, 0.8, "MLP 训练：输入 measured panel  →  预测 missing panel\n监督 = MSE(预测, Ỹ_B^1 / Ỹ_A^2)",
            "#1A237E", "#1A237E", 11, text_color="white")
        arrow(3.6, 1.6, 3.0, 0.95, color="black")
        arrow(6.4, 1.6, 7.0, 0.95, color="black")

        # Legend
        ax.text(0.3, 0.15, "MNN: mutual nearest neighbor\n(k=5, mnn_k=20)", ha="left", va="bottom",
                fontsize=9, color=medium_gray)

        plt.tight_layout()
        fig.savefig(chart_path, format="svg", bbox_inches="tight", pad_inches=0.1,
                    transparent=False, facecolor="white")
        plt.close(fig)
    return chart_path


def render_top_marker_gains_chart(top_n=10):
    """Generate horizontal bar chart: MLP+MNN vs HGNN+Cycle top gene gains."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "fig3"))
    from generate_top_marker_gains_chart import plot_top_marker_gains, CSV_PATH

    if not os.path.exists(CSV_PATH):
        old_path = os.path.join(IMG_DIR, "09_top_marker_gains.svg")
        return old_path if os.path.exists(old_path) else None

    chart_path = os.path.join(OTHER_IMG_DIR, "top_marker_gains_hgnn_vs_mnn.svg")
    return plot_top_marker_gains(top_n=top_n, out_path=chart_path)


def render_branch_decomposition_design():
    """Generate a clean flowchart for Branch Decomposition experiment design."""
    from matplotlib.patches import FancyBboxPatch

    os.makedirs(OTHER_IMG_DIR, exist_ok=True)
    chart_path = os.path.join(OTHER_IMG_DIR, "branch_decomposition_design.svg")
    if os.path.exists(chart_path):
        return chart_path

    def rgb(c):
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)

    dark_blue = rgb(DARK_BLUE)
    accent_orange = rgb(ACCENT_ORANGE)
    medium_gray = rgb(MEDIUM_GRAY)
    green = (0.30, 0.62, 0.33)
    purple = (0.45, 0.25, 0.60)

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK JP", "DejaVu Sans"],
    }):
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        ax.set_xlim(0, 12.5)
        ax.set_ylim(0, 5.2)
        ax.axis("off")

        def box(x, y, w, h, text, facecolor, edgecolor, fontsize=10, text_color="black"):
            rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                   facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fontsize, color=text_color, weight="bold", wrap=True)

        def arrow(x1, y1, x2, y2, color="black", style="->", lw=2):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                       connectionstyle="arc3,rad=0"))

        # Title
        ax.text(6.25, 4.95, "Branch Decomposition：拆解两条预测路径", ha="center", va="center",
                fontsize=16, weight="bold", color=dark_blue)

        # Input slices
        box(0.5, 3.75, 2.4, 0.75, "Slice 1\nX1 (H&E) + Y_A^1", "#EFF2FF", dark_blue, 10)
        box(9.6, 3.75, 2.4, 0.75, "Slice 2\nX2 (H&E) + Y_B^2", "#FFF4E6", accent_orange, 10)
        ax.text(1.7, 4.55, "held-out: Y_B^1", ha="center", va="center", fontsize=9, color="red")
        ax.text(10.8, 4.55, "held-out: Y_A^2", ha="center", va="center", fontsize=9, color="red")

        # Branch 1: H&E
        ax.text(0.15, 3.05, "Branch 1\nH&E only", ha="left", va="center", fontsize=11,
                weight="bold", color=dark_blue)
        box(1.6, 2.7, 1.3, 0.65, "X1", "#DBE4FF", dark_blue, 10)
        box(3.15, 2.7, 1.7, 0.65, "H&E encoder\n(MLP)", "#DBE4FF", dark_blue, 10)
        box(5.15, 2.7, 1.3, 0.65, "Ỹ_B^1", "#FFF4E6", accent_orange, 10)
        arrow(2.9, 3.02, 3.15, 3.02, color=dark_blue)
        arrow(4.85, 3.02, 5.15, 3.02, color=dark_blue)

        box(7.4, 2.7, 1.3, 0.65, "X2", "#DBE4FF", dark_blue, 10)
        box(8.95, 2.7, 1.7, 0.65, "H&E encoder\n(MLP)", "#DBE4FF", dark_blue, 10)
        box(10.95, 2.7, 1.3, 0.65, "Ỹ_A^2", "#FFF4E6", accent_orange, 10)
        arrow(8.7, 3.02, 8.95, 3.02, color=dark_blue)
        arrow(10.65, 3.02, 10.95, 3.02, color=dark_blue)

        # Branch 2: Panel
        ax.text(0.15, 1.75, "Branch 2\nPanel only", ha="left", va="center", fontsize=11,
                weight="bold", color=green)
        box(1.6, 1.4, 1.3, 0.65, "Y_A^1", "#E8F5E9", green, 10)
        box(3.15, 1.4, 1.7, 0.65, "Panel translator\n(MLP)", "#E8F5E9", green, 10)
        box(5.15, 1.4, 1.3, 0.65, "Ỹ_B^1", "#FFF4E6", accent_orange, 10)
        arrow(2.9, 1.72, 3.15, 1.72, color=green)
        arrow(4.85, 1.72, 5.15, 1.72, color=green)

        box(7.4, 1.4, 1.3, 0.65, "Y_B^2", "#E8F5E9", green, 10)
        box(8.95, 1.4, 1.7, 0.65, "Panel translator\n(MLP)", "#E8F5E9", green, 10)
        box(10.95, 1.4, 1.3, 0.65, "Ỹ_A^2", "#FFF4E6", accent_orange, 10)
        arrow(8.7, 1.72, 8.95, 1.72, color=green)
        arrow(10.65, 1.72, 10.95, 1.72, color=green)

        # Late fusion
        box(3.6, 0.25, 5.3, 0.65, "Late Fusion：0.5 average / reliability-weighted", "#F3E5F5", purple, 10)
        arrow(4.5, 2.7, 4.5, 0.9, color=purple, lw=1.5)
        arrow(5.6, 1.4, 5.4, 0.9, color=purple, lw=1.5)
        arrow(8.7, 2.7, 7.8, 0.9, color=purple, lw=1.5)
        arrow(9.7, 1.4, 8.0, 0.9, color=purple, lw=1.5)

        # Legend
        ax.text(0.3, 0.15, "评估：PCC(Ỹ_B^1, Y_B^1) 与 PCC(Ỹ_A^2, Y_A^2)",
                ha="left", va="bottom", fontsize=9, color=medium_gray)

        plt.tight_layout()
        fig.savefig(chart_path, format="svg", bbox_inches="tight", pad_inches=0.12,
                    transparent=False, facecolor="white")
        plt.close(fig)
    return chart_path


def render_branch_decomposition_results():
    """Generate a clean grouped bar chart for Branch Decomposition results."""
    os.makedirs(OTHER_IMG_DIR, exist_ok=True)
    chart_path = os.path.join(OTHER_IMG_DIR, "branch_decomposition_results.svg")
    if os.path.exists(chart_path):
        return chart_path

    def rgb(c):
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)

    variants = ["H&E\nbranch", "Panel\nbranch", "0.5 avg\n(calibrated)", "Reliability\nweighted"]
    s1 = [-0.001, 0.016, 0.011, -0.001]
    s2 = [0.010, 0.248, 0.177, 0.010]

    x = np.arange(len(variants))
    width = 0.32

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK JP", "DejaVu Sans"],
    }):
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        bars1 = ax.bar(x - width/2, s1, width, label="Slice1 PanelB PCC",
                       color=rgb(DARK_BLUE))
        bars2 = ax.bar(x + width/2, s2, width, label="Slice2 PanelA PCC",
                       color=rgb(ACCENT_ORANGE))

        ax.set_ylabel("PCC", fontsize=12)
        ax.set_title("Branch Decomposition：H&E vs Panel vs Fusion", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=10)
        ax.legend(loc="upper right", fontsize=10)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                offset = 0.005 if height >= 0 else -0.015
                ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                        f"{height:.3f}", ha="center", va="bottom" if height >= 0 else "top",
                        fontsize=8)

        plt.tight_layout()
        fig.savefig(chart_path, format="svg", bbox_inches="tight", pad_inches=0.15,
                    transparent=False, facecolor="white")
        plt.close(fig)
    return chart_path


def add_image_slide(slide, img_path, title_text, max_height=Inches(5.5)):
    """
    Add an image slide with title, centered figure, and caption area.
    """
    add_title_shape(
        slide, title_text,
        top=Inches(0.3),
        height=Inches(0.8),
        font_size=28,
        bold=True,
        color=DARK_BLUE
    )

    if not os.path.exists(img_path):
        add_bullet_box(
            slide, [f"Image not found: {img_path}"],
            top=Inches(2.0),
            font_size=18
        )
        return

    img_w, img_h = get_image_pixel_size(img_path)

    max_w = Inches(12.0)
    scale = min(max_w / Inches(img_w / 150), max_height / Inches(img_h / 150))

    new_w = Inches(img_w / 150) * scale
    new_h = Inches(img_h / 150) * scale

    left = (SLIDE_WIDTH - new_w) / 2
    top = Inches(1.3) + (max_height - new_h) / 2

    slide.shapes.add_picture(img_path, left, top, width=new_w, height=new_h)


def create_presentation():
    # Backup existing PPT
    backup_existing(OUTPUT_PATH, BACKUP_PATH)

    prs = Presentation()
    set_slide_size(prs)
    blank_layout = prs.slide_layouts[6]

    # ================================================================
    # 1. Title
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                  SLIDE_WIDTH, Inches(1.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    add_title_shape(slide, "SpatialEx/SpatialEx+ 复现与改进",
                    top=Inches(1.5), height=Inches(1.0),
                    font_size=40, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER,
                    add_underline=False)
    add_title_shape(slide, "Fig.3 Panel Diagonal Integration 上的监督信号探索",
                    top=Inches(2.6), height=Inches(0.7),
                    font_size=24, bold=False, color=ACCENT_ORANGE, align=PP_ALIGN.CENTER,
                    add_underline=False)
    add_bullet_box(slide, [
        "论文：High-parameter spatial multi-omics through histology-anchored integration (Nature Methods 2025)",
        "答辩日期：2025 年 6 月 21 日（线上，约 40 分钟）",
        "答辩人：李熹鸣",
        "项目仓库：https://github.com/965120527lxm-maker/SpatialEvo",
        "核心发现：MLP + Strict MNN 显著优于官方 HGNN/GT + Cycle"
    ], top=Inches(3.7), left=Inches(1.8), width=Inches(9.8), height=Inches(2.5),
       font_size=18, color=BLACK)

    # ================================================================
    # 2. Outline
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "答辩提纲")
    add_bullet_box(slide, [
        "1. 背景与任务：Spatial Diagonal Integration & Fig.1–5 & 数学符号",
        "2. 环境搭建与复现：Fig.2–5 复现与官方代码 6 个 bug 修复",
        "3. 改进一：Graph Transformer 替代 HGNN（架构 + 结果）",
        "4. 诊断：Branch Decomposition 与信号来源分析",
        "5. 改进二：Strict MNN 伪标签替代 Cycle（架构 + 结果）",
        "6. 讨论：为什么 MLP + MNN 反而最强？",
        "7. 结论、局限与展望"
    ], top=Inches(1.4), font_size=21)

    # ================================================================
    # 3. Section divider - Background
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_section_divider(slide, "一. 背景与任务",
                        "展示对论文 Fig.1–5 的理解，并用数学符号把 Fig.3 任务严格化")

    # ================================================================
    # 4. Background
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "背景：空间组学的高参数困境")
    add_section_title(slide, "技术困境", top=Inches(1.3))
    add_bullet_box(slide, [
        "分辨率、通量、panel 大小三者不可兼得：",
        "  • 高分辨率技术（Xenium、Merfish、CosMx）通常只能测几十到几百个基因",
        "  • 大 panel 技术（Visium）分辨率低，且仍是转录组子集"
    ], top=Inches(1.85), height=Inches(1.7), font_size=18)
    add_section_title(slide, "解决思路", top=Inches(3.45))
    add_bullet_box(slide, [
        "相邻切片测互补 panel / 互补组学，再计算整合：",
        "  • 切片 1 测 panel A，切片 2 测 panel B",
        "  • 切片 1 测转录组，切片 2 测蛋白组"
    ], top=Inches(4.0), height=Inches(1.4), font_size=18)
    add_highlight_block(
        slide,
        "关键：Spatial Diagonal Integration —— 两个切片没有共测 omics 特征，"
        "必须借助空间位置或形态先验做桥接。",
        top=Inches(5.6), height=Inches(0.7), font_size=16
    )
    img_path = os.path.join(OTHER_IMG_DIR, "fig1_bottom_three_scenarios.png")
    if os.path.exists(img_path):
        from PIL import Image
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        max_w = Inches(12.0)
        max_h = Inches(1.5)
        scale = min(max_w / Inches(img_w / 150), max_h / Inches(img_h / 150))
        new_w = Inches(img_w / 150) * scale
        new_h = Inches(img_h / 150) * scale
        left = (SLIDE_WIDTH - new_w) / 2
        top = Inches(5.55)
        slide.shapes.add_picture(img_path, left, top, width=new_w, height=new_h)

    # ================================================================
    # 5. Paper Fig 1
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    img_path = os.path.join(OTHER_IMG_DIR, "fig1_top_framework.png")
    add_image_slide(slide, img_path, "论文原图 Fig.1：SpatialEx/SpatialEx+ 方法框架")
    add_caption(slide, "上半部分：SpatialEx 的 H&E foundation model、HGNN encoder、DGI 对比学习；下半部分：SpatialEx+ 的数据准备与 Cycle Module。Fig.3 的 panel diagonal 是论文方法的重要应用场景。", top=Inches(6.5), font_size=15)

    # ================================================================
    # 6. Two types
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "两种 Diagonal Integration")
    add_two_column_bullets(slide,
        "Panel Diagonal Integration",
        [
            "同一切片技术（如都是 Xenium）",
            "不同基因 panel（panel A vs panel B）",
            "Fig.3 任务",
            "本工作聚焦于此",
            "",
            "相邻切片各测互补基因集，",
            "目标是互相预测缺失 panel。"
        ],
        "Omics Diagonal Integration",
        [
            "不同组学类型",
            "如转录组 + 蛋白、转录组 + 代谢",
            "Fig.5 任务",
            "已初步复现",
            "",
            "相邻切片测不同模态，",
            "目标是跨模态预测。"
        ],
        top=Inches(1.4), height=Inches(4.8), font_size=18)

    # ================================================================
    # 7. Paper Fig 2 (mathematical)
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "论文原图 Fig.2：H&E → 全转录组任务（数学化）")
    img_path = os.path.join(PAPER_IMG_DIR, "Fig2_41592_2025_2926_Fig2_HTML.png")
    from PIL import Image
    with Image.open(img_path) as img:
        img_w, img_h = img.size
    max_h = Inches(4.8)
    max_w = Inches(6.0)
    scale = min(max_w / Inches(img_w / 150), max_h / Inches(img_h / 150))
    pic_w = Inches(img_w / 150) * scale
    pic_h = Inches(img_h / 150) * scale
    left = Inches(0.5)
    top = Inches(1.3)
    slide.shapes.add_picture(img_path, left, top, width=pic_w, height=pic_h)
    add_section_title(slide, "任务设定（同一组织切片内）", top=Inches(1.3),
                     left=Inches(7.0), width=Inches(5.8))
    add_bullet_box(slide, [
        "H&E 嵌入：X ∈ R^(N×d)",
        "全转录组表达：Y ∈ R^(N×313)",
        "目标：学习映射 f: X → Ŷ，使得 Ŷ ≈ Y",
        "与 Fig.3 区别：同一切片同时观测 X 与 Y，属于有监督回归"
    ], top=Inches(1.95), left=Inches(7.0), width=Inches(5.8), height=Inches(2.4),
       font_size=17, emphasize_words=["f: X → Ŷ", "Ŷ ≈ Y"])
    add_formula(slide, r"f: X \to \hat{Y}, \quad \hat{Y} \approx Y",
                left=Inches(7.0), top=Inches(4.5), height=Inches(0.5), fontsize=18)
    add_caption(slide, "Fig.2 是 pure H&E → omics 任务：输入仅为 H&E，输出为全转录组（313 genes），不涉及 panel split。本工作补充 Fig.2 baseline 作为理解 Fig.3 的对照。", top=Inches(6.5), font_size=15)

    # ================================================================
    # 8. Paper Fig 3 (mathematical)
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "论文原图 Fig.3：Panel Diagonal Integration（数学化）")
    img_path = os.path.join(PAPER_IMG_DIR, "Fig3_41592_2025_2926_Fig3_HTML.png")
    with Image.open(img_path) as img:
        img_w, img_h = img.size
    max_h = Inches(4.8)
    max_w = Inches(5.5)
    scale = min(max_w / Inches(img_w / 150), max_h / Inches(img_h / 150))
    pic_w = Inches(img_w / 150) * scale
    pic_h = Inches(img_h / 150) * scale
    left = Inches(0.5)
    top = Inches(1.3)
    slide.shapes.add_picture(img_path, left, top, width=pic_w, height=pic_h)
    add_section_title(slide, "相邻切片，互补 panel", top=Inches(1.3),
                     left=Inches(6.8), width=Inches(6.0))
    add_bullet_box(slide, [
        "Slice 1：已测 panel A YA1，缺失 panel B YB1",
        "Slice 2：已测 panel B YB2，缺失 panel A YA2",
        "目标：ŶB1 ≈ YB1，ŶA2 ≈ YA2",
        "关键约束：YB1, YA2 ∉ D_train",
        "官方方法：SpatialEx+ Cycle 约束"
    ], top=Inches(1.95), left=Inches(6.8), width=Inches(6.0), height=Inches(2.8),
       font_size=17, emphasize_words=["YB1, YA2 ∉ D_train"])
    add_formula(slide, r"\{X_1, Y_A^1\} \to \hat{Y}_B^1",
                left=Inches(6.8), top=Inches(4.85), height=Inches(0.45), fontsize=18)
    add_formula(slide, r"\{X_2, Y_B^2\} \to \hat{Y}_A^2",
                left=Inches(6.8), top=Inches(5.35), height=Inches(0.45), fontsize=18)
    add_caption(slide, "原文使用两张相邻乳腺癌切片（Rep1 / Rep2），各测互补 panel。官方方法通过 Cycle 约束进行 cross-panel 预测；本工作对该设定进行严格复现并重新评估。", top=Inches(6.45), font_size=15)

    # ================================================================
    # 9. Paper Fig 4 (mathematical)
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "论文原图 Fig.4：百万细胞大切片 Panel Diagonal（可扩展性）")
    img_path = os.path.join(PAPER_IMG_DIR, "Fig4_41592_2025_2926_Fig4_HTML.png")
    with Image.open(img_path) as img:
        img_w, img_h = img.size
    max_h = Inches(4.8)
    max_w = Inches(5.5)
    scale = min(max_w / Inches(img_w / 150), max_h / Inches(img_h / 150))
    pic_w = Inches(img_w / 150) * scale
    pic_h = Inches(img_h / 150) * scale
    left = Inches(0.5)
    top = Inches(1.3)
    slide.shapes.add_picture(img_path, left, top, width=pic_w, height=pic_h)
    add_section_title(slide, "Fig.3 在百万细胞尺度上的扩展", top=Inches(1.3),
                     left=Inches(6.8), width=Inches(6.0))
    add_bullet_box(slide, [
        "10x Xenium 大切片：两片合计约 1.8M 细胞",
        "把 280-gene panel 平分为 A / B，两片互补测量",
        "Slice 1：H&E X1 + 已测 panel A YA1，缺失 panel B YB1",
        "Slice 2：H&E X2 + 已测 panel B YB2，缺失 panel A YA2",
        "目标：ŶB1 ≈ YB1，ŶA2 ≈ YA2"
    ], top=Inches(1.95), left=Inches(6.8), width=Inches(6.0), height=Inches(2.8),
       font_size=17, emphasize_words=["1.8M 细胞"])
    add_formula(slide, r"\{X_1, Y_A^1\} \to \hat{Y}_B^1",
                left=Inches(6.8), top=Inches(4.85), height=Inches(0.45), fontsize=18)
    add_formula(slide, r"\{X_2, Y_B^2\} \to \hat{Y}_A^2",
                left=Inches(6.8), top=Inches(5.35), height=Inches(0.45), fontsize=18)
    add_caption(slide, "Fig.4 验证 SpatialEx+ 在 ~900k 细胞大切片上的可扩展性。我们训练完成；评估时遇到 coo_matrix subscript bug，metrics 来自已有输出。", top=Inches(6.45), font_size=15)

    # ================================================================
    # 10. Paper Fig 5 (mathematical)
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "论文原图 Fig.5：Omics Diagonal Integration（跨组学）")
    img_path = os.path.join(PAPER_IMG_DIR, "Fig5_41592_2025_2926_Fig5_HTML.png")
    with Image.open(img_path) as img:
        img_w, img_h = img.size
    max_h = Inches(4.8)
    max_w = Inches(5.5)
    scale = min(max_w / Inches(img_w / 150), max_h / Inches(img_h / 150))
    pic_w = Inches(img_w / 150) * scale
    pic_h = Inches(img_h / 150) * scale
    left = Inches(0.5)
    top = Inches(1.3)
    slide.shapes.add_picture(img_path, left, top, width=pic_w, height=pic_h)
    add_section_title(slide, "相邻切片，跨组学", top=Inches(1.3),
                     left=Inches(6.8), width=Inches(6.0))
    add_bullet_box(slide, [
        "Slice 1：已测组学 A YA1（如 RNA），缺失组学 B YB1（如蛋白）",
        "Slice 2：已测组学 B YB2（如蛋白），缺失组学 A YA2（如 RNA）",
        "目标：ŶB1 ≈ YB1，ŶA2 ≈ YA2",
        "关键约束：YB1, YA2 ∉ D_train",
        "Fig.3 是 panel-level diagonal；Fig.5 是 omics-level diagonal"
    ], top=Inches(1.95), left=Inches(6.8), width=Inches(6.0), height=Inches(2.8),
       font_size=17, emphasize_words=["YB1, YA2 ∉ D_train"])
    add_formula(slide, r"\{X_1, Y_A^1\} \to \hat{Y}_B^1",
                left=Inches(6.8), top=Inches(4.85), height=Inches(0.45), fontsize=18)
    add_formula(slide, r"\{X_2, Y_B^2\} \to \hat{Y}_A^2",
                left=Inches(6.8), top=Inches(5.35), height=Inches(0.45), fontsize=18)
    add_caption(slide, "Fig.5a,b 展示 transcriptomics ↔ proteomics；Fig.5c–f 展示 transcriptomics ↔ metabolomics。我们已初步复现 transcriptomics ↔ proteomics，protein→RNA 方向 PCC≈0.20；RNA→protein 仅 2 个蛋白 marker，任务过难。Strict MNN 思想可进一步推广。", top=Inches(6.45), font_size=15)

    # ================================================================
    # 11. Strict protocol + metrics
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Fig.3 Strict 协议与评价指标")
    add_section_title(slide, "Strict 协议", top=Inches(1.25))
    add_bullet_box(slide, [
        "在 Fig.3 数学化基础上，进一步明确训练约束：",
        "  • Slice 1：{X1, YA1} → ŶB1",
        "  • Slice 2：{X2, YB2} → ŶA2",
        "  • held-out panel YB1、YA2 不可用于训练"
    ], top=Inches(1.75), height=Inches(1.7), font_size=17,
       emphasize_words=["held-out panel"])
    add_section_title(slide, "可用信号来源", top=Inches(3.55))
    add_bullet_box(slide, [
        "H&E 形态先验：从组织学图像推断基因表达",
        "跨切片 pseudo-label：用相邻切片 measured panel 构造伪标签",
        "自监督约束：Cycle Consistency、DGI 对比学习"
    ], top=Inches(4.05), height=Inches(1.5), font_size=17)
    add_section_title(slide, "评价指标", top=Inches(5.55))
    add_bullet_box(slide, [
        "gene-level PCC（主指标）",
        "SSIM（空间结构保真）",
        "CMD（中心矩差异）"
    ], top=Inches(6.05), left=Inches(7.5), width=Inches(5.5), height=Inches(1.0), font_size=17)
    add_formula(slide, r"Y_B^1, Y_A^2 \notin \mathcal{D}_{\text{train}}",
                left=Inches(7.5), top=Inches(1.75), height=Inches(0.45), fontsize=18)
    add_formula(slide, r"\mathcal{L}_{\text{PCC}} = \frac{1}{G} \sum_{g=1}^{G} \rho(\hat{y}_g, y_g)",
                left=Inches(7.2), top=Inches(4.55), height=Inches(0.5), fontsize=16)

    # ================================================================
    # 12. fig3b
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    img_path = os.path.join(OTHER_IMG_DIR, "fig3b.png")
    add_image_slide(slide, img_path, "复现图：Fig.3 Strict 协议数据划分")
    add_caption(slide, "左侧 Slice1：X1 + YA1 → predict YB1；右侧 Slice2：X2 + YB2 → predict YA2。红色框表示 held-out、训练不可用的 panel。", top=Inches(6.6), font_size=15)

    # ================================================================
    # 13. Section divider - Reproduction
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_section_divider(slide, "二. 环境搭建与复现",
                        "先把官方代码跑通，并补充 Fig.2 baseline；GT/MNN 改进放到后面章节")

    # ================================================================
    # 14. Environment and conda config
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "个人环境与 Conda 配置")
    add_section_title(slide, "硬件环境", top=Inches(1.25))
    add_bullet_box(slide, [
        "GPU：RTX 5090 / A100（大图模型训练时使用）",
        "显存峰值：HGNN-512 约 20–24 GB",
        "GT Cross-Attention 因 $O(N^2)$ 复杂度超出显存，实际退化为轻量 MLP"
    ], top=Inches(1.75), height=Inches(1.35), font_size=17)
    add_section_title(slide, "软件环境", top=Inches(3.1))
    add_bullet_box(slide, [
        "Python 3.10 + PyTorch 2.x + CUDA 11.8",
        "关键包：DGL（图神经网络）、anndata（单细胞数据）、UNI（病理图像预训练）、cellpose",
        "Conda 环境：requirements.txt 已锁定主要依赖；DGL 需按 CUDA 版本单独安装"
    ], top=Inches(3.6), height=Inches(1.4), font_size=17)
    add_section_title(slide, "数据来源", top=Inches(5.0))
    add_bullet_box(slide, [
        "论文官方预处理好的 Xenium Human Breast Cancer .h5ad",
        "Rep1 ~164k 细胞 / Rep2 ~111k 细胞，各 313 个基因"
    ], top=Inches(5.5), height=Inches(0.9), font_size=17)

    # ================================================================
    # 15. Dataset and reproduced tasks
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "数据集与已复现任务")
    add_section_title(slide, "数据集内容", top=Inches(1.25))
    add_bullet_box(slide, [
        "Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad",
        "  • ~164k 细胞，313 基因，UNI H&E 嵌入，空间坐标",
        "Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad",
        "  • ~111k 细胞，313 基因，UNI H&E 嵌入，空间坐标",
        "data/panel_split_official.csv：官方 150/163 gene split",
        "10x_breast_cancer/human_breast_cancer.h5ad：NicheTrans 蛋白数据（Fig.5 Omics Diagonal）",
        "xenium_big/：Fig.4 百万细胞大切片数据"
    ], top=Inches(1.8), height=Inches(2.5), font_size=17)
    add_section_title(slide, "已复现任务", top=Inches(4.4))
    add_bullet_box(slide, [
        "Fig.2：SpatialEx 单切片 H&E → 全转录组",
        "Fig.3：SpatialEx+ Panel Diagonal（官方 150/163 split，Cycle baseline）",
        "Fig.4：SpatialEx+ 百万细胞大切片 Panel Diagonal",
        "Fig.5 (Omics Diagonal)：SpatialEx+ 转录组 ↔ 蛋白组"
    ], top=Inches(4.9), height=Inches(1.6), font_size=17,
       emphasize_words=["Fig.2", "Fig.3", "Fig.4", "Fig.5 (Omics Diagonal)"])

    # ================================================================
    # 16. Reproduction summary: one figure + one table
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "复现成果总览：一图一表")
    # Left: bar chart
    chart_path = render_reproduction_summary_chart()
    img_w, img_h = get_image_pixel_size(chart_path)
    max_h = Inches(4.6)
    max_w = Inches(6.0)
    scale = min(max_w / Inches(img_w / 150), max_h / Inches(img_h / 150))
    pic_w = Inches(img_w / 150) * scale
    pic_h = Inches(img_h / 150) * scale
    slide.shapes.add_picture(chart_path, Inches(0.5), Inches(1.3),
                             width=pic_w, height=pic_h)
    # Right: summary table
    headers = ["任务", "方法", "方向", "PCC", "SSIM", "CMD"]
    rows = [
        ["Fig.2", "HGNN SpatialEx-512", "S1 / S2", "0.257 / 0.273", "0.419 / 0.425", "0.205 / 0.207"],
        ["Fig.3", "HGNN-512 + Cycle", "S1->B / S2->A", "0.275 / 0.301", "0.308 / 0.332", "0.334 / 0.267"],
        ["Fig.4", "HGNN-512 + Cycle", "S1->B / S2->A", "0.008 / 0.013", "0.439 / 0.433", "0.328 / 0.383"],
        ["Fig.5 (Omics)", "SpatialEx+", "p->RNA / RNA->p", "0.196 / -0.042", "0.367 / 0.187", "0.222 / 0.003"]
    ]
    table = slide.shapes.add_table(len(rows) + 1, len(headers),
                                   Inches(6.8), Inches(1.4),
                                   Inches(6.0), Inches(1.8)).table
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE
        p = cell.text_frame.paragraphs[0]
        p.font.color.rgb = WHITE
        p.font.bold = True
        p.font.size = Pt(11)
        p.alignment = PP_ALIGN.CENTER
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.text = str(val)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(10)
            p.font.color.rgb = DARK_GRAY
            p.alignment = PP_ALIGN.CENTER
            if r_idx % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GRAY
    add_caption(slide, "表中只列官方 SpatialEx/SpatialEx+ 复现结果；GT、MNN 等改进方法在第 3–5 章介绍。", top=Inches(6.45), font_size=15)

    # ================================================================
    # 17. Fig.2 results vs paper
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Fig.2 复现：与原文对比")
    add_section_title(slide, "任务设定", top=Inches(1.25))
    add_bullet_box(slide, [
        "输入仅为 H&E，输出为全转录组 313 genes",
        "协议：双向跨切片 train->test（A 训练 -> B 全部评估；反向再做一次）",
        "与 Fig.3 关键区别：同一切片内同时有 X 与 Y，属于有监督回归"
    ], top=Inches(1.8), height=Inches(1.45), font_size=17)
    add_section_title(slide, "复现结果", top=Inches(3.35))
    headers = ["方法", "S1 PCC", "S2 PCC", "S1 SSIM", "S2 SSIM", "S1 CMD", "S2 CMD"]
    rows = [
        ["HGNN SpatialEx (512)", "0.257", "0.273", "0.419", "0.425", "0.205", "0.207"],
        ["DeepPT", "0.268", "0.276", "0.357", "0.366", "0.271", "0.281"]
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(3.85),
                    font_size_header=13, font_size_body=12,
                    row_height_factor=0.55)
    add_highlight_block(
        slide,
        "与原文对比：论文 Fig.2c 报告 SpatialEx 在 PCC/SSIM/CMD 上均优于 DeepPT。"
        "我们的复现中 DeepPT PCC 略高，但 HGNN SSIM/CMD 更好，空间结构保留更完整；"
        "总体趋势与论文一致。",
        top=Inches(5.55), height=Inches(0.95), font_size=15
    )

    # ================================================================
    # 18. Fig.3 results vs paper
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Fig.3 复现：与原文对比")
    add_section_title(slide, "任务设定", top=Inches(1.25))
    add_bullet_box(slide, [
        "Panel Diagonal Integration，官方 150/163 gene split",
        "Slice 1 测 panel A（150 genes），Slice 2 测 panel B（163 genes）",
        "方法：SpatialEx+ HGNN-512 + Cycle Consistency"
    ], top=Inches(1.8), height=Inches(1.35), font_size=17)
    add_section_title(slide, "复现结果", top=Inches(3.25))
    headers = ["编码器", "监督", "S1 PCC", "S2 PCC", "S1 SSIM", "S2 SSIM", "S1 CMD", "S2 CMD"]
    rows = [
        ["HGNN-512", "Cycle", "0.275", "0.301", "0.308", "0.332", "0.334", "0.267"],
        ["DeepPT", "—", "0.262", "0.280", "0.334", "0.360", "0.298", "0.304"]
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(3.75),
                    font_size_header=12, font_size_body=11,
                    row_height_factor=0.5)
    add_highlight_block(
        slide,
        "与原文对比：论文 Fig.3 展示了 SpatialEx+ 在 panel diagonal 上的性能；"
        "我们的 Cycle baseline 在官方 split 上复现出相近数量级，"
        "并补充了 DeepPT baseline 与 CMD 指标。"
        "SpatialEx+ PCC 略高，DeepPT SSIM 略高，两者互有胜负，"
        "是后续 GT、MNN 改进的对照基准。",
        top=Inches(5.45), height=Inches(1.0), font_size=15
    )

    # ================================================================
    # 19. Fig.4 + Fig.5 (Omics Diagonal) results vs paper
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Fig.4 与 Fig.5 (Omics Diagonal) 复现：与原文对比")
    add_section_title(slide, "Fig.4：百万细胞大切片 Panel Diagonal", top=Inches(1.25))
    headers = ["方向", "PCC", "SSIM", "CMD"]
    rows = [
        ["Slice1 -> panel B", "0.008", "0.439", "0.328"],
        ["Slice2 -> panel A", "0.013", "0.433", "0.383"]
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(1.75),
                    font_size_header=13, font_size_body=12,
                    row_height_factor=0.5)
    add_section_title(slide, "Fig.5 (Omics Diagonal)：转录组 <-> 蛋白组", top=Inches(3.35))
    headers = ["方向", "PCC", "SSIM", "CMD"]
    rows = [
        ["protein -> RNA", "0.196", "0.367", "0.222"],
        ["RNA -> protein", "-0.042", "0.187", "0.003"]
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(3.85),
                    font_size_header=13, font_size_body=12,
                    row_height_factor=0.5)
    add_caption(slide, "论文 Fig.5 主图以空间可视化为主，定量指标详见 Extended Data Fig.8a；我们复现了 transcriptomics ↔ proteomics。RNA→protein 方向仅 2 个蛋白 marker，两种方法均接近随机。Fig.5 的 MNN 探索结果见 Appendix。", top=Inches(6.35), font_size=14)

    # ================================================================
    # 20. Bugs
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "复现中修复的 6 个关键 Bug")
    headers = ["#", "位置", "问题", "影响"]
    rows = [
        ["1", "preprocess.normalize_graph", "未定义变量 adj", "图归一化函数无法运行"],
        ["2", "preprocess", "'crs' 拼写错误", "sparse matrix 转换失败"],
        ["3", "Build_hypergraph_spatial_and_HE", "默认返回 coo", "后续期望 CSR，类型/维度错误"],
        ["4", "Model_Plus.forward", "agg_mtx 维度不匹配", "forward 矩阵乘法报错"],
        ["5", "SpatialExP.train", "Regression 维度不匹配", "Cycle module 训练崩溃"],
        ["6", "Model_Plus / Regression", "BatchNorm batch=1 崩溃", "小 batch 训练中断"]
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(1.4),
                    font_size_header=15, font_size_body=14)
    add_bullet_box(slide, [
        "所有 bug 均在官方原始仓库中验证，详见 docs/report/bug_verification_original_repo.md",
        "修复这些 bug 是后续实验的前提；否则 Fig.3 主实验无法完整跑通。"
    ], top=Inches(6.2), font_size=15, color=ACCENT_ORANGE)

    # ================================================================
    # 21. Section divider - Network improvement
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_section_divider(slide, "三. 改进一：Graph Transformer 替代 HGNN",
                        "复现跑通后，先尝试改进网络架构，看是否能把 HGNN 换成更强的 GT")

    # ================================================================
    # 22. GT motivation and module replacement
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "改进一：动机与模块替换")
    add_section_title(slide, "HGNN 的局限", top=Inches(1.25))
    add_bullet_box(slide, [
        "每个细胞与其 k 个空间近邻构成超边，邻居做固定权重均值聚合",
        "无法根据特征相似性自适应调整聚合权重",
        "改进思路：Graph Transformer 在局部邻域内做稀疏自注意力",
        "同时用 MFP 替代 DGI，用 LayerNorm 替代 BatchNorm"
    ], top=Inches(1.75), height=Inches(1.55), font_size=17)
    add_formula(
        slide,
        r"\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j\in\mathcal{N}(i)} \frac{1}{|\mathcal{N}(i)|}\mathbf{W}^{(l)}\mathbf{h}_j^{(l)}\right)",
        left=Inches(0.7), top=Inches(3.35), height=Inches(0.55), fontsize=15,
    )
    add_section_title(slide, "SpatialEx-GT 模块对照", top=Inches(4.05))
    headers = ["模块", "SpatialEx+ (HGNN)", "SpatialEx-GT", "改动动机"]
    rows = [
        ["空间编码器", "HGNN 固定超边聚合", "GT 稀疏邻居注意力", "自适应权重"],
        ["自监督", "DGI 全局对比", "MFP 掩码重建", "局部结构信号"],
        ["翻译器", "MLP Regression", "轻量 MLP", "164k 细胞可训练"],
        ["归一化", "BatchNorm", "LayerNorm", "适配 batch=1"],
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(4.45),
                    font_size_header=12, font_size_body=11, row_height_factor=0.52)
    add_highlight_block(
        slide,
        "实现：SpatialEx/model_improved.py → SpatialExP_GT；"
        "GT hidden=128（显存峰值限制），HGNN baseline=512。",
        top=Inches(6.55), height=Inches(0.55), font_size=14,
    )

    # ================================================================
    # 23. GT architecture
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)

    def _gt_arch_sidebar(sl, left):
        add_section_title(sl, "模块对照", top=Inches(1.2), left=left)
        headers = ["模块", "HGNN", "GT"]
        rows = [
            ["编码器", "固定聚合", "稀疏注意力"],
            ["自监督", "DGI", "MFP"],
            ["翻译器", "MLP", "轻量 MLP"],
            ["归一化", "BatchNorm", "LayerNorm"],
        ]
        add_table_slide(sl, "", headers, rows, top=Inches(1.55), left=left,
                        width=Inches(4.9), font_size_header=11, font_size_body=10,
                        row_height_factor=0.48)
        add_bullet_box(sl, [
            "H&E + 空间坐标 → k-NN → GT Encoder",
            "三路损失：Panel / MFP / Cycle",
            "Cross-Attention 因显存退化为轻量 MLP",
            "实现：SpatialExP_GT，hidden=128",
        ], top=Inches(4.35), left=left, width=Inches(4.9), font_size=14)

    add_architecture_slide(slide, "GT 架构说明", "12_gt_architecture", _gt_arch_sidebar)

    # ================================================================
    # 24. Fig.3 HGNN vs GT (Cycle)
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Fig.3 结果：GT vs HGNN（Official Split + Cycle）")
    headers = ["编码器", "Hidden", "S1 PCC", "S2 PCC", "S1 SSIM", "S2 SSIM"]
    rows = [
        ["HGNN-512 + Cycle", "512", "0.275", "0.301", "0.308", "0.332"],
        ["GT-128 + Cycle", "128", "0.267", "0.276", "0.345", "0.357"],
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(1.25),
                    font_size_header=12, font_size_body=12, row_height_factor=0.55)
    img_path = diag_fig("14_gt_vs_hgnn_fig3_bar")
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.45), Inches(2.85), width=Inches(12.4))
    add_highlight_block(
        slide,
        "PCC：HGNN 两方向均更高；SSIM：GT 略优（+0.04），但未转化为 missing panel 预测精度提升。",
        top=Inches(6.55), height=Inches(0.55), font_size=14,
    )

    # ================================================================
    # 25. Fig.3 four-method context
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_image_slide(slide, diag_fig("17_fig3_cycle_methods_bar"),
                    "Fig.3 多方法对照（DeepPT / HGNN / GT / MLP+MNN）",
                    max_height=Inches(5.2))
    add_caption(
        slide,
        "GT+Cycle 与 HGNN+Cycle 同属论文方法线；MLP+Strict MNN 作为后续改进参照。"
        "此处 GT 未超过 HGNN，更未接近 MNN。",
        top=Inches(6.65), font_size=14,
    )

    # ================================================================
    # 26. Fig.3 per-gene GT vs HGNN
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_image_slide(slide, diag_fig("16_gt_vs_hgnn_pergene_scatter"),
                    "Fig.3 Per-Gene：GT 未系统性超越 HGNN",
                    max_height=Inches(5.2))
    add_caption(
        slide,
        "对角线下方表示 HGNN 更好。GT 仅赢 62/163（Slice1，38%）和 28/150（Slice2，19%）基因。",
        top=Inches(6.65), font_size=14,
    )

    # ================================================================
    # 27. Fig.2 encoder ablation
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Fig.2 对照：纯 H&E 任务上 HGNN 仍优于 GT")
    headers = ["方法", "S1 PCC", "S2 PCC", "S1 SSIM", "S2 SSIM"]
    rows = [
        ["HGNN SpatialEx (512)", "0.257", "0.273", "0.419", "0.425"],
        ["GT-128 + MFP", "0.228", "0.236", "0.309", "0.320"],
        ["GT-512 + MFP", "0.244", "0.246", "0.339", "0.329"],
        ["GT-128 + DGI", "0.225", "0.235", "0.297", "0.313"],
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(1.2),
                    font_size_header=12, font_size_body=11, row_height_factor=0.52)
    img_path = diag_fig("15_fig2_encoder_ablation")
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.45), Inches(2.75), width=Inches(12.4))
    add_highlight_block(
        slide,
        "GT-512 容量提升后仍追不平 HGNN；MFP vs DGI 几乎无差别 → 瓶颈不在自监督形式。",
        top=Inches(6.55), height=Inches(0.55), font_size=14,
    )

    # ================================================================
    # 28. GT summary and transition
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "GT 改进小结")
    add_bullet_box(slide, [
        "Fig.3：换 GT 未带来 PCC 优势；per-gene 胜率低",
        "Fig.2：HGNN 在纯 H&E 任务上仍全面领先",
        "根因：Fig.3 有效信号来自 panel-to-panel，H&E branch 弱",
        "容量：GT backward peak 限制 hidden=128",
        "下一步 → Branch Decomposition：量化 H&E vs panel 贡献"
    ], top=Inches(1.35), height=Inches(3.5), font_size=19, width=Inches(6.3),
       bullet_levels=[0, 0, 0, 0, 0])
    img_path = diag_fig("08_signal_contribution")
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(6.9), Inches(1.45), width=Inches(6.0))
    add_caption(
        slide,
        "Branch Decomposition 预告：measured panel 贡献占主导，H&E branch 接近 0。",
        top=Inches(6.7), font_size=14,
    )

    # ================================================================
    # 29. Section divider - Diagnosis
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_section_divider(slide, "四. 诊断：Branch Decomposition 与信号来源分析",
                        "网络不是瓶颈，那么真正有效的信号来自哪里？")

    # ================================================================
    # 30. Branch decomposition motivation
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "诊断动机：SpatialEx+ 的有效信号来自哪里？")
    add_section_title(slide, "两条预测路径", top=Inches(1.25))
    add_bullet_box(slide, [
        "H&E branch：$X \\to \\hat{Y}_{\\text{missing}}$（形态 → 缺失 panel）",
        "Panel branch：$Y_{\\text{measured}} \\to \\hat{Y}_{\\text{missing}}$（已测 panel → 缺失 panel）",
        "改进一结论：换 GT 未带来 Fig.3 增益 → 瓶颈可能不在图网络"
    ], top=Inches(1.65), height=Inches(1.55), font_size=17)
    add_section_title(slide, "Branch Decomposition 要回答", top=Inches(3.45))
    headers = ["问题", "若成立 → 含义"]
    rows = [
        ["H&E branch 单独能预测缺失 panel？", "能 → 换 encoder 应有帮助"],
        ["Panel branch 是否占主导？", "是 → 应改进 panel-to-panel 监督"],
        ["两分支融合能否互补？", "能 → late fusion 应提升"],
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(3.85),
                    font_size_header=13, font_size_body=12, row_height_factor=0.58)
    add_highlight_block(
        slide,
        "机制诊断：把两条路径拆开独立训练与评估，量化各自贡献；不是再训一个更强模型。",
        top=Inches(6.55), height=Inches(0.55), font_size=14,
    )

    # ================================================================
    # 31. Branch decomposition experiment design
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Branch Decomposition 实验设计")

    # Main flowchart
    flow_path = render_branch_decomposition_design()
    slide.shapes.add_picture(flow_path, Inches(0.55), Inches(1.15), width=Inches(11.8))

    # Bottom explanation
    add_bullet_box(slide, [
        "H&E branch：只用形态嵌入 X 预测缺失 panel，测试 H&E 本身的信息量",
        "Panel branch：只用已测 panel Y 预测缺失 panel，测试 panel-to-panel 信息量",
        "Late Fusion：把两分支预测做 0.5 平均或 reliability-weighted，测试能否互补",
        "机制诊断用 random gene split；定性结论对 official split 同样成立"
    ], top=Inches(6.15), height=Inches(1.0), font_size=15, color=DARK_BLUE)

    # ================================================================
    # 32. Branch decomposition results
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Branch Decomposition 结果与结论")
    headers = ["Variant", "Slice1 PanelB PCC", "Slice2 PanelA PCC"]
    rows = [
        ["H&E branch", "-0.001", "0.010"],
        ["Panel branch", "0.016", "0.248"],
        ["0.5 average (calibrated)", "0.011", "0.177"],
        ["Reliability-weighted (calibrated)", "-0.001", "0.010"],
    ]
    add_table_slide(slide, "Random gene split 机制诊断（定性结论对 official split 成立）",
                    headers, rows, top=Inches(1.25),
                    font_size_header=12, font_size_body=12, row_height_factor=0.55)
    add_bullet_box(slide, [
        "H&E branch ≈ 0：H&E 不能形成稳定的 missing panel 先验",
        "Panel branch dominant：有效信号来自 measured → missing panel",
        "Late fusion 有害：Slice2 从 0.248 降至 0.177 → H&E 是噪声而非补充",
        "→ GT 无提升的根因是 H&E 输入信息量低；后续应改监督信号（Strict MNN）"
    ], top=Inches(3.55), height=Inches(2.6), font_size=17, color=DARK_BLUE)

    # ================================================================
    # 33. Branch decomposition bar chart
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Branch Decomposition 可视化")

    chart_path = render_branch_decomposition_results()
    slide.shapes.add_picture(chart_path, Inches(1.5), Inches(1.3), width=Inches(10.3))

    add_caption(
        slide,
        "四 variant 对比：Panel branch 在 Slice2 显著高于 H&E branch；"
        "融合 variant 低于 Panel branch，验证 late fusion 拉低强信号。",
        top=Inches(6.25), font_size=15,
    )

    # ================================================================
    # 34. Signal contribution
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    img_path = diag_fig("08_signal_contribution")
    add_image_slide(slide, img_path, "信号贡献定量分析")
    add_caption(slide, "Slice2 方向：Panel branch 0.248 vs H&E 0.010 vs fusion 0.177。从另一角度验证 panel dominant、late fusion 有害。", top=Inches(6.6), font_size=15)

    # ================================================================
    # 35. Per-gene scatter
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    img_path = diag_fig("03_per_gene_scatter")
    add_image_slide(slide, img_path, "诊断图：Per-Gene 预测散点")
    add_caption(slide, "大多数基因上 MNN 监督优于 Cycle 监督，为改进二（Strict MNN）做铺垫。", top=Inches(6.6), font_size=15)

    # ================================================================
    # 36. Section divider - Supervision improvement
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_section_divider(slide, "五. 改进二：Strict MNN 伪标签替代 Cycle",
                        "H&E 和 Cycle 都不够强，转向跨切片 pseudo-label")

    # ================================================================
    # 31. Cycle trap
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Cycle Consistency 的自洽陷阱")
    add_bullet_box(slide, [
        "Cycle 约束的定义：",
        "问题：Cycle 只保证自洽，不保证预测接近真实 missing panel",
        "  • 模型可以学到一种“幻觉映射”让循环闭合",
        "  • 缺少外部真实监督时，无法约束预测靠近真实生物学信号",
        "类比：翻译模型把英语→法语→英语，句子通顺≠法语翻译准确",
        "实验证据：",
        "  • MLP + Cycle only 的 PCC 仅 0.005 / 0.013，接近随机",
        "  • 说明 Cycle 本身几乎不提供有效监督"
    ], top=Inches(1.3), font_size=18,
       bullet_levels=[0, 0, 1, 1, 0, 0, 1, 1])
    add_formula(slide, r"Y_A \to \hat{Y}_B \to Y_A' \approx Y_A",
                left=Inches(7.8), top=Inches(1.35), height=Inches(0.5), fontsize=18)
    add_formula(slide, r"Y_B \to \hat{Y}_A \to Y_B' \approx Y_B",
                left=Inches(7.8), top=Inches(2.05), height=Inches(0.5), fontsize=18)
    add_formula(slide, r"\text{Self-consistency } \nRightarrow \hat{Y}_B \approx Y_B",
                left=Inches(6.8), top=Inches(2.75), height=Inches(0.5), fontsize=17)

    # ================================================================
    # 32. Cycle trap figure
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    img_path = diag_fig("07_cycle_trap")
    add_image_slide(slide, img_path, "Cycle Self-Consistency Trap")
    add_caption(slide, "Cycle 让模型在训练集上“自我闭环”，但测试时预测可能与真实 panel 无关。这是 Cycle 作为唯一监督信号的根本缺陷。", top=Inches(6.7), font_size=15)

    # ================================================================
    # 39. Strict MNN motivation and protocol
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "改进二动机：Strict MNN 替代 Cycle")
    add_bullet_box(slide, [
        "相邻切片组织连续 → 跨切片细胞存在形态/空间对应",
        "用 MNN 建立可信配对，把参考切片 measured panel 转移为 pseudo-label",
        "替代 Cycle 自环：外部锚定监督，而非同切片自我一致",
    ], top=Inches(1.25), height=Inches(1.55), font_size=18)
    add_section_title(slide, "Strict Fig.3 协议", top=Inches(3.0))
    headers = ["切片", "训练/伪标签可用", "Held-out（eval only）"]
    rows = [
        ["Slice 1", "X1, YA1", "YB1"],
        ["Slice 2", "X2, YB2", "YA2"],
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(3.4),
                    font_size_header=13, font_size_body=13, row_height_factor=0.58)
    add_highlight_block(
        slide,
        "YB1、YA2 绝不参与 MNN 匹配或训练；pseudo-label 只能来自允许输入。",
        top=Inches(5.15), height=Inches(0.55), font_size=15,
    )

    # ================================================================
    # 40. MNN single-step algorithm
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)

    def _mnn_algo_sidebar(sl, left):
        add_section_title(sl, "单步原子（非 Step 综合）", top=Inches(1.15), left=left)
        add_bullet_box(sl, [
            "函数：build_mnn_pseudo(query, ref, y_ref)",
            "Strict MNN 的 Step 1 / Step 2 均重复调用此函数",
            "L2 normalize → Fwd → Rev → Mutual → mean",
            "Fallback：无 Mutual 时用 Fwd 前 k 个",
            "默认 k=5, mnn_k=20",
            "实现：SpatialEx_conditional_gt.py",
        ], top=Inches(1.5), left=left, width=Inches(4.9), font_size=12)

    add_architecture_slide(
        slide, "MNN 单步原子：build_mnn_pseudo（图 19）",
        "19_mnn_filter_algorithm", _mnn_algo_sidebar,
    )

    # ================================================================
    # 41. Strict MNN two-step bridge
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Strict MNN：Step 1 + Step 2 桥接")

    # Left: simple flowchart
    flow_path = render_mnn_bridge_flowchart()
    slide.shapes.add_picture(flow_path, Inches(0.35), Inches(1.15), width=Inches(7.4))

    # Right: step-by-step text explanation
    add_section_title(slide, "Step 1 · H&E Bridge", top=Inches(1.2), left=Inches(7.95),
                      width=Inches(4.95), font_size=17, accent_color=ACCENT_ORANGE)
    add_formula(
        slide,
        r"\tilde{Y}_B^1=\mathrm{MNN}(X_1,X_2,Y_B^2)",
        left=Inches(7.95), top=Inches(1.65), height=Inches(0.4), fontsize=14,
    )
    add_bullet_box(slide, [
        "目标：构造 Slice1 缺失 panel B 的 pseudo-label",
        "匹配空间：UNI H&E 嵌入 X1 ↔ X2",
        "转移目标：Slice2 已测的 Y_B^2",
    ], top=Inches(2.1), left=Inches(7.95), width=Inches(4.95), height=Inches(1.15), font_size=12)

    add_section_title(slide, "Step 2 · Panel B Bridge", top=Inches(3.35), left=Inches(7.95),
                      width=Inches(4.95), font_size=17, accent_color=ACCENT_ORANGE)
    add_formula(
        slide,
        r"\tilde{Y}_A^2=\mathrm{MNN}(Y_B^2,\tilde{Y}_B^1,Y_A^1)",
        left=Inches(7.95), top=Inches(3.8), height=Inches(0.4), fontsize=14,
    )
    add_bullet_box(slide, [
        "目标：构造 Slice2 缺失 panel A 的 pseudo-label",
        "匹配空间：Y_B^2（Slice2 query）↔ Ỹ_B^1（Step1 产物）",
        "转移目标：Slice1 已测的 Y_A^1",
        "注意：Ỹ_B^1 是 Step 1 构造产物，不是真实 Y_B^1",
    ], top=Inches(4.25), left=Inches(7.95), width=Inches(4.95), height=Inches(1.5), font_size=12)

    add_highlight_block(
        slide,
        "核心：两次调用同一个 build_mnn_pseudo 原子；全程不使用 held-out Y_B^1、Y_A^2。"
        "Step 1 用形态找跨切片对应，Step 2 在 B-panel 空间 refine 对应。",
        top=Inches(5.95), height=Inches(0.85), font_size=14,
    )

    # ================================================================
    # 42. Full MNN architecture + MLP training
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)

    def _mnn_pipeline_sidebar(sl, left):
        add_section_title(sl, "全流程（图 13）", top=Inches(1.05), left=left)
        add_section_title(sl, "Step 1（左图蓝框）", top=Inches(1.38), left=left, font_size=16)
        add_bullet_box(sl, [
            r"$\tilde{Y}_B^1=\mathrm{MNN}(X_1,X_2,Y_B^2)$",
            "H&E 跨切片匹配 → 构造 Slice1 pseudo-B",
        ], top=Inches(1.72), left=left, width=Inches(4.9), font_size=11, height=Inches(0.75))
        add_section_title(sl, "Step 2（左图橙框）", top=Inches(2.35), left=left, font_size=16)
        add_bullet_box(sl, [
            r"$\tilde{Y}_A^2=\mathrm{MNN}(Y_B^2,\tilde{Y}_B^1,Y_A^1)$",
            "B-panel 空间 refine → 构造 Slice2 pseudo-A",
        ], top=Inches(2.68), left=left, width=Inches(4.9), font_size=11, height=Inches(0.75))
        add_section_title(sl, "MLP 训练", top=Inches(3.3), left=left, font_size=16)
        add_bullet_box(sl, [
            "输入 measured panel；损失对齐 $\\tilde{Y}_B^1, \\tilde{Y}_A^2$",
            "SpatialExP_ConditionalMLP",
        ], top=Inches(3.62), left=left, width=Inches(4.9), font_size=11, height=Inches(0.65))
        add_formula(
            sl,
            r"\mathcal{L}=\|\hat{Y}_B^1-\tilde{Y}_B^1\|_2^2+\|\hat{Y}_A^2-\tilde{Y}_A^2\|_2^2",
            left=left - Inches(0.12), top=Inches(4.35), height=Inches(0.38), fontsize=11,
        )
        add_section_title(sl, "vs Cycle", top=Inches(4.75), left=left, font_size=16)
        headers = ["维度", "Cycle", "Strict MNN"]
        rows = [
            ["监督", "同切片自环", "Step1+2 外部锚定"],
            ["PCC", "0.005/0.013", "0.334/0.371"],
        ]
        add_table_slide(sl, "", headers, rows, top=Inches(5.05), left=left,
                        width=Inches(4.9), font_size_header=9, font_size_body=9,
                        row_height_factor=0.48)

    add_architecture_slide(
        slide, "Strict MNN 全流程：Step 1 → Step 2 → MLP（图 13）",
        "13_mnn_architecture", _mnn_pipeline_sidebar,
    )

    # ================================================================
    # 42b. Combinatorial space: encoder × supervision
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Fig.3 的搜索空间：编码器 × 监督信号")
    add_bullet_box(slide, [
        "复现后，我们把 SpatialEx+ 解耦成两个可独立替换的模块：",
        "  • 编码器（Encoder）：决定如何从输入提取特征",
        "  • 监督信号（Supervision）：决定 missing panel 的学习目标",
        "通过 shuffle 这两个模块，可以定位 Fig.3 的真正瓶颈。"
    ], top=Inches(1.2), height=Inches(1.5), font_size=17)
    headers = ["编码器", "监督信号", "实验意图"]
    rows = [
        ["HGNN-512", "Cycle", "官方 SpatialEx+ baseline"],
        ["GT-128", "Cycle", "改进一：换更强的图编码器"],
        ["MLP", "Cycle only", "Ablation：去掉 MNN 的外部监督"],
        ["MLP", "Strict MNN", "改进二：换监督信号"],
        ["MLP", "MNN + Cycle", "验证 Cycle 是否会干扰 MNN"],
        ["GT-128", "Strict MNN", "同时换编码器 + 监督"],
        ["HGNN-512", "Strict MNN", "官方编码器 + 新监督"],
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(2.85),
                    font_size_header=13, font_size_body=12,
                    row_height_factor=0.48)
    add_highlight_block(
        slide,
        "关键问题：瓶颈在编码器，还是在监督信号？"
        "如果换 GT 无效而换 MNN 有效，说明监督信号才是 Fig.3 的主瓶颈。",
        top=Inches(6.25), height=Inches(0.75), font_size=15,
    )

    # ================================================================
    # 43. Headline Fig.3 results
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Fig.3 主结果：MLP + Strict MNN")
    headers = ["编码器", "监督", "S1 PCC", "S2 PCC", "S1 SSIM", "S2 SSIM"]
    rows = [
        ["HGNN-512", "Cycle", "0.275", "0.301", "0.308", "0.332"],
        ["MLP", "Cycle only", "0.005", "0.013", "0.114", "0.107"],
        ["MLP", "Strict MNN", "0.334", "0.371", "0.374", "0.398"],
        ["MLP", "MNN + Cycle", "0.315", "0.353", "0.344", "0.388"],
    ]
    add_table_slide(slide, "Official Split", headers, rows, top=Inches(1.35),
                    font_size_header=12, font_size_body=12, row_height_factor=0.55)
    add_highlight_block(
        slide,
        "MNN + Cycle 略低于纯 MNN → Cycle 与外部监督冲突；MLP + Strict MNN 为当前最优。",
        top=Inches(4.55), height=Inches(0.55), font_size=15,
    )
    add_formula(
        slide,
        r"\mathcal{L}=\|\hat{Y}_B^1-\tilde{Y}_B^1\|_2^2+\|\hat{Y}_A^2-\tilde{Y}_A^2\|_2^2",
        left=Inches(2.5), top=Inches(5.35), height=Inches(0.5), fontsize=16,
    )

    # ================================================================
    # 44. MNN summary
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "MNN 改进小结")
    add_bullet_box(slide, [
        "外部锚定：pseudo-label 来自跨切片 measured panel，非自环",
        "空间信息已编码：H&E MNN 配对形态相似细胞；Panel B MNN 配对表达相似细胞",
        "MNN 过滤降噪：双向互为近邻，剔除单向 kNN 低质量配对",
        "任务匹配：Fig.3 本质是 panel-to-panel；MLP 足够，图模型 + MNN 反而 SSIM 崩",
        "与改进一：GT 增强 H&E 无效；MNN 改善监督有效 → 监督质量 > 模型复杂度",
    ], top=Inches(1.35), height=Inches(4.8), font_size=18, color=DARK_BLUE)

    # ================================================================
    # 45. MNN sweep
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    img_path = diag_fig("02_mnn_sweep")
    add_image_slide(slide, img_path, "MNN 参数敏感性扫描")
    add_caption(slide, "横轴：邻居数 k 或 mnn_k；纵轴：gene-level PCC。MNN 方法在一定范围内鲁棒，不是依赖某个特定超参数“凑”出来的结果。", top=Inches(6.6), font_size=15)

    # ================================================================
    # 37. Latent alignment + Latent MNN table
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "跨切片 Latent Alignment 与 Latent MNN")
    add_bullet_box(slide, [
        "UMAP/t-SNE 展示 H&E 空间或 panel 空间的细胞分布",
        "比较 MNN 配对前后的跨切片重叠程度",
        "不同 matching 空间的效果对比："
    ], top=Inches(1.3), font_size=18, bullet_levels=[0, 0, 0])
    headers = ["Matching space", "Slice1 learned PCC", "Slice2 learned PCC"]
    rows = [
        ["raw measured panel", "0.015", "0.264"],
        ["PCA latent (50-d)", "0.007", "0.291"],
        ["CORAL aligned", "0.010", "0.228"]
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(3.0),
                    font_size_header=16, font_size_body=15,
                    row_height_factor=0.55)
    add_bullet_box(slide, [
        "PCA latent + MNN 在 Slice2 上进一步提升到 0.291",
        "CORAL 线性对齐反而下降，可能过度抹平生物学相关的切片间差异",
        "Slice1 始终接近 0，说明该方向的信息桥本身较弱"
    ], top=Inches(5.4), font_size=16, color=DARK_BLUE)

    # ================================================================
    # 38. Main results
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Official Split 主结果")
    headers = ["编码器", "监督", "S1 PCC", "S2 PCC", "S1 SSIM", "S2 SSIM", "S1 CMD", "S2 CMD"]
    rows = [
        ["HGNN-512", "Cycle", "0.275", "0.301", "0.308", "0.332", "0.334", "0.267"],
        ["DeepPT", "—", "0.262", "0.280", "0.334", "0.360", "0.298", "0.304"],
        ["GT-128", "Cycle", "0.267", "0.276", "0.345", "0.357", "0.296", "0.289"],
        ["MLP", "Cycle only", "0.005", "0.013", "0.114", "0.107", "0.780", "0.788"],
        ["MLP", "Strict MNN", "0.334", "0.371", "0.374", "0.398", "0.383", "0.330"],
        ["MLP", "MNN + Cycle", "0.315", "0.353", "0.344", "0.388", "0.355", "0.323"],
        ["GT-128", "Strict MNN", "0.258", "0.289", "0.359", "0.387", "0.327", "0.356"],
        ["HGNN-512", "Strict MNN", "0.234", "0.273", "0.072", "0.055", "0.347", "0.369"]
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(1.4),
                    font_size_header=12, font_size_body=11,
                    row_height_factor=0.45)
    add_highlight_block(
        slide,
        "结论：MLP + Strict MNN（0.334 / 0.371）显著优于官方 HGNN/GT + Cycle 与 DeepPT；"
        "监督信号质量比模型复杂度更关键。",
        top=Inches(5.65), height=Inches(0.8), font_size=16
    )

    # ================================================================
    # 39. PCC distribution
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    img_path = diag_fig("10_pcc_distribution")
    add_image_slide(slide, img_path, "PCC 分布对比")
    add_caption(slide, "小提琴图/箱线图比较不同方法在所有基因上的 PCC 分布。MLP + Strict MNN 的分布整体右移；Cycle 方法分布接近 0 或负值。", top=Inches(6.6), font_size=15)

    # ================================================================
    # 40. Gain distribution
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    img_path = diag_fig("04_mnn_gain_distribution")
    add_image_slide(slide, img_path, "Per-Gene 提升分布")
    add_caption(slide, "横轴：ΔPCC（MNN - Cycle）；纵轴：基因密度。大多数基因提升为正；78.7% 基因获得提升。少数基因下降，说明仍有改进空间。", top=Inches(6.5), font_size=15)

    # ================================================================
    # 41. Top markers + table
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Top Marker Genes 提升（HGNN+Cycle vs MLP+MNN）")

    # Left: bar chart of top gains
    chart_path = render_top_marker_gains_chart(top_n=10)
    if chart_path and os.path.exists(chart_path):
        slide.shapes.add_picture(chart_path, Inches(0.4), Inches(1.2), width=Inches(6.2))
    else:
        add_bullet_box(slide, ["Top marker gains chart not found"], top=Inches(1.2), font_size=16)

    # Right: top 5 table
    add_section_title(slide, "Slice 2 上提升最显著的 marker 基因", top=Inches(1.2), left=Inches(7.0),
                      width=Inches(5.7), font_size=17, accent_color=ACCENT_ORANGE)
    headers = ["Gene", "HGNN+Cycle", "MLP+MNN", "提升"]
    rows = [
        ["CPA3", "0.079", "0.578", "+0.498"],
        ["CTSG", "0.072", "0.555", "+0.483"],
        ["CLEC14A", "0.330", "0.638", "+0.308"],
        ["EGFL7", "0.141", "0.358", "+0.217"],
        ["CLDN5", "0.046", "0.262", "+0.216"],
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(1.75), left=Inches(7.0),
                    width=Inches(5.7), font_size_header=13, font_size_body=12,
                    row_height_factor=0.52)
    add_bullet_box(slide, [
        "这些基因涉及免疫、上皮/基质、血管等生物学过程",
        "CPA3 / CTSG 为肥大细胞/胰蛋白酶标记；CLEC14A / EGFL7 / CLDN5 与血管内皮相关",
        "MNN 的改进不只在平均指标上，也在真实空间表达结构上有视觉可辨的提升"
    ], top=Inches(4.6), left=Inches(7.0), width=Inches(5.7), height=Inches(2.1),
       font_size=14, color=DARK_BLUE)

    # ================================================================
    # 42. Marker gene figure
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    img_path = os.path.join(IMG_DIR, "11_marker_genes_slice2.jpg")
    add_image_slide(slide, img_path, "Slice2 Marker Gene 预测对比")
    add_caption(
        slide,
        "PanelA (Slice2) 空间热力图：Ground truth vs HGNN+Cycle vs MLP+MNN。"
        "MLP+MNN 热点区域更接近真实表达；标题中 PCC 与 Top Marker 实验一致。",
        top=Inches(6.6), font_size=14,
    )

    # ================================================================
    # 43. Section divider - Discussion
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_section_divider(slide, "六. 讨论：为什么 MLP + MNN 反而最强？",
                        "为什么一个简单 MLP 能打败复杂网络？")

    # ================================================================
    # 44. Why MLP wins
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "为什么 MLP + MNN 反而最强？")
    add_bullet_box(slide, [
        "1. 任务本质是 panel-to-panel 分子映射，不是空间聚合",
        "   同一切片上 measured panel 与 missing panel 存在生物学相关性",
        "2. MNN 已通过跨切片 matching 编码了空间/形态信息",
        "   H&E MNN 配对形态/位置相似细胞；Panel B MNN 进一步配对表达相似细胞",
        "3. HGNN/GT 的额外空间聚合会 over-smooth 或引入 H&E 噪声",
        "   Branch Decomposition 已显示 H&E 分支接近 0",
        "4. Cycle 与 MNN 冲突：自洽约束干扰可靠的外部监督",
        "   MNN 是有噪声但无偏的外部信号；Cycle 会把预测拉离 MNN 目标",
        "5. HGNN-512 + MNN 的 SSIM 极低（0.07 / 0.06），空间结构严重失真",
        "   而 MLP + MNN 的 SSIM 最高（0.37 / 0.40），空间保真最好",
        "6. 当监督信号强时，简单 MLP 足够",
        "   复杂网络需要大量有效监督才能发挥优势；在 strict 协议下 MLP 更稳健"
    ], top=Inches(1.25), font_size=16,
       bullet_levels=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # ================================================================
    # 44b. Why trained MLP beats pseudo-labels
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "为什么 MLP 能超越 MNN pseudo-label 本身？")
    add_section_title(slide, "实验对比：pseudo-label vs 训练后 MLP", top=Inches(1.25))
    headers = ["构造方式", "pseudo-label 质量", "训练后 MLP", "提升"]
    rows = [
        ["raw cosine kNN", "0.178 / 0.252", "0.327 / 0.381", "+0.149 / +0.129"],
        ["Strict MNN", "0.153 / 0.187", "0.334 / 0.371", "+0.181 / +0.184"]
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(1.75),
                    font_size_header=13, font_size_body=12,
                    row_height_factor=0.55)
    add_section_title(slide, "解释", top=Inches(3.45))
    add_bullet_box(slide, [
        "pseudo-label 是带噪的 per-cell 估计：",
        "  • 每个细胞来自跨切片 kNN/MNN 的少数邻居平均",
        "  • Slice2 还要通过 pseudo_B1 做两步桥接，误差累积",
        "MLP 学到的是全局 panel-to-panel 映射：",
        "  • 输入是同一切片的 measured panel（强信号）",
        "  • 目标是有噪 pseudo-label，模型把噪声平均掉",
        "  • 直接学习 Y_B2 → Y_A2，绕过两步桥接",
        "结论：MLP 把有噪但方向正确的 pseudo-label 蒸馏成稳定 translator"
    ], top=Inches(3.95), height=Inches(2.6), font_size=16,
       bullet_levels=[0, 1, 1, 0, 1, 1, 1, 0])

    # ================================================================
    # 56. Conclusion
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "结论")
    add_bullet_box(slide, [
        "复现：跑通 SpatialEx/SpatialEx+（Fig.2–3），修复 6 个官方代码 bug",
        "复现：补充 Fig.4 百万细胞与 Fig.5 Omics Diagonal 初步结果",
        "复现：Fig.2 baseline 中 HGNN 在纯 H&E 任务上仍优于 GT",
        "诊断：Branch Decomposition 揭示 H&E branch 弱、panel branch dominant",
        "改进：Strict MNN 伪标签替代 Cycle，提供可靠的外部监督",
        "发现：MLP + Strict MNN 效果最佳（PCC 0.334 / 0.371，SSIM 0.374 / 0.398）"
    ], top=Inches(1.35), height=Inches(3.4), font_size=20,
       emphasize_words=["MLP + Strict MNN"])
    add_highlight_block(
        slide,
        "核心洞察：缺失 panel 的监督信号质量比模型复杂度更重要；"
        "在 strict 协议下，简单模型 + 好信号 > 复杂模型 + 弱信号。",
        top=Inches(5.1), height=Inches(0.9), font_size=18
    )

    # ================================================================
    # 57. Limitations
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "局限与后续方向")
    add_two_column_bullets(slide,
        "局限",
        [
            "Slice1 方向仍较弱（official split 信息桥不对称）",
            "仅单一数据集验证",
            "未与 DeepPT 在 Fig.3 对比",
            "CNN_Reg、Hist2ST 等待实现",
            "Fig.4 评估脚本仍有 coo_matrix subscript bug",
            "GT 显存限制，Cross-Attention 退化为 MLP",
            "MNN pseudo-label 有噪声但未被建模"
        ],
        "后续方向",
        [
            "加入位置编码的多模态 matching",
            "在更多癌种和组织上验证",
            "与 DeepPT、CNN_Reg、Hist2ST 公平比较",
            "开发更轻量的 GT 变体",
            "MNN 配对不确定性建模",
            "系统扩展到 Omics Diagonal 更多 marker"
        ],
        top=Inches(1.5), height=Inches(4.8), font_size=17)

    # ================================================================
    # 58. Appendix: MNN results on Fig.4 / Fig.5
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "Appendix：Fig.4 / Fig.5 上的 Strict MNN 探索")
    add_section_title(slide, "Fig.5 Omics Diagonal（transcriptomics ↔ proteomics）", top=Inches(1.25))
    headers = ["方向", "方法", "PCC", "SSIM", "CMD"]
    rows = [
        ["protein -> RNA", "pseudo ceiling", "0.143", "0.916", "0.151"],
        ["protein -> RNA", "MLP + Strict MNN", "0.202", "0.915", "0.464"],
        ["RNA -> protein", "pseudo ceiling", "-0.031", "0.085", "0.001"],
        ["RNA -> protein", "MLP + Strict MNN", "-0.038", "0.076", "0.007"]
    ]
    add_table_slide(slide, "", headers, rows, top=Inches(1.75),
                    font_size_header=12, font_size_body=11,
                    row_height_factor=0.45)
    add_section_title(slide, "Fig.4 百万细胞 Panel Diagonal", top=Inches(4.35))
    add_bullet_box(slide, [
        "MNN 尚未在 Fig.4 大切片上系统运行（两片各 ~900k 细胞，计算与内存开销大）",
        "可作为后续方向：验证 Strict MNN 在百万细胞尺度上的可扩展性"
    ], top=Inches(4.85), height=Inches(1.0), font_size=16)
    add_caption(slide, "Fig.5 中 MLP + Strict MNN 在 protein→RNA 方向略优于 SpatialEx+，但 RNA→protein 方向因 marker 过少而接近随机。", top=Inches(6.35), font_size=14)

    # ================================================================
    # 59. Appendix + Thank you
    # ================================================================
    slide = prs.slides.add_slide(blank_layout)
    add_title_shape(slide, "附录 + 谢谢 / Q&A")
    add_bullet_box(slide, [
        "可引用/展示的论文原图：",
        "  • Fig.4：百万细胞 Panel Diagonal 可扩展性",
        "  • Fig.5：Omics Diagonal Integration（转录组 ↔ 蛋白/代谢）",
        "  • Fig.6：ablation 与扩展分析",
        "  • Extended Data Fig.1–10：方法细节与补充实验",
        "讲点：",
        "  • 论文 Fig.5–6 展示了 Omics Diagonal 等更复杂任务",
        "  • 我们的 Strict MNN 思想可推广到这些场景",
        "  • 未来可借鉴 Extended Data 中的更多分析"
    ], top=Inches(1.4), font_size=19,
       bullet_levels=[0, 1, 1, 1, 1, 0, 1, 1, 1])
    add_highlight_block(
        slide,
        "谢谢！欢迎各位老师斧正 / 欢迎提问",
        top=Inches(5.8), height=Inches(0.8), font_size=20
    )

    # Save
    prs.save(OUTPUT_PATH)
    print(f"{len(prs.slides)}-page presentation saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_presentation()
