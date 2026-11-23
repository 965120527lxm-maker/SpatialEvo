#!/usr/bin/env python3
"""Top Slice2 marker gene gains: MLP+Strict MNN vs HGNN+Cycle (horizontal bar chart)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = os.path.join(PROJECT_ROOT, "experiments", "fig3", "top_marker_gains_hgnn_vs_mnn.csv")
OUT_SVG = os.path.join(PROJECT_ROOT, "docs", "image", "top_marker_gains_hgnn_vs_mnn.svg")
OUT_DIAG = os.path.join(PROJECT_ROOT, "docs", "image", "fig3_diagnosis", "09_top_marker_gains.svg")
A_MD = os.path.join(PROJECT_ROOT, "docs", "a.md")

C_GREEN = "#59A14F"


def load_top_gains(top_n=10):
    df = pd.read_csv(CSV_PATH)
    top = df.nlargest(top_n, "gain").copy()
    top = top.sort_values("gain", ascending=True)  # barh: smallest at bottom
    return top


def write_results_markdown(top_df, path=A_MD):
    lines = [
        "# Top Marker Genes：HGNN+Cycle vs MLP+Strict MNN（Slice2）",
        "",
        "ΔPCC = MLP+MNN PCC − HGNN+Cycle PCC（official split，Slice2 PanelA held-out）",
        "",
        "| Gene | HGNN+Cycle | MLP+MNN | ΔPCC |",
        "|------|----------:|--------:|-----:|",
    ]
    ranked = top_df.sort_values("gain", ascending=False)
    for _, row in ranked.iterrows():
        lines.append(
            f"| {row['gene']} | {row['HGNN_Cycle_PCC']:.3f} | {row['MLP_MNN_PCC']:.3f} | {row['gain']:+.3f} |"
        )
    lines.extend(["", f"数据来源：`{os.path.relpath(CSV_PATH, PROJECT_ROOT)}`", ""])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_top_marker_gains(top_n=10, out_path=OUT_SVG):
    top = load_top_gains(top_n)
    xmax = float(top["gain"].max())
    x_pad = max(0.05, xmax * 0.08)

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False,
    }):
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        bars = ax.barh(
            top["gene"], top["gain"],
            color=C_GREEN, height=0.58, edgecolor="none",
        )
        ax.set_xlim(0, xmax + x_pad)
        ax.set_xlabel(r"$\Delta$PCC (MLP+MNN $-$ HGNN+Cycle)", fontsize=11, labelpad=10)
        ax.set_title(
            f"Top {top_n} Slice2 marker genes improved by MNN",
            fontsize=13, fontweight="bold", pad=18,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=10)

        # Value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f}", ha="left", va="center",
                    fontsize=8, color="dimgray")

        # Generous margins to avoid overlap between title/labels and the chart
        fig.subplots_adjust(left=0.20, right=0.95, top=0.88, bottom=0.13)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.15,
                    facecolor="white")
        plt.close(fig)

    write_results_markdown(top)
    return out_path


def main():
    plot_top_marker_gains(top_n=10, out_path=OUT_SVG)
    plot_top_marker_gains(top_n=10, out_path=OUT_DIAG)
    print("Saved", OUT_SVG)
    print("Saved", OUT_DIAG)
    print("Updated", A_MD)


if __name__ == "__main__":
    main()
