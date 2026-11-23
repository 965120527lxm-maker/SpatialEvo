"""Shared matplotlib figure export helpers (SVG for defense slides)."""

import os

import matplotlib.pyplot as plt


def save_figure(fig, out_dir, name, *, facecolor="white", transparent=False, pad_inches=0.05):
    """Save *fig* as SVG under *out_dir*; *name* may include or omit ``.svg``."""
    os.makedirs(out_dir, exist_ok=True)
    if not name.lower().endswith(".svg"):
        name = os.path.splitext(name)[0] + ".svg"
    path = os.path.join(out_dir, name)
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor=facecolor,
        transparent=transparent,
    )
    plt.close(fig)
    print("Saved", path)
    return path
