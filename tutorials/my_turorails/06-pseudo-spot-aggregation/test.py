"""Tests for Lesson 06: Pseudo-Spot Aggregation."""
import sys
from pathlib import Path

starter = Path(__file__).parent / "starter.py"
if not starter.exists():
    print("FAIL: starter.py not found.")
    sys.exit(1)

from starter import aggregate_pseudo_spots


def test_basic():
    import torch
    N, M, G = 10, 3, 5
    expr = torch.ones(N, G)
    cell_coords = torch.randn(N, 2) * 100
    spot_coords = cell_coords[:M].clone()
    out = aggregate_pseudo_spots(expr, cell_coords, spot_coords, radius=1e-6)
    assert out.shape == (M, G), f"Expected ({M}, {G}), got {out.shape}"
    # Each spot should include at least itself
    assert (out.sum(dim=1) > 0).all(), "Some spots got zero expression"
    print("PASS")


if __name__ == "__main__":
    test_basic()
