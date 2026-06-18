"""Tests for Lesson 09: Graph-Aware Evaluation."""
import sys
from pathlib import Path

starter = Path(__file__).parent / "starter.py"
if not starter.exists():
    print("FAIL: starter.py not found.")
    sys.exit(1)

from starter import graph_ssim


def test_basic():
    import torch
    N, G = 100, 3
    pred = torch.rand(N, G)
    true = pred.clone()
    coords = torch.rand(N, 2) * 100
    s = graph_ssim(pred, true, coords, img_size=32)
    assert 0.0 <= s <= 1.0, f"SSIM out of range: {s}"
    print("PASS")


if __name__ == "__main__":
    test_basic()
