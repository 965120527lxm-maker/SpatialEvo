"""Tests for Lesson 08: Six-Loss Trainer."""
import sys
from pathlib import Path

starter = Path(__file__).parent / "starter.py"
if not starter.exists():
    print("FAIL: starter.py not found.")
    sys.exit(1)

from starter import compute_six_losses


def test_basic():
    import torch
    torch.manual_seed(42)
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)
    losses = compute_six_losses(a, b, b, a, a, b, a, b)
    total = sum(losses[k] for k in losses if k != "total")
    assert torch.isclose(losses["total"], total), "Total mismatch"
    print("PASS")


if __name__ == "__main__":
    test_basic()
