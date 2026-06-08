"""Tests for Lesson 04: MLP Encoder."""
import sys
from pathlib import Path

starter = Path(__file__).parent / "starter.py"
if not starter.exists():
    print("FAIL: starter.py not found.")
    sys.exit(1)

from starter import MLPEncoder


def test_basic():
    import torch
    torch.manual_seed(42)
    enc = MLPEncoder(100, 64, 32)
    x = torch.randn(5, 100)
    out = enc(x)
    assert out.shape == (5, 32), f"Expected (5, 32), got {out.shape}"
    
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in enc.parameters()), "No gradients!"
    print("PASS")


if __name__ == "__main__":
    test_basic()
