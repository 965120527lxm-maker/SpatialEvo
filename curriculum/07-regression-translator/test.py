"""Tests for Lesson 07: Regression Translator."""
import sys
from pathlib import Path

starter = Path(__file__).parent / "starter.py"
if not starter.exists():
    print("FAIL: starter.py not found.")
    sys.exit(1)

from starter import RegressionTranslator


def test_basic():
    import torch
    trans = RegressionTranslator(64, 100)
    h = torch.randn(5, 64)
    out = trans(h)
    assert out.shape == (5, 100), f"Expected (5, 100), got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert trans.fc.weight.grad is not None
    print("PASS")


if __name__ == "__main__":
    test_basic()
