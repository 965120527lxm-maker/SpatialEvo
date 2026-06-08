"""Tests for Lesson 05: Single-Slice Model."""
import sys
from pathlib import Path

starter = Path(__file__).parent / "starter.py"
if not starter.exists():
    print("FAIL: starter.py not found.")
    sys.exit(1)

from starter import SingleSliceModel


def test_basic():
    import torch
    torch.manual_seed(42)
    model = SingleSliceModel(10, 16, 5)
    N = 4
    x = torch.randn(N, 10)
    indices = torch.tensor([[0,1,2,3],[1,2,3,0]])
    values = torch.ones(4)
    adj = torch.sparse_coo_tensor(indices, values, (N, N))
    
    out = model(x, adj)
    assert out.shape == (N, 5), f"Expected (4, 5), got {out.shape}"
    
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    print("PASS")


if __name__ == "__main__":
    test_basic()
