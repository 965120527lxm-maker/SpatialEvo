"""Tests for Lesson 03: HGNN Layer."""
import sys
from pathlib import Path

starter = Path(__file__).parent / "starter.py"
if not starter.exists():
    print("FAIL: starter.py not found. Please complete the exercise first.")
    sys.exit(1)

from starter import HGNNLayer


def test_basic():
    import torch
    torch.manual_seed(42)
    layer = HGNNLayer(16, 32, dropout=0.0)
    
    N = 4
    x = torch.randn(N, 16)
    indices = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    values = torch.ones(4)
    adj_norm = torch.sparse_coo_tensor(indices, values, (N, N))
    
    out = layer(x, adj_norm)
    assert out.shape == (N, 32), f"Expected shape (4, 32), got {out.shape}"
    print("PASS")


if __name__ == "__main__":
    test_basic()
