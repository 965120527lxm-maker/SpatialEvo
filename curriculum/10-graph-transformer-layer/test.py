"""Tests for Lesson 10: Graph Transformer Layer."""
import sys
from pathlib import Path

starter = Path(__file__).parent / "starter.py"
if not starter.exists():
    print("FAIL: starter.py not found.")
    sys.exit(1)

from starter import GraphTransformerLayer


def test_basic():
    import torch
    torch.manual_seed(42)
    layer = GraphTransformerLayer(64, num_heads=4, dropout=0.0)
    N = 4
    x = torch.randn(N, 64)
    indices = torch.tensor([[0,1,2,3],[1,2,3,0]])
    values = torch.ones(4)
    adj = torch.sparse_coo_tensor(indices, values, (N, N))
    
    out = layer(x, adj)
    assert out.shape == (N, 64), f"Expected (4, 64), got {out.shape}"
    
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in layer.parameters())
    print("PASS")


if __name__ == "__main__":
    test_basic()
