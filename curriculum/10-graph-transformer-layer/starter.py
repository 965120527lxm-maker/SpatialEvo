"""Starter code for Lesson 10: Graph Transformer Layer."""
import torch
import torch.nn as nn
import numpy as np


def scatter_softmax(src, index, num_nodes):
    """Sparse softmax over edges."""
    out = torch.zeros(num_nodes, src.size(1), device=src.device, dtype=src.dtype)
    for i in range(num_nodes):
        mask = index == i
        if mask.any():
            out[i] = torch.softmax(src[mask], dim=0).sum(dim=0)
    return out


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(GraphTransformerLayer, self).__init__()
        # YOUR CODE HERE
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        raise NotImplementedError
