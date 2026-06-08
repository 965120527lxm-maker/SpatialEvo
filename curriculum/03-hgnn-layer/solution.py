"""Solution for Lesson 03: HGNN Layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super(HGNNLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x_proj = self.W(x)
        out = torch.sparse.mm(adj_norm, x_proj)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.dropout(out)
        return out
