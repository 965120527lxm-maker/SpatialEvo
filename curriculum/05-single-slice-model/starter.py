"""Starter code for Lesson 05: Single-Slice Model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """Minimal MLP encoder for self-contained lesson."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class HGNNLayer(nn.Module):
    """Minimal HGNN layer for self-contained lesson."""
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


class SingleSliceModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super(SingleSliceModel, self).__init__()
        # YOUR CODE HERE
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        raise NotImplementedError
