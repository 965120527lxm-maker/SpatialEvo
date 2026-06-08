"""Solution for Lesson 05: Single-Slice Model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
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
        self.encoder = MLPEncoder(in_dim, hidden_dim, hidden_dim, dropout)
        self.hgnn = HGNNLayer(hidden_dim, hidden_dim, dropout)
        self.decoder = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.hgnn(h, adj_norm)
        out = self.decoder(h)
        return out
