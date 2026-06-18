"""Solution for Lesson 10: Graph Transformer Layer."""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint


def scatter_softmax(src, index, num_nodes):
    out = torch.zeros(num_nodes, src.size(1), device=src.device, dtype=src.dtype)
    for i in range(num_nodes):
        mask = index == i
        if mask.any():
            out[i] = torch.softmax(src[mask], dim=0).sum(dim=0)
    return out


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(GraphTransformerLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        
        indices = adj._indices()
        values = adj._values()
        
        def _attn_block(x_in, indices_in, values_in):
            N = x_in.size(0)
            q = self.q_proj(x_in).view(N, self.num_heads, self.head_dim)
            k = self.k_proj(x_in).view(N, self.num_heads, self.head_dim)
            v = self.v_proj(x_in).view(N, self.num_heads, self.head_dim)
            src, dst = indices_in[0], indices_in[1]
            
            attn_scores = (q[src] * k[dst]).sum(dim=-1) / np.sqrt(self.head_dim)
            if values_in is not None:
                attn_scores = attn_scores * values_in.unsqueeze(1)
            attn_weights = scatter_softmax(attn_scores, src, num_nodes=N)
            
            out = torch.zeros(N, self.num_heads, self.head_dim, device=x_in.device, dtype=x_in.dtype)
            chunk_size = 50000
            for start in range(0, src.size(0), chunk_size):
                end = min(start + chunk_size, src.size(0))
                s, d = src[start:end], dst[start:end]
                v_chunk = v[d]
                weighted = v_chunk * attn_weights[start:end].unsqueeze(-1)
                out.scatter_add_(0, s.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.head_dim), weighted)
            
            out = out.reshape(N, self.hidden_dim)
            return self.out_proj(out)
        
        out = checkpoint(_attn_block, x, indices, values, use_reentrant=False)
        x = residual + self.dropout(out)
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x
