"""
Improved SpatialEx model using Graph Transformer and Masked Feature Prediction.

This module provides an alternative architecture that:
1. Replaces HGNN with Graph Transformer for adaptive neighborhood aggregation
2. Replaces DGI contrastive learning with Masked Feature Prediction self-supervision
3. Uses Cross-Attention based Omics Translator in SpatialEx+

Key design choice: Neighbor-sparse attention (O(N*k) instead of O(N^2))
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .utils import create_activation


def scatter_softmax(src, index, num_nodes=None):
    """
    Sparse softmax along index dimension.
    src: [E, H] attention scores
    index: [E] row indices (source nodes)
    """
    if num_nodes is None:
        num_nodes = index.max().item() + 1
    
    # For numerical stability, subtract max per row
    # Use scatter_max equivalent
    max_per_node = torch.full((num_nodes, src.size(1)), float('-inf'), device=src.device, dtype=src.dtype)
    max_per_node = max_per_node.scatter_reduce(0, index.unsqueeze(1).expand(-1, src.size(1)), src, reduce='amax', include_self=False)
    max_per_node = torch.where(torch.isinf(max_per_node), torch.zeros_like(max_per_node), max_per_node)
    
    exp_src = torch.exp(src - max_per_node[index])
    sum_exp = torch.zeros(num_nodes, src.size(1), device=src.device, dtype=src.dtype)
    sum_exp = sum_exp.scatter_add(0, index.unsqueeze(1).expand(-1, src.size(1)), exp_src)
    
    out = exp_src / (sum_exp[index] + 1e-8)
    return out


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer with sparse neighbor attention.
    
    Only computes attention between each node and its neighbors,
    achieving O(N * k) complexity instead of O(N^2).
    """
    
    def __init__(self, in_dim, hidden_dim, num_heads=8, dropout=0.1, activation='prelu'):
        super(GraphTransformerLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        
        self.q_proj = nn.Linear(in_dim, hidden_dim)
        self.k_proj = nn.Linear(in_dim, hidden_dim)
        self.v_proj = nn.Linear(in_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            create_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = create_activation(activation)
        
    def forward(self, x, adj, return_attn=False):
        """
        x: [N, in_dim]
        adj: sparse adjacency matrix [N, N] (torch sparse tensor)
               neighbors have value 1 (or any positive weight)
        """
        residual = x
        x = self.norm1(x)
        
        N = x.size(0)
        
        # Extract edges from sparse adjacency outside checkpoint so that
        # the checkpointed function only receives dense tensors.
        if adj.is_sparse:
            indices = adj._indices()  # [2, nnz]
            values = adj._values()
        else:
            # If dense (should not happen in our pipeline), convert
            indices = adj.nonzero().t()
            values = adj[indices[0], indices[1]]
        
        # Add self-loops explicitly
        eye_indices = torch.arange(N, device=x.device)
        self_loop_idx = torch.stack([eye_indices, eye_indices], dim=0)
        indices = torch.cat([indices, self_loop_idx], dim=1)
        if values.numel() > 0:
            values = torch.cat([values, torch.ones(N, device=x.device, dtype=values.dtype)], dim=0)
        else:
            values = torch.ones(indices.size(1), device=x.device, dtype=x.dtype)
        
        def _attn_block(x_in, indices_in, values_in):
            """Memory-efficient attention block wrapped by gradient checkpointing."""
            N_in = x_in.size(0)
            q = self.q_proj(x_in).view(N_in, self.num_heads, self.head_dim)
            k = self.k_proj(x_in).view(N_in, self.num_heads, self.head_dim)
            v = self.v_proj(x_in).view(N_in, self.num_heads, self.head_dim)
            
            src = indices_in[0]  # [E]
            dst = indices_in[1]  # [E]
            num_edges = src.size(0)
            chunk_size = 50000
            
            # -----------------------------------------------------------------
            # 1) Compute attention scores in edge chunks.  This avoids
            #    materialising the full [E, H, D] q[src] / k[dst] tensors at
            #    once; peak memory for this phase is only ~3*[chunk, H, D].
            # -----------------------------------------------------------------
            attn_scores = torch.empty(num_edges, self.num_heads, device=x_in.device, dtype=x_in.dtype)
            for start in range(0, num_edges, chunk_size):
                end = min(start + chunk_size, num_edges)
                attn_scores[start:end] = (
                    q[src[start:end]] * k[dst[start:end]]
                ).sum(dim=-1) / np.sqrt(self.head_dim)
            
            # Apply edge weights if available (optional)
            if values_in is not None and values_in.numel() > 0:
                attn_scores = attn_scores * values_in.unsqueeze(1)  # [E, H]
            
            # Sparse softmax per source node (needs the full [E, H] score vector)
            attn_weights = scatter_softmax(attn_scores, src, num_nodes=N_in)  # [E, H]
            del attn_scores
            
            # -----------------------------------------------------------------
            # 2) Aggregate values in edge chunks so v[dst] also never exists
            #    in its full [E, H, D] form.
            # -----------------------------------------------------------------
            out = torch.zeros(N_in, self.num_heads, self.head_dim, device=x_in.device, dtype=x_in.dtype)
            for start in range(0, num_edges, chunk_size):
                end = min(start + chunk_size, num_edges)
                src_chunk = src[start:end]
                dst_chunk = dst[start:end]
                weights_chunk = attn_weights[start:end]
                v_chunk = v[dst_chunk]
                weighted_v_chunk = v_chunk * weights_chunk.unsqueeze(-1)
                out.scatter_add_(
                    0,
                    src_chunk.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.head_dim),
                    weighted_v_chunk
                )
            
            out = out.reshape(N_in, self.hidden_dim)
            out = self.out_proj(out)
            out = self.dropout(out)
            return out
        
        # Gradient checkpointing trades a small amount of compute for a large
        # reduction in activation memory inside the attention block.
        out = checkpoint(_attn_block, x, indices, values, use_reentrant=False)
        x = residual + out
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        if return_attn:
            return x, None
        return x


class GraphTransformerEncoder(nn.Module):
    """Stack of Graph Transformer layers."""
    
    def __init__(self, in_dim, hidden_dim, num_layers=2, num_heads=8, dropout=0.1, activation='prelu'):
        super(GraphTransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim if i > 0 else hidden_dim,
                hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation
            ) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = create_activation(activation)
        
    def forward(self, x, adj):
        x = self.input_proj(x)
        x = self.activation(x)
        for layer in self.layers:
            x = layer(x, adj)
        x = self.norm(x)
        return x


class MaskedFeaturePrediction(nn.Module):
    """Masked feature prediction for self-supervised learning."""
    
    def __init__(self, hidden_dim, mask_ratio=0.15):
        super(MaskedFeaturePrediction, self).__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.criterion = nn.MSELoss()
        
    def forward(self, h, he_input):
        """
        h: [N, hidden_dim] encoded features
        he_input: [N, he_dim] original H&E embeddings (target)
        """
        N = h.shape[0]
        num_mask = max(1, int(N * self.mask_ratio))
        mask_indices = torch.randperm(N, device=h.device)[:num_mask]
        
        pred = self.decoder(h[mask_indices])
        target = he_input[mask_indices]
        
        # Simple projection if dimensions differ
        if pred.shape[-1] != target.shape[-1]:
            target_proj = nn.Linear(target.shape[-1], pred.shape[-1], device=h.device)
            target = target_proj(target)
        
        loss = self.criterion(pred, target)
        return loss


class DGIEmbeddingLoss(nn.Module):
    """DGI-style contrastive loss on encoder node embeddings."""

    def __init__(self):
        super().__init__()
        self.b_xent = nn.CosineEmbeddingLoss()

    def forward(self, h_pos, h_neg):
        n = h_pos.shape[0]
        c = torch.mean(h_pos, dim=0, keepdim=True)
        dev = h_pos.device
        lbl_pos = torch.ones(n, device=dev)
        lbl_neg = -torch.ones(n, device=dev)
        return (
            self.b_xent(h_pos, c.expand_as(h_pos), lbl_pos)
            + self.b_xent(h_neg, c.expand_as(h_neg), lbl_neg)
        )


class Model_GT(nn.Module):
    """Graph Transformer based SpatialEx model."""
    
    def __init__(self,
                 num_layers=2,
                 in_dim=2048,
                 hidden_dim=512,
                 out_dim=150,
                 loss_fn="mse",
                 device="cpu",
                 num_heads=8,
                 dropout=0.1,
                 use_mfp=True,
                 use_dgi=False,
                 mfp_weight=0.1,
                 dgi_weight=1.0,
                 mask_ratio=0.15):
        super(Model_GT, self).__init__()

        if use_mfp and use_dgi:
            raise ValueError("Model_GT: use_mfp and use_dgi are mutually exclusive")

        self.use_mfp = use_mfp
        self.use_dgi = use_dgi
        self.mfp_weight = mfp_weight
        self.dgi_weight = dgi_weight
        self.device = device
        
        self.encoder = GraphTransformerEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )
        
        if self.use_mfp:
            self.mfp = MaskedFeaturePrediction(hidden_dim, mask_ratio=mask_ratio)
        if self.use_dgi:
            self.dgi_loss = DGIEmbeddingLoss()
        
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_fn}")
            
    def forward(self, graph, he_rep, exp, agg_mtx=None, selection=None):
        enc = self.encoder(he_rep, graph)
        x_prime = self.predictor(enc)
        
        if agg_mtx is not None and selection is not None:
            loss = self.criterion(torch.sparse.mm(agg_mtx, x_prime[selection]), exp)
        else:
            loss = self.criterion(x_prime, exp)
        
        if self.use_mfp:
            loss = loss + self.mfp_weight * self.mfp(enc, he_rep)
        elif self.use_dgi:
            idx = torch.randperm(he_rep.shape[0], device=he_rep.device)
            enc_corrupt = self.encoder(he_rep[idx], graph)
            loss = loss + self.dgi_weight * self.dgi_loss(enc, enc_corrupt)

        return loss, x_prime
    
    def predict(self, he_representations, graph, grad=False):
        if not grad:
            with torch.no_grad():
                enc = self.encoder(he_representations, graph)
                x_prime = self.predictor(enc)
        else:
            enc = self.encoder(he_representations, graph)
            x_prime = self.predictor(enc)
        return x_prime


class CrossAttentionTranslator(nn.Module):
    """Lightweight omics translator for SpatialEx+.
    
    The original design used a full quadratic self-attention over all cells,
    which is infeasible for hundred-thousand-cell Xenium slices.  We keep the
    same interface but replace the O(N^2) MultiheadAttention with a shallow
    MLP projector so training scales to full tissue sections.
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, dropout=0.1):
        super(CrossAttentionTranslator, self).__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_dim)
        )
        
        self.criterion = nn.MSELoss()
        
    def forward(self, x, origin_y=None, agg_y=None, agg_mtx=None, use_agg=True):
        """
        x: source omics features
        origin_y: target omics at cell level
        agg_y: target omics at spot level
        """
        x_proj = self.input_proj(x)
        x_proj = self.norm1(x_proj)
        out = self.ffn(x_proj)
        
        if origin_y is None and agg_y is None:
            return out
        
        if agg_mtx is not None and use_agg:
            if agg_mtx.is_sparse:
                loss = self.criterion(torch.sparse.mm(agg_mtx, out), agg_y)
            else:
                loss = self.criterion(torch.mm(agg_mtx, out), agg_y)
        else:
            loss = self.criterion(out, origin_y)
        
        return loss, out
    
    def predict(self, x, grad=False):
        if not grad:
            with torch.no_grad():
                x_proj = self.input_proj(x)
                x_proj = self.norm1(x_proj)
                out = self.ffn(x_proj)
        else:
            x_proj = self.input_proj(x)
            x_proj = self.norm1(x_proj)
            out = self.ffn(x_proj)
        return out


class Model_Plus_GT(nn.Module):
    """Graph Transformer based SpatialEx+ model."""
    
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 activation: str = 'prelu',
                 use_mfp: bool = True,
                 loss_fn: str = 'mse',
                 platform: str = 'Xenium'):
        super(Model_Plus_GT, self).__init__()
        
        self.platform = platform
        self.use_mfp = use_mfp
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        self.encoder = GraphTransformerEncoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )
        
        self.predictor = nn.Linear(hidden_dim, out_dim)
        
        if self.use_mfp:
            self.mfp = MaskedFeaturePrediction(hidden_dim, mask_ratio=0.15)
        
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_fn}")
    
    def forward(self, x, adj, origin_y, agg_y=None, agg_mtx=None, use_agg=True, selection=None):
        x = self.mlp(x)
        h = self.encoder(x, adj)
        x_prime = F.leaky_relu(self.predictor(h))
        
        if self.platform == 'Visium' or not use_agg:
            loss = self.criterion(x_prime, origin_y)
        else:
            if selection is not None:
                loss = self.criterion(torch.mm(agg_mtx, x_prime[selection]), agg_y)
            else:
                loss = self.criterion(torch.mm(agg_mtx, x_prime), agg_y)
        
        if self.use_mfp:
            mfp_loss = self.mfp(h, x)
            loss = loss + 0.1 * mfp_loss
        
        return loss, x_prime
    
    def predict(self, x, adj, grad=False):
        if not grad:
            with torch.no_grad():
                x = self.mlp(x)
                h = self.encoder(x, adj)
                x_prime = F.leaky_relu(self.predictor(h))
        else:
            x = self.mlp(x)
            h = self.encoder(x, adj)
            x_prime = F.leaky_relu(self.predictor(h))
        return x_prime
