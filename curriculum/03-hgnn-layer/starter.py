"""Starter code for Lesson 03: HGNN Layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNNLayer(nn.Module):
    """Single layer of Hypergraph Neural Network."""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super(HGNNLayer, self).__init__()
        # YOUR CODE HERE
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Hypergraph message passing.
        
        Parameters
        ----------
        x : torch.Tensor, shape (N, in_dim)
            Node features.
        adj_norm : torch.Tensor, sparse, shape (N, N)
            Normalized adjacency matrix.
        
        Returns
        -------
        out : torch.Tensor, shape (N, out_dim)
            Updated node features.
        """
        # YOUR CODE HERE
        raise NotImplementedError
