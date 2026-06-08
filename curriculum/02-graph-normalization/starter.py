"""Starter code for Lesson 02: Graph Normalization."""
import numpy as np
import scipy.sparse as sp


def normalize_graph(adj: sp.csr_matrix, norm_type: str = 'gcn') -> sp.csr_matrix:
    """
    Normalize an adjacency matrix.
    
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix, shape (N, N)
        Unnormalized symmetric adjacency matrix.
    norm_type : str
        'gcn'  → D^{-0.5} A D^{-0.5}
        'hpnn' → DV @ H @ W @ DE @ H.T @ DV
    
    Returns
    -------
    norm_adj : scipy.sparse.csr_matrix, shape (N, N)
        Normalized adjacency matrix.
    """
    # YOUR CODE HERE
    if norm_type=="gcn":
        dim=adj.shape
        adj_=adj+np.eye(dim[0])
        norm_adj=adj

    raise NotImplementedError
