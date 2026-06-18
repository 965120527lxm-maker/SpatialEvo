"""Solution for Lesson 02: Graph Normalization."""
import numpy as np
import scipy.sparse as sp


def normalize_graph(adj: sp.csr_matrix, norm_type: str = 'gcn') -> sp.csr_matrix:
    if norm_type == 'gcn':
        D = np.array(adj.sum(axis=1)).flatten()
        D_inv_sqrt = sp.diags(np.power(D + 1e-8, -0.5))
        return D_inv_sqrt @ adj @ D_inv_sqrt
    elif norm_type == 'hpnn':
        # For a symmetric adjacency matrix, HPNN simplifies to GCN-like normalization
        D = np.array(adj.sum(axis=1)).flatten()
        DV = sp.diags(np.power(D + 1e-8, -0.5))
        DE = sp.diags(np.power(D + 1e-8, -1.0))
        W = sp.eye(adj.shape[0], format='csr')
        return DV @ adj @ W @ DE @ adj.T @ DV
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
