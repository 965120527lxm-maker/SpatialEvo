"""Solution for Lesson 01: Hypergraph Construction."""
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import BallTree


def build_knn_graph(coords: np.ndarray, k: int = 7) -> sp.csr_matrix:
    N = coords.shape[0]
    tree = BallTree(coords)
    distances, indices = tree.query(coords, k=k + 1)
    
    # Source nodes: each node repeated k times
    src = np.repeat(np.arange(N), k)
    # Target nodes: flatten neighbor indices, skipping self (first column)
    dst = indices[:, 1:].flatten()
    
    data = np.ones_like(src, dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(N, N))
    
    # Symmetrize
    adj = adj + adj.T
    adj = (adj > 0).astype(np.float32)
    
    return adj.tocsr()
