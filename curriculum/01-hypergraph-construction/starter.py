"""Starter code for Lesson 01: Hypergraph Construction.

Implement the function below so that all tests pass.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import BallTree


def build_knn_graph(coords: np.ndarray, k: int = 7) -> sp.csr_matrix:
    """
    Build a k-NN graph from spatial coordinates.
    
    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Spatial coordinates of N cells.
    k : int
        Number of nearest neighbors.
    
    Returns
    -------
    adj : scipy.sparse.csr_matrix, shape (N, N)
        Symmetric adjacency matrix. adj[i, j] = 1 if i and j are neighbors.
    """
    N = coords.shape[0]
    # YOUR CODE HERE
    # 1. Build BallTree and query k+1 neighbors (including self)
    # 2. Extract (src, dst) pairs, excluding self
    # 3. Build sparse matrix
    # 4. Symmetrize
    # 5. Convert to float32 csr_matrix
    
    raise NotImplementedError
