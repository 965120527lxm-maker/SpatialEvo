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
    tree=BallTree(coords)
    distances,indices=tree.query(coords,k=k+1) # shape of indices is [N,k+1]
    # 2. Extract (src, dst) pairs, excluding self
    src=np.repeat(np.arange(N),k) # shape of src is [N*k]
    dst=indices[:,1:].flatten() # shape of dst is [N*k]
    data=[1 for _ in range(N*k)]
    # 3. Build sparse matrix
    adj=sp.csr_matrix((data,(src,dst)), shape=(N,N))
    # 4. Symmetrize
    adj=adj.T+adj
    # print(adj)
    adj=(adj>0).astype(np.float32)

    # 5. Convert to float32 csr_matrix
    return adj.tocsr()
