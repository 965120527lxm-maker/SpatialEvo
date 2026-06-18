"""Solution for Lesson 01: Hypergraph Construction."""
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import BallTree


def build_knn_graph(coords: np.ndarray, k: int = 7) -> sp.csr_matrix: # 首先, 这里返回的稀疏矩阵
    N = coords.shape[0]
    tree = BallTree(coords)
    distances, indices = tree.query(coords, k=k + 1)
    # indices 是邻接表(链表) 便于查邻居, 不方便进行GNN聚合,但是adj方便图卷积,消息传递 
    # Source nodes: each node repeated k times
    src = np.repeat(np.arange(N), k)
    # Target nodes: flatten neighbor indices, skipping self (first column)
    dst = indices[:, 1:].flatten()
    
    # 稀疏矩阵就是有很多0的矩阵, 因此只需要存非零元素的位置就好了
    # 那么存放的方法就是: value, i ,j , 分别用三个一维数组对齐就好了. 
    data = np.ones_like(src, dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(N, N)) # value: 1 , i: src, j :dst
    
    # Symmetrize
    adj = adj + adj.T 
    adj = (adj > 0).astype(np.float32) # 为何需要统一成1? 会出现非1的吗?   
    
    return adj.tocsr()
