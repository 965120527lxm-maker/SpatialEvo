# 第 1 课：超图构建（Hypergraph Construction）

## 目标

给定 N 个细胞的空间坐标，构建一个 k-NN 邻接矩阵（超图）。

这是整个系统的**地基**：没有图，就没有邻居聚合；没有邻居聚合，模型就只能看单个细胞，无法利用空间信息。

## 输入

- `coords`: `np.ndarray`，形状 `(N, 2)`，每行是一个细胞的 `(x, y)` 空间坐标
- `k`: 整数，每个细胞的邻居数（默认 7）

## 输出

- `adj`: `scipy.sparse.csr_matrix`，形状 `(N, N)`
  - 如果细胞 i 是细胞 j 的 k 近邻之一，则 `adj[i, j] = 1`
  - 矩阵应是对称的（双向边）
  - 对角线为 0（无自环，后续会单独添加）

## 核心任务

实现函数：

```python
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
    # YOUR CODE HERE
    pass
```

## 约束

- 使用 `sklearn.neighbors.BallTree` 或 `scipy.spatial.KDTree` 找邻居
- 邻接矩阵使用 **CSR 格式**（`scipy.sparse.csr_matrix`）
- 矩阵必须是对称的：如果 i→j 有边，j→i 也必须有边
- 不要添加自环（对角线为 0）

## 提示

1. **找邻居**：
   ```python
   from sklearn.neighbors import BallTree
   tree = BallTree(coords)
   distances, indices = tree.query(coords, k=k+1)  # k+1 因为包含自身
   ```
   `indices` 形状是 `(N, k+1)`，每行是最近邻的索引。

2. **构造稀疏矩阵**：
   - 从 `indices` 中提取所有 `(src, dst)` 对
   - `src` 是源节点索引（重复 N 次，每次 k 个）
   - `dst` 是目标节点索引（从 indices 中来，去掉自身）
   - 使用 `np.repeat` 和 `np.tile` 构造 row/col indices

3. **对称化**：
   ```python
   adj = adj + adj.T
   adj = adj > 0  # 去重
   adj = adj.astype(np.float32)
   ```

4. **CSR 格式**：
   ```python
   adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
   ```

## 测试说明

运行 `python test.py`，它会检查：
- 输出是否为 `csr_matrix`
- 形状是否为 `(N, N)`
- 是否对称
- 对角线是否为 0
- 每个节点是否有 `2k` 个非零元素（因为双向边，实际可能略少）

## 参考输出

```python
>>> coords = np.array([[0, 0], [1, 0], [2, 0], [0, 1]])
>>> adj = build_knn_graph(coords, k=2)
>>> adj.shape
(4, 4)
>>> adj.toarray()
array([[0., 1., 0., 1.],
       [1., 0., 1., 1.],
       [0., 1., 0., 1.],
       [1., 1., 1., 0.]], dtype=float32)
```
