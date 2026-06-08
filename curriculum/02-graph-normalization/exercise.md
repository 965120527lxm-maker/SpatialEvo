# 第 2 课：图归一化（Graph Normalization）

## 目标

对邻接矩阵应用 **GCN 对称归一化**：`D^(-0.5) @ A @ D^(-0.5)`。

这是消息传递能够**稳定收敛**的关键。没有归一化，度数高的节点会累积过大的信号，导致梯度爆炸；度数低的节点信号过弱，导致梯度消失。

## 输入

- `adj`: `scipy.sparse.csr_matrix`，形状 `(N, N)`，来自第 1 课
- `norm_type`: 字符串，`'gcn'` 或 `'hpnn'`

## 输出

- `norm_adj`: `scipy.sparse.csr_matrix`，形状 `(N, N)`，归一化后的邻接矩阵

## 核心任务

实现函数：

```python
def normalize_graph(adj: sp.csr_matrix, norm_type: str = 'gcn') -> sp.csr_matrix:
    """
    Normalize an adjacency matrix.
    
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix, shape (N, N)
        Unnormalized symmetric adjacency matrix.
    norm_type : str
        'gcn'  → D^{-0.5} A D^{-0.5}  (GCN normalization)
        'hpnn' → DV @ H @ W @ DE @ H.T @ DV  (Hypergraph Neural Network normalization)
    
    Returns
    -------
    norm_adj : scipy.sparse.csr_matrix, shape (N, N)
        Normalized adjacency matrix.
    """
    # YOUR CODE HERE
    pass
```

## 约束

- **GCN 归一化**（`'gcn'`）必须实现：
  1. 计算度矩阵 `D = adj.sum(axis=1).A.flatten()`
  2. 计算 `D^{-0.5} = diag((D + 1e-8)^{-0.5})`
  3. 返回 `D^{-0.5} @ adj @ D^{-0.5}`

- **HPNN 归一化**（`'hpnn'`）可以简化为与 GCN 相同（对于对称邻接矩阵，两者等价），或实现完整的 HPNN 公式。

- 使用 `scipy.sparse.diags` 构造对角矩阵
- 所有运算在 sparse 矩阵上进行，不要转为 dense

## 提示

1. **计算度**：
   ```python
   D = np.array(adj.sum(axis=1)).flatten()  # shape (N,)
   ```

2. **构造 D^{-0.5}**：
   ```python
   D_inv_sqrt = sp.diags(np.power(D + 1e-8, -0.5))
   ```

3. **稀疏矩阵乘法**：
   ```python
   norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
   ```

4. **注意**：`@` 运算符对 scipy sparse matrix 有效，等同于 `.dot()`

## 测试说明

运行 `python test.py`，它会检查：
- 输出是否为 `csr_matrix`
- 形状是否为 `(N, N)`
- GCN 归一化后每行的和是否近似为 1

## 参考输出

```python
>>> adj = csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
>>> norm = normalize_graph(adj, 'gcn')
>>> norm.toarray()
array([[0.        , 0.5       , 0.        ],
       [0.5       , 0.        , 0.5       ],
       [0.        , 0.5       , 0.        ]])
```
