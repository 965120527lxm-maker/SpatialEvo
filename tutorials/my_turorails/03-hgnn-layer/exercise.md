# 第 3 课：HGNN 层（Hypergraph Neural Network Layer）

## 目标

实现单层超图神经网络的消息传递。

这是整个模型的**核心引擎**。所有空间信息的学习都发生在这里：每个细胞通过邻居聚合更新自己的表示。

## 输入

- `x`: `torch.Tensor`，形状 `(N, in_dim)`，节点特征
- `adj_norm`: `torch.Tensor`，稀疏格式 `torch.sparse.FloatTensor`，形状 `(N, N)`，归一化邻接矩阵

## 输出

- `out`: `torch.Tensor`，形状 `(N, out_dim)`，更新后的节点特征

## 核心任务

实现 `HGNNLayer` 类：

```python
class HGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.dropout=dropout
        # YOUR CODE HERE
    
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
        # 1. build hypergraph

        # 2. message passing

        # 3. return x
        w=random((indim,outdim))
        out=np.sigma(out=adj_norm @ x @ w)
        return out
```

## 数学公式

单层 HGNN 的前向传播：

$$
h^{(l+1)} = \sigma(D^{-0.5} H D^{-0.5} \cdot h^{(l)} \cdot W^{(l)})
$$

其中：
- $D^{-0.5} H D^{-0.5}$ 就是输入的 `adj_norm`
- $h^{(l)}$ 是输入特征 `x`
- $W^{(l)}` 是可学习的线性投影
- $\sigma$ 是激活函数（LeakyReLU）

简化为代码：
```python
out = torch.sparse.mm(adj_norm, self.W(x))
out = F.leaky_relu(out)
```

## 约束

- `self.W` 必须是 `nn.Linear(in_dim, out_dim)`
- 使用 `torch.sparse.mm` 进行稀疏矩阵乘法
- 应用 `F.leaky_relu(0.1)` 激活函数
- （可选）应用 `nn.Dropout(dropout)`

## 提示

1. **稀疏矩阵乘法**：
   ```python
   out = torch.sparse.mm(adj_norm, x_transformed)
   ```
   注意：`torch.sparse.mm(sparse, dense)` 返回 dense tensor。

2. **顺序**：
   - 先通过 `W` 投影特征：`x_proj = self.W(x)` → `(N, out_dim)`
   - 再通过邻接矩阵聚合：`out = torch.sparse.mm(adj_norm, x_proj)` → `(N, out_dim)`

3. **激活函数**：
   ```python
   out = F.leaky_relu(out, negative_slope=0.1)
   ```

4. **Dropout**（可选）：
   ```python
   out = self.dropout(out)
   ```

## 测试说明

运行 `python test.py`，它会检查：
- 输出形状是否为 `(N, out_dim)`
- 输出是否包含非零值（激活函数生效）
- 输入全零时输出是否为零（线性层的零输入性质）

## 参考输出

```python
>>> layer = HGNNLayer(64, 128)
>>> x = torch.randn(100, 64)
>>> adj = torch.sparse_coo_tensor(indices, values, (100, 100))
>>> out = layer(x, adj)
>>> out.shape
torch.Size([100, 128])
>>> out.min() < 0  # LeakyReLU 允许负值
True
```
