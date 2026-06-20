# 原始仓库 Bug 验证报告

> 为了确认本仓库记录的 bug 不是由本仓库的修改引入，我们单独 clone 了官方原始仓库到 `/root/autodl-tmp/SpatialEx_original`，并逐行核对了关键文件。以下所有 bug 均在原始代码中存在。

---

## 验证方法

```bash
cd /root/autodl-tmp/
git clone https://github.com/KEAML-JLU/SpatialEx.git SpatialEx_original
```

对比文件：
- `/root/autodl-tmp/SpatialEx_original/SpatialEx/preprocess.py`
- `/root/autodl-tmp/SpatialEx_original/SpatialEx/model.py`
- `/root/autodl-tmp/SpatialEx_original/SpatialEx/SpatialEx.py`

---

## Bug 1：`normalize_graph` 使用未定义变量 `adj`

**文件**：`SpatialEx_original/SpatialEx/preprocess.py`，第 381–408 行

**原始代码片段**（第 383–384 行）：

```python
def normalize_graph(H, edge_weight=None, norm_type='gcn'):
    if norm_type == 'row':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(1))  # ❌ adj 未定义
        adj = adj.multiply(normalization_factors)
```

**验证结果**： confirmed。函数参数为 `H`，但函数体内直接使用 `adj`，当 `norm_type` 为 `'row'`、`'col'` 或 `'both'` 时会触发 `NameError`。

**本仓库修复**：在函数开头添加 `adj = H.copy()`。

---

## Bug 2：`return_type` 拼写错误 `'crs'` → `'csr'`

**文件**：`SpatialEx_original/SpatialEx/preprocess.py`，第 335–337 行

**原始代码片段**：

```python
    if return_type == 'coo':
        if not isinstance(adj, sp.coo_matrix):
            adj = adj.tocoo()
    elif return_type == 'crs':  # ❌ 应为 'csr'
        if not isinstance(adj, sp.csr_matrix):
            adj = adj.tocsr()
```

**验证结果**：confirmed。`'crs'` 是拼写错误，调用方传入 `'csr'` 时无法进入该分支。

**本仓库修复**：改为 `elif return_type == 'csr':`。

---

## Bug 3：`Build_hypergraph_spatial_and_HE` 默认返回 `coo_matrix`

**文件**：`SpatialEx_original/SpatialEx/preprocess.py`，第 351–352 行

**原始代码片段**：

```python
def Build_hypergraph_spatial_and_HE(adata, num_neighbors=7, batch_size=4096, normalize=False, graph_kind='spatial',
                                    return_type='coo', device="cpu"):  # ❌ 默认 'coo' 不支持子图索引
```

**验证结果**：confirmed。`return_type='coo'` 导致 `Build_dataloader` 中的子图索引 `graph[self.roi_dict[name]][:, self.roi_dict[name]]` 触发 `TypeError: 'coo_matrix' object is not subscriptable`。

**本仓库修复**：默认参数改为 `return_type='csr'`。

---

## Bug 4：`Model_Plus.forward` 中 `agg_mtx` 与 `x_prime` 维度不匹配

**文件**：`SpatialEx_original/SpatialEx/model.py`，第 340–347 行

**原始代码片段**：

```python
    def forward(self, x, adj, origin_y, agg_y=None, agg_mtx=None, use_agg=True):
        x = self.mlp(x)
        h = F.leaky_relu(self.hgnn(x, adj))
        ...
        x_prime = F.leaky_relu(self.predictor(h))
        if self.platform == 'Visium' or not use_agg:
            loss = self.criterion(x_prime, origin_y)
        else:
            loss = self.criterion(torch.mm(agg_mtx, x_prime), agg_y)  # ❌ 维度不匹配
```

**验证结果**：confirmed。原始 `forward` 没有 `selection` 参数，`x_prime` 是整个 ROI 的预测结果，而 `agg_mtx` 仅覆盖 `selection` 区域，矩阵乘法会报 `RuntimeError: addmm size mismatch`。

**本仓库修复**：增加 `selection` 参数，并改为 `torch.mm(agg_mtx, x_prime[selection])`。

---

## Bug 5：`SpatialExP.train` 中 `Regression.forward` 的 `agg_mtx` 维度同样不匹配

**文件**：`SpatialEx_original/SpatialEx/SpatialEx.py`，第 449–456 行

**原始代码片段**：

```python
                    panel_2a = self.module_HA.predict(he2, graph2, grad=False)
                    panel_1b = self.module_HB.predict(he1, graph1, grad=False)

                    loss3, _ = self.rm_AB(panel_1a, panel_1b, torch.spmm(agg_mtx1, panel_1b), agg_mtx1, self.use_agg)  # ❌ panel_1b 维度不匹配
                    loss4, _ = self.rm_BA(panel_2b, panel_2a, torch.spmm(agg_mtx2, panel_2a), agg_mtx2, self.use_agg)  # ❌ panel_2a 维度不匹配

                    loss5, _ = self.rm_AB(panel_2a, panel_2b, agg_exp2, agg_mtx2, self.use_agg)  # ❌ panel_2a 维度不匹配
                    loss6, _ = self.rm_BA(panel_1b, panel_1a, agg_exp1, agg_mtx1, self.use_agg)  # ❌ panel_1b 维度不匹配
```

**验证结果**：confirmed。`panel_1b` 和 `panel_2a` 是整个 ROI 的预测结果，但 `agg_mtx1`/`agg_mtx2` 仅覆盖 `selection` 区域，导致 `torch.spmm` 维度错误。

**本仓库修复**：在传入 `rm_AB` / `rm_BA` 前，对预测结果做 `selection` 切片：

```python
loss3, _ = self.rm_AB(panel_1a, panel_1b, torch.spmm(agg_mtx1, panel_1b[selection1]), agg_mtx1, self.use_agg)
loss4, _ = self.rm_BA(panel_2b, panel_2a, torch.spmm(agg_mtx2, panel_2a[selection2]), agg_mtx2, self.use_agg)
loss5, _ = self.rm_AB(panel_2a[selection2], panel_2b, agg_exp2, agg_mtx2, self.use_agg)
loss6, _ = self.rm_BA(panel_1b[selection1], panel_1a, agg_exp1, agg_mtx1, self.use_agg)
```

---

## Bug 6：`BatchNorm1d` 在 batch_size=1 时崩溃

**文件**：`SpatialEx_original/SpatialEx/model.py`

**原始代码中 `BatchNorm1d` 出现位置**：
- 第 187 行：`Model` 的 MLP
- 第 232 行：`HyperSAGE` 的 MLP
- 第 324 行：`Model_Plus` 的 MLP
- 第 382–384 行：`Regression`
- 第 430–433 行：`Model_Big`

**原始代码片段**（以 `Model_Plus` 为例，第 322–324 行）：

```python
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.BatchNorm1d(hidden_dim))  # ❌ batch_size=1 时崩溃
```

**验证结果**：confirmed。当 dataloader 最后一个 batch 只有 1 个样本时，`BatchNorm1d` 无法计算 batch 统计量，会触发：

```
ValueError: Expected more than 1 value per channel when training
```

**本仓库修复**：将 `Model_Plus` 和 `Regression` 中的 `BatchNorm1d` 替换为 `LayerNorm`。

---

## 结论

以上 6 个 bug 均可在 `/root/autodl-tmp/SpatialEx_original/` 的官方原始代码中直接定位到。它们不是本仓库引入的修改，而是官方实现中存在的阻塞性错误。修复这些错误是本仓库能够复现 SpatialEx/SpatialEx+ 的前提。
