# SpatialEx 官方代码 Bug 修复记录

## 概述

在复现 `KEAML-JLU/SpatialEx` 官方代码的过程中，我们在 `preprocess.py`、`model.py` 和 `SpatialEx.py` 中发现了 5 个会导致运行时错误的 bug，并进行了修复。

---

## Bug 1: `normalize_graph` 中使用了未定义变量 `adj`

**文件**: `SpatialEx/preprocess.py`  
**函数**: `normalize_graph`  
**严重性**: 🔴 高（直接导致 NameError 或错误结果）

### 问题描述

函数签名接受参数 `H`，但函数体内大量使用了未定义的变量 `adj`：

```python
def normalize_graph(H, edge_weight=None, norm_type='gcn'):
    if norm_type == 'row':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(1))  # ❌ adj 未定义
        adj = adj.multiply(normalization_factors)
    elif norm_type == 'col':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(0))  # ❌ adj 未定义
        adj = adj.multiply(normalization_factors)
    ...
```

### 修复

在函数开头添加 `adj = H.copy()`，统一使用 `adj`：

```python
def normalize_graph(H, edge_weight=None, norm_type='gcn'):
    adj = H.copy()  # ✅ 修复
    if norm_type == 'row':
        normalization_factors = sp.csr_matrix(1.0 / adj.sum(1))
        adj = adj.multiply(normalization_factors)
    ...
```

---

## Bug 2: `return_type` 拼写错误 `'crs'` → `'csr'`

**文件**: `SpatialEx/preprocess.py`  
**位置**: `Build_graph` 和 `Build_hypergraph_spatial_and_HE`  
**严重性**: 🟡 中（导致无法正确返回 CSR 格式）

### 问题描述

在两处代码中，`return_type` 的检查将 `'csr'` 误写为 `'crs'`：

```python
elif return_type == 'crs':  # ❌ 拼写错误
    if not isinstance(adj, sp.csr_matrix):
        adj = adj.tocsr()
```

### 修复

将 `'crs'` 更正为 `'csr'`：

```python
elif return_type == 'csr':  # ✅ 修复
    if not isinstance(adj, sp.csr_matrix):
        adj = adj.tocsr()
```

---

## Bug 3: `Build_hypergraph_spatial_and_HE` 默认返回 `coo_matrix`，但子图索引需要 `csr_matrix`

**文件**: `SpatialEx/preprocess.py`  
**函数**: `Build_hypergraph_spatial_and_HE`  
**严重性**: 🔴 高（直接导致 TypeError）

### 问题描述

该函数默认参数 `return_type='coo'`，但 `Build_dataloader` 中需要对其进行子图索引：

```python
sub_graph = normalize_graph(graph[self.roi_dict[name]][:, self.roi_dict[name]], ...)
# => TypeError: 'coo_matrix' object is not subscriptable
```

### 修复

将默认参数改为 `return_type='csr'`：

```python
def Build_hypergraph_spatial_and_HE(..., return_type='csr', ...):  # ✅ 修复
```

---

## Bug 4: `SpatialExP.train()` 中 `Model_Plus.forward` 的 `agg_mtx` 维度与 `x_prime` 维度不匹配

**文件**: `SpatialEx/SpatialEx.py` + `SpatialEx/model.py`  
**函数**: `SpatialExP.train` / `Model_Plus.forward`  
**严重性**: 🔴 高（直接导致 RuntimeError: addmm size mismatch）

### 问题描述

在 `Model_Plus.forward` 中：

```python
def forward(self, x, adj, origin_y, agg_y=None, agg_mtx=None, use_agg=True):
    ...
    x_prime = F.leaky_relu(self.predictor(h))
    if self.platform == 'Visium' or not use_agg:
        loss = self.criterion(x_prime, origin_y)
    else:
        loss = self.criterion(torch.mm(agg_mtx, x_prime), agg_y)  # ❌ 维度不匹配
```

- `x_prime` 的 shape 是 `[roi_cells, out_dim]`（整个 ROI 区域）
- `agg_mtx` 的 shape 是 `[n_spots, selection_cells]`（仅中心区域）
- `roi_cells != selection_cells`，导致矩阵乘法失败

### 修复

1. 在 `Model_Plus.forward` 中增加 `selection` 参数：

```python
def forward(self, x, adj, origin_y, agg_y=None, agg_mtx=None, use_agg=True, selection=None):
    ...
    if selection is not None:
        loss = self.criterion(torch.mm(agg_mtx, x_prime[selection]), agg_y)
    else:
        loss = self.criterion(torch.mm(agg_mtx, x_prime), agg_y)
```

2. 在 `SpatialExP.train` 中传入 `selection`：

```python
selection1 = data1[0]['selection']
selection2 = data2[0]['selection']
loss1, _ = self.module_HA(he1, graph1, panel_1a, agg_exp1, agg_mtx1, self.use_agg, selection1)
loss2, _ = self.module_HB(he2, graph2, panel_2b, agg_exp2, agg_mtx2, self.use_agg, selection2)
```

---

## Bug 5: `SpatialExP.train()` 中 `Regression.forward` 的 `agg_mtx` 维度同样不匹配

**文件**: `SpatialEx/SpatialEx.py`  
**函数**: `SpatialExP.train`  
**严重性**: 🔴 高（直接导致 RuntimeError: addmm size mismatch）

### 问题描述

在计算 `loss3, loss4, loss5, loss6` 时，传入的 `panel_1b`, `panel_2a` 是整个 ROI 的预测结果，但 `agg_mtx` 仅覆盖 `selection` 区域：

```python
loss3, _ = self.rm_AB(panel_1a, panel_1b, torch.spmm(agg_mtx1, panel_1b), agg_mtx1, self.use_agg)
# ❌ panel_1b 是 [roi_cells, genes]，但 agg_mtx1 列数是 selection_cells
```

### 修复

在传入 `rm_AB` / `rm_BA` 之前，先对预测结果做 `selection` 切片：

```python
loss3, _ = self.rm_AB(panel_1a, panel_1b, torch.spmm(agg_mtx1, panel_1b[selection1]), agg_mtx1, self.use_agg)
loss4, _ = self.rm_BA(panel_2b, panel_2a, torch.spmm(agg_mtx2, panel_2a[selection2]), agg_mtx2, self.use_agg)
loss5, _ = self.rm_AB(panel_2a[selection2], panel_2b, agg_exp2, agg_mtx2, self.use_agg)
loss6, _ = self.rm_BA(panel_1b[selection1], panel_1a, agg_exp1, agg_mtx1, self.use_agg)
```

---

## 修复验证

所有修复已通过以下测试脚本验证：

```bash
python test_spatialex.py       # 官方代码跑通测试
python test_comparison.py      # 改进算法对比测试
```

测试环境：
- Python 3.10
- PyTorch 2.12 (CPU)
- numpy 1.26, anndata 0.8, scanpy 1.9
