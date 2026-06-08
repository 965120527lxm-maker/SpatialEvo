# A100 服务器运行指南

> 本文档用于在 A100 GPU 服务器上复现 SpatialEx 并验证改进算法效果。

---

## 1. 环境准备

```bash
# 1.1 克隆代码（或直接用已上传的代码）
git clone <你的GitHub仓库> /root/autodl-tmp/SpatialEx
cd /root/autodl-tmp/SpatialEx

# 1.2 创建 conda 环境
conda create -n spatialex python=3.10 -y
conda activate spatialex

# 1.3 安装依赖（先装 PyTorch GPU 版）
conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install anndata==0.8.0 scanpy==1.9.3 pandas==2.0.3 scikit-learn==1.3.2
pip install scipy==1.10.1 tqdm cellpose==3.0.10 timm==1.0.8 huggingface-hub==0.24.6 transformers

# 1.4 安装 SpatialEx
pip install -e .
```

---

## 2. 数据准备

将下载的两个切片数据放到 `/root/autodl-tmp/data/`：

```
/root/autodl-tmp/data/
├── slice1_uni.h5ad    # Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad
└── slice2_uni.h5ad    # Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad
```

> 数据路径建议用 `/root/autodl-tmp/data/` 而非 `/dataset`，因为 `data/` 是 ML 项目的标准命名。

---

## 3. Bug 修复清单（已修复）

在官方代码基础上修复了 **6 个 bug**：

| #   | 位置                                              | 问题                                                      | 修复                      |
| --- | ------------------------------------------------- | --------------------------------------------------------- | ------------------------- |
| 1   | `preprocess.py` `normalize_graph`                 | 使用未定义变量 `adj`                                      | `adj = H.copy()`          |
| 2   | `preprocess.py`                                   | `'crs'` 拼写错误                                          | 改为 `'csr'`              |
| 3   | `preprocess.py` `Build_hypergraph_spatial_and_HE` | 默认返回 `coo_matrix` 不支持子图索引                      | 默认改为 `csr`            |
| 4   | `model.py` + `SpatialEx.py`                       | `Model_Plus.forward` 中 `agg_mtx` 与 `x_prime` 维度不匹配 | 增加 `selection` 参数     |
| 5   | `SpatialEx.py`                                    | `Regression.forward` 中 `agg_mtx` 与预测维度不匹配        | 传入前做 `selection` 切片 |
| 6   | `model.py` `Regression` / `Model_Plus`            | `BatchNorm1d` 在 batch_size=1 时崩溃                      | 替换为 `LayerNorm`        |

---

## 4. 改进算法：SpatialEx-GT

### 4.1 架构对比

| 模块       | 官方 SpatialEx+  | 改进 SpatialEx-GT                               |
| ---------- | ---------------- | ----------------------------------------------- |
| 空间编码器 | HGNN（固定超边） | **Graph Transformer**（稀疏邻居注意力，O(N·k)） |
| 自监督     | DGI（全局对比）  | **Masked Feature Prediction**（局部掩码预测）   |
| 组学翻译器 | MLP Regression   | **Cross-Attention Translator**（显式基因交互）  |

### 4.2 关键代码文件

- `SpatialEx/model_improved.py` — 改进模型
- `SpatialEx/SpatialEx_improved.py` — 改进训练器

---

## 5. 运行脚本

创建文件 `/root/autodl-tmp/run_benchmark.py`：

```python
"""
在 A100 上运行 SpatialEx+ 对比实验
使用真实的 Xenium Human Breast Cancer 数据
"""
import sys
import numpy as np
import torch
import scanpy as sc

import SpatialEx as se
from SpatialEx import preprocess as pp
from SpatialEx.utils import Compute_metrics
from SpatialEx.SpatialEx_improved import SpatialExP_GT

np.random.seed(42)
torch.manual_seed(42)

# ===================== 加载真实数据 =====================
print("Loading real Xenium data...")
adata1 = sc.read_h5ad('/root/autodl-tmp/data/slice1_uni.h5ad')
adata2 = sc.read_h5ad('/root/autodl-tmp/data/slice2_uni.h5ad')
print(f"Slice 1: {adata1.shape}, Slice 2: {adata2.shape}")

# 预处理
for adata in [adata1, adata2]:
    if hasattr(adata.X, 'todense'):
        adata.X = adata.X.todense().A
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# 构建图
graph1 = pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors=7, normalize=True)
graph2 = pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors=7, normalize=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ===================== 实验配置 =====================
CONFIG = {
    'hidden_dim': 512,
    'num_layers': 2,
    'epochs': 500,
    'lr': 0.001,
    'prune': 10000,
    'batch_size': 4,
    'seed': 42,
    'device': device,
}

# ===================== 官方版本 =====================
print("\n" + "=" * 60)
print("Training Original SpatialEx+")
print("=" * 60)
model_orig = se.SpatialExP(
    adata1=adata1, adata2=adata2, graph1=graph1, graph2=graph2, **CONFIG
)
model_orig.train()

# ===================== 改进版本 =====================
print("\n" + "=" * 60)
print("Training Improved SpatialExP_GT")
print("=" * 60)
model_imp = SpatialExP_GT(
    adata1=adata1, adata2=adata2, graph1=graph1, graph2=graph2,
    num_heads=8, dropout=0.1, use_mfp=True, **CONFIG
)
model_imp.train()

# ===================== 评估 =====================
print("\n" + "=" * 60)
print("Evaluation")
print("=" * 60)

A1_orig = model_orig.inference_direct(adata1.obsm['he'], graph1, 'panelA')
B2_orig = model_orig.inference_direct(adata2.obsm['he'], graph2, 'panelB')

A1_imp = model_imp.module_HA.predict(
    torch.Tensor(adata1.obsm['he']).to(device),
    pp.sparse_mx_to_torch_sparse_tensor(graph1).to(device), grad=False
).detach().cpu().numpy()
B2_imp = model_imp.module_HB.predict(
    torch.Tensor(adata2.obsm['he']).to(device),
    pp.sparse_mx_to_torch_sparse_tensor(graph2).to(device), grad=False
).detach().cpu().numpy()

def report(name, pred, true):
    pcc, pcc_m = Compute_metrics(pred, true, metric='pcc', reduce='mean')
    rmse, rmse_m = Compute_metrics(pred, true, metric='rmse', reduce='mean')
    ssim, ssim_m = Compute_metrics(pred, true, metric='ssim', reduce='mean', graph=graph1 if 'A1' in name else graph2)
    print(f"{name}: PCC={pcc_m:.4f}, RMSE={rmse_m:.4f}, SSIM={ssim_m:.4f}")
    return pcc_m, rmse_m, ssim_m

p1o, r1o, s1o = report("Orig Slice1", A1_orig, adata1.X)
p1i, r1i, s1i = report("Imp  Slice1", A1_imp, adata1.X)
p2o, r2o, s2o = report("Orig Slice2", B2_orig, adata2.X)
p2i, r2i, s2i = report("Imp  Slice2", B2_imp, adata2.X)

print("\n--- Summary ---")
print(f"Slice1: PCC {p1i-p1o:+.4f}, RMSE {r1o-r1i:+.4f}, SSIM {s1i-s1o:+.4f}")
print(f"Slice2: PCC {p2i-p2o:+.4f}, RMSE {r2o-r2i:+.4f}, SSIM {s2i-s2o:+.4f}")

# 保存结果
import os
os.makedirs('/root/autodl-tmp/results', exist_ok=True)
np.save('/root/autodl-tmp/results/A1_orig.npy', A1_orig)
np.save('/root/autodl-tmp/results/B2_orig.npy', B2_orig)
np.save('/root/autodl-tmp/results/A1_imp.npy', A1_imp)
np.save('/root/autodl-tmp/results/B2_imp.npy', B2_imp)
print("\nResults saved to /root/autodl-tmp/results/")
```

### 运行命令

```bash
cd /root/autodl-tmp/SpatialEx
conda activate spatialex
python /root/autodl-tmp/run_benchmark.py
```

---

## 6. 预期结果与解读

- **PCC ↑**、**SSIM ↑**、**RMSE ↓** 表示改进有效
- 如果改进在真实数据上提升不明显，可能原因：
  1. UNI 特征质量极高，HGNN 已足够表达
  2. 需要更多 epochs（1000+）
  3. 可尝试关闭 MFP（`use_mfp=False`）或调整 `dropout`

---

## 7. 快速调试命令

```python
# 检查数据
import scanpy as sc
adata = sc.read_h5ad('/root/autodl-tmp/data/slice1_uni.h5ad')
print(adata.shape)
print(adata.obsm.keys())  # 应包含 'he' 和 'spatial'
print(adata.obsm['he'].shape)  # 如 (n_cells, 1024) 或类似
```

---

## 8. Git 提交建议

```bash
cd /root/autodl-tmp/SpatialEx
git add -A
git commit -m "A100 benchmark: SpatialEx+ vs SpatialEx-GT on Xenium Breast Cancer"
```
