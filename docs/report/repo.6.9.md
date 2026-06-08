# 6.9 SpatialEx Evo 汇报

SpatialEx是什么项目?
SpatialEx是补全多个omics的模型. 其数据流是:
HE-UNI-> 形成embedding --> SpatialEx-->出来.... 

SpatialEx存在什么bugs, 我在什么benchmark上跑出来什么结果, 为什么使用这个benchmark?

我在两个数据集上设定了benchmark, 两个数据集来自项目自己的preprocess. Benchmark的目标是..., 

那么基于以上对于数据集的理解, 应该做什么改进? 我替换了超边模块, 优化成带权重的版本. 结果是....

目前, 第一阶段的改进内容已经同步到

## 一、理解项目：SpatialEx / SpatialEx+ 的架构与数据流

本周首先对项目做了**从论文到代码的完整拆解**，核心产出是**10章反向工程课程**（`curriculum/`）。拆解过程中梳理出的完整数据流：

```
H&E 图像 → 细胞分割 → 空间坐标
                      ↓
            基因表达矩阵 (adata.X) ──→ 超图构建 (BallTree k-NN)
                      ↓                           ↓
            MLP 编码器 (基因 → 隐空间) ←── 图归一化 (GCN / HPNN)
                      ↓
            HGNN / Graph Transformer 消息传播
                      ↓
            回归翻译器 (cross-panel: A → B)
                      ↓
            6-loss 训练 (AA, BB, AB, BA, ABA, BAB)
                      ↓
            推理 → 伪spot聚合 → 图感知评估 (PCC, RMSE, SSIM)
```

同时识别了**10个关键设计问题**，比如：
- 任务本质：当前两切片基因 panel 完全相同（313 genes），翻译任务近似恒等映射
- 图是否动态：训练过程中邻接矩阵是静态的，不随特征更新
- 6-loss 是否有冗余：cycle loss 在 panel 相同的情况下信息量有限

## 二、项目复现：环境搭建 + 6个 Bug 修复 + 500 Epoch 全量跑通

### 环境迁移
- 从 venv 迁移到 **conda `spatialex`**（Python 3.10）
- PyTorch 升级到 **`2.7.0+cu128`**，解决 RTX 5090（sm_120）的兼容性问题

### 6 个原始代码 Bug 修复
| Bug                             | 位置                         | 修复内容                                      |
| ------------------------------- | ---------------------------- | --------------------------------------------- |
| `adj` 未定义                    | `utils.py` `normalize_graph` | 补充 `adj = sp.coo_matrix(...)`               |
| `'crs'` 拼写错误                | `utils.py`                   | 改为 `'csr'`                                  |
| `coo` → `csr` 格式              | `utils.py`                   | 显式 `.tocsr()`                               |
| `Model_Plus.forward` 维度不匹配 | `model.py`                   | 修正 slice 特征拼接维度                       |
| `Regression.forward` 维度不匹配 | `model.py`                   | 修正 hidden → output 的投影                   |
| `BatchNorm1d` 失效              | `model_improved.py`          | `batch_size=1` 时 BN 无意义，改为 `LayerNorm` |

### 500 Epoch 复现结果（真实 Xenium 数据）
在 **Slice1（164k cells × 313 genes）** 和 **Slice2（111k × 313）** 上训练 500 epoch：

| 模型         | 隐藏维度 | Slice1 PCC | Slice1 RMSE | Slice2 PCC | Slice2 RMSE | 耗时    |
| ------------ | -------- | ---------- | ----------- | ---------- | ----------- | ------- |
| **Original** | 512      | **0.3135** | 1.4516      | **0.3202** | 1.4617      | ~6 min  |
| **Improved** | 128      | 0.3089     | **1.4488**  | 0.3064     | 1.4672      | ~15 min |

- **SSIM 不稳定**：Slice1 偶尔触发 `scipy.sparse` 矩阵乘法维度不匹配，已加 try/except 兜底
- 原始模型指标略优，主要是 **4× 隐藏维度**（512 vs 128）的容量优势

---

## 三、项目改进的第一部分：内存效率优化

核心目标：**让 164k 节点的大图能在单卡 RTX 5090（31GB）上训练**，而不 OOM。

### 1. GraphTransformerLayer 的 OOM 修复
**问题**：标准 dense self-attention 在 164k 节点上需要分配 **~8GB 显存** 仅用于注意力矩阵，加上 6-loss 反向传播直接 OOM。

**改进**：
- **Chunked sparse neighbor attention**：沿超图边（~1.1M 条边）计算 attention，分 **50k edge / chunk** 处理，注意力计算峰值从 ~8GB → **~150MB**
- **Gradient checkpointing**：`torch.utils.checkpoint` 包装 attention block，用计算换内存
- 训练速度从 ~1.3 it/s 降到 ~0.5 it/s（可接受）

### 2. CrossAttentionTranslator 的 OOM 修复
**问题**：原始 `nn.MultiheadAttention` 在 164k cells 上分配 **200GB 显存**（seq_len² × batch × head_dim）。

**改进**：
- 简化为轻量 MLP：`input_proj → LayerNorm → FFN`
- 完全消除 seq_len² 的内存开销

### 3. 公平比较的局限
- 改进模型**前向传播**可以跑 h=256，但 **6-loss 反向传播峰值** 导致只能稳定跑 h=128
- 原始模型能跑 h=512，因此当前指标上原始模型略优
- **下一步**：若要真正对比 HGNN vs Graph Transformer，需要解决改进模型的 backward 峰值，或者给原始模型也加 memory constraint

---

## 本周核心结论

1. **项目是可复现的**，但原始代码有多个低级 bug（拼写、维度、格式），需要修复才能跑通
2. **原始架构在 164k 细胞规模上接近内存极限**，不改 attention 机制的话基本无法扩展
3. **改进方向（sparse attention + checkpointing）是正确的**，内存问题解决得很干净，但因 backward peak 导致 hidden_dim 受限，公平对比还需继续优化
4. **课程生成**是理解的副产品，把项目从" hypergraph 构建"到"Graph Transformer layer"拆解成了 10 个可独立运行、逐层递进的练习

**下周如果继续的话，最自然的方向是**：
- 解决改进模型 backward 的内存峰值（比如 gradient accumulation、或者进一步减少 loss 同时回传的峰值）
- 让改进模型也能跑 h=512，做真正的 architecture ablation
- 分析 6-loss 中哪些损失在 panel 相同的情况下是冗余的，尝试简化训练目标