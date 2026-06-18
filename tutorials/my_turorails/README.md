# 从零实现 SpatialEx+

> **不要阅读源码。重建源码。**

本课程通过 10 个渐进式练习，让你从零写出 SpatialEx+ 的核心系统。

## 你将实现什么

SpatialEx+ 是一个 **空间转录组学预测系统**：给定组织切片的 H&E 图像嵌入和空间坐标，预测每个细胞的基因表达。其核心能力包括：

1. **超图构建**：从空间坐标构建 k-NN 邻接矩阵
2. **超图归一化**：GCN / HPNN 对称归一化
3. **HGNN 消息传递**：在超图上聚合邻居信息
4. **MLP 编码器**：将 H&E 嵌入投影到隐藏维度
5. **单 slice 预测**：端到端前向传播
6. **伪 Spot 聚合**：单细胞预测 → Spot 级别监督
7. **回归翻译器**：跨 Panel 映射（panelA → panelB）
8. **六 Loss 训练循环**：完整的 SpatialExP 训练器
9. **图感知评估**：PCC、RMSE、图 SSIM
10. **Graph Transformer 层**：稀疏邻居注意力（进阶）

## 课程结构

```
curriculum/
├── dependency_graph.md      ← Stage 1: 认知依赖图
├── curriculum.yaml          ← Stage 2: 完整课程规划
├── README.md                ← 本文件
│
├── 01-hypergraph-construction/
│   ├── exercise.md
│   ├── starter.py
│   ├── test.py
│   └── solution.py
├── 02-graph-normalization/
│   └── ...
├── 03-hgnn-layer/
│   └── ...
├── 04-mlp-encoder/
│   └── ...
├── 05-single-slice-model/
│   └── ...
├── 06-pseudo-spot-aggregation/
│   └── ...
├── 07-regression-translator/
│   └── ...
├── 08-six-loss-trainer/
│   └── ...
├── 09-graph-aware-evaluation/
│   └── ...
└── 10-graph-transformer-layer/
    └── ...
```

## 如何学习

### 环境准备

```bash
conda create -n spatialex python=3.10
conda activate spatialex
pip install torch numpy scipy scikit-learn
```

### 学习流程

1. **阅读 `exercise.md`**：理解本课目标和约束
2. **运行 `python test.py`**：看看当前会失败什么（红色）
3. **打开 `starter.py`**（如果有）：从骨架开始，或创建新文件
4. **编写代码**：实现 exercise.md 中要求的功能
5. **再次运行 `python test.py`**：直到全部 PASS（绿色）
6. **（可选）查看 `solution.py`**：对照参考实现
7. **进入下一课**

### 黄金法则

- **每课只引入 1 个新概念**
- **每课核心代码 ≤ 50 行**
- **不要跳课**：前面的课是后面的地基
- **不要读原仓库代码**：靠自己推导

## 前置知识

- Python 3.10+
- PyTorch 基础（`nn.Module`, `forward`, 自动求导）
- `scipy.sparse` 基础（csr_matrix, sparse matrix multiplication）
- MLP 和注意力机制的基本概念

## 总代码量

完成全部 10 课后，你的总代码量约为 **300 行**，但功能等价于原仓库 >3000 行的核心能力。

---

**准备好了吗？从 `01-hypergraph-construction/exercise.md` 开始。**
