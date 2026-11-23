# 答辩 PPT 大纲（7 章 59 页版）

> 对应文件：`docs/SpatialEx_Defense.pptx`  
> 目标页数：**59 页**  
> 风格：深蓝主色 + 橙色重点 + 标题下划线 + 正文卡片分段
>
> 图片资源：
> - `docs/image/fig3_diagnosis/`（20 张 SVG 诊断/实验图 + 公式缓存 `docs/image/formulas/`）
> - `docs/image/from_paper/`（论文原图 Fig.1–6 及 Extended Data）
> - `docs/image/` 下的复现/结果图

---

## 第 1 页：标题

- 标题：SpatialEx/SpatialEx+ 复现与改进
- 副标题：Fig.3 Panel Diagonal Integration 上的监督信号探索
- 论文：*High-parameter spatial multi-omics through histology-anchored integration* (Nature Methods 2025)
- 答辩日期：2025 年 6 月 21 日（线上，约 40 分钟）
- 答辩人：李熹鸣
- 项目仓库：https://github.com/965120527lxm-maker/SpatialEvo
- 核心发现：MLP + Strict MNN 显著优于官方 HGNN/GT + Cycle

---

## 第 2 页：答辩提纲

1. 背景与任务：Spatial Diagonal Integration & Fig.1–5 & 数学符号
2. 环境搭建与复现：Fig.2–5 复现与官方代码 6 个 bug 修复
3. 改进一：Graph Transformer 替代 HGNN（架构 + 结果）
4. 诊断：Branch Decomposition 与信号来源分析
5. 改进二：Strict MNN 伪标签替代 Cycle（架构 + 结果）
6. 讨论：为什么 MLP + MNN 反而最强？
7. 结论、局限与展望

---

## 第 3 页：章节分隔 — 一、背景与任务

**过渡语**：先介绍我们要解决什么问题，以及论文原文在 Fig.1–5 中展示了哪些任务。

---

## 第 4 页：背景 — 空间组学的高参数困境

- 空间组学技术面临分辨率、通量、panel 大小之间的权衡
  - 高分辨率技术（Xenium、Merfish、CosMx）通常只能测几十到几百个基因
  - 大 panel 技术（Visium）分辨率低，且仍是转录组子集
  - 同一组织切片无法同时满足“高分辨率 + 大 panel + 多组学”
- 替代方案：相邻切片测互补 panel / 互补组学，再计算整合
- 引出 Spatial Diagonal Integration：两个切片没有共测 omics 特征
  - 区别于 Horizontal Integration（同特征空间）
  - 区别于 Vertical Integration（同一细胞多模态）

**配图**：`docs/image/fig1_bottom_three_scenarios.png`
- 论文 Fig.1 下半部分，展示三种空间组学整合场景
- 左边：H&E-to-omics prediction
- 中间：Panel diagonal integration（本工作聚焦）
- 右边：Omics diagonal integration

---

## 第 5 页：论文原图 Fig.1 — SpatialEx/SpatialEx+ 方法框架

- 配图：`docs/image/fig1_top_framework.png`
- 论文 Fig.1 上半部分，展示 SpatialEx/SpatialEx+ 的整体方法框架
- 用途：说明 SpatialEx 在论文中的定位
- 讲点：Fig.3 的 panel diagonal 是论文方法的重要应用场景

---

## 第 6 页：两种 Diagonal Integration

| Panel Diagonal Integration | Omics Diagonal Integration |
|----------------------------|----------------------------|
| 同一切片技术（如都是 Xenium） | 不同组学类型 |
| 不同基因 panel（panel A vs panel B） | 如转录组 + 蛋白 |
| **Fig.3 任务** | **Fig.5 任务** |
| 本工作聚焦于此 | 已初步复现 |

---

## 第 7 页：论文原图 Fig.2 — H&E → 全转录组任务（数学化）

- 配图：`docs/image/from_paper/Fig2_41592_2025_2926_Fig2_HTML.png`
- 任务设定（同一组织切片内）：
  - H&E 嵌入：$X \in \mathbb{R}^{N \times d}$
  - 全转录组表达：$Y \in \mathbb{R}^{N \times 313}$
  - 目标：学习映射 $f: X \to \hat{Y}$，使得 $\hat{Y} \approx Y$
- 与 Fig.3 的关键区别：
  - Fig.2 在同一切片内同时观测 $X$ 与 $Y$，属于**有监督回归**
  - Fig.3 的两个切片没有共测 panel，属于**跨切片缺失 panel 预测**
- 本工作补充 Fig.2 baseline 作为理解 Fig.3 的对照

---

## 第 8 页：论文原图 Fig.3 — Panel Diagonal Integration（数学化）

- 配图：`docs/image/from_paper/Fig3_41592_2025_2926_Fig3_HTML.png`
- 符号设定（相邻切片）：
  - Slice 1：H&E $X_1$，已测 panel A $Y_A^1 \in \mathbb{R}^{N_1 \times |A|}$，缺失 panel B $Y_B^1 \in \mathbb{R}^{N_1 \times |B|}$
  - Slice 2：H&E $X_2$，已测 panel B $Y_B^2 \in \mathbb{R}^{N_2 \times |B|}$，缺失 panel A $Y_A^2 \in \mathbb{R}^{N_2 \times |A|}$
- 目标：$\hat{Y}_B^1 \approx Y_B^1$，$\hat{Y}_A^2 \approx Y_A^2$
- 关键约束：训练时不可用 held-out panel（$Y_B^1$、$Y_A^2$）
- 官方方法：SpatialEx+ 通过 Cycle 约束做 cross-panel 预测

---

## 第 9 页：论文原图 Fig.4 — 百万细胞大切片 Panel Diagonal（可扩展性）

- 配图：`docs/image/from_paper/Fig4_41592_2025_2926_Fig4_HTML.png`
- 任务设定：10x Xenium 大切片，两片合计约 1.8M 细胞，每片 ~900k 细胞
- 协议：把 280-gene Xenium panel 平分为 panel A / panel B，两片互补测量
  - Slice 1：H&E $X_1$ + 已测 panel A $Y_A^1$，缺失 panel B $Y_B^1$
  - Slice 2：H&E $X_2$ + 已测 panel B $Y_B^2$，缺失 panel A $Y_A^2$
- 目标：$\hat{Y}_B^1 \approx Y_B^1$，$\hat{Y}_A^2 \approx Y_A^2$
- 与 Fig.3 的关系：Fig.4 是 Fig.3 在百万细胞尺度上的可扩展性验证
- 讲点：我们训练完成；评估时遇到 `coo_matrix` subscript bug，metrics 来自已有输出

---

## 第 10 页：论文原图 Fig.5 — Omics Diagonal Integration（跨组学）

- 配图：`docs/image/from_paper/Fig5_41592_2025_2926_Fig5_HTML.png`
- 符号设定（跨组学）：
  - Slice 1：H&E $X_1$，已测组学 A $Y_A^1$（如 RNA），缺失组学 B $Y_B^1$（如蛋白）
  - Slice 2：H&E $X_2$，已测组学 B $Y_B^2$（如蛋白），缺失组学 A $Y_A^2$（如 RNA）
- 目标：$\hat{Y}_B^1 \approx Y_B^1$，$\hat{Y}_A^2 \approx Y_A^2$
- 关键约束：训练时不可用 held-out omics（$Y_B^1$、$Y_A^2$）
- 与 Fig.3 的关系：Fig.3 是 panel-level 的 diagonal；Fig.5 是 omics-level 的 diagonal
- 讲点：Fig.5a,b 为 transcriptomics ↔ proteomics；Fig.5c–f 为 transcriptomics ↔ metabolomics。我们已初步复现 transcriptomics ↔ proteomics，protein→RNA 方向 PCC≈0.20；RNA→protein 仅 2 个蛋白 marker，任务过难。Strict MNN 思想可进一步推广

---

## 第 11 页：Fig.3 Strict 协议与评价指标

- 在 Fig.3 数学化的基础上，进一步明确 Strict 协议：
  - Slice 1：$\{X_1, Y_A^1\} \to \hat{Y}_B^1$
  - Slice 2：$\{X_2, Y_B^2\} \to \hat{Y}_A^2$
  - 约束：$Y_B^1, Y_A^2 \notin \mathcal{D}_{\text{train}}$
- 可用信号来源：
  1. H&E 形态先验
  2. 跨切片 pseudo-label
  3. 自监督约束（Cycle / DGI）
- 评价指标：gene-level PCC（主指标）、SSIM、CMD

---

## 第 12 页：复现图 — Fig.3 Strict 协议数据划分

- 配图：`docs/image/fig3b.png`
- 左侧 Slice1：$X_1 + Y_A^1 \to$ predict $Y_B^1$
- 右侧 Slice2：$X_2 + Y_B^2 \to$ predict $Y_A^2$
- 红色框表示 held-out、训练不可用的 panel

---

## 第 13 页：章节分隔 — 二、环境搭建与复现

**过渡语**：先把官方代码跑通，并补充 Fig.2 baseline；所有改进（GT、MNN）放到后面章节。

---

## 第 14 页：个人环境与 Conda 配置

- 硬件环境：
  - GPU：RTX 5090 / A100（训练大图模型时使用）
  - 显存峰值：HGNN-512 约 20–24 GB；GT Cross-Attention 因序列长度平方复杂度超出显存，退化为轻量 MLP
- 软件环境：
  - Python 3.10 + PyTorch 2.x + CUDA 11.8
  - 关键包：DGL（图神经网络）、anndata（单细胞数据）、UNI（病理图像预训练模型）、cellpose（分割）
- Conda 环境配置要点：
  - `requirements.txt` 已锁定主要依赖
  - DGL 需按 CUDA 版本单独安装
  - UNI 权重需从 HuggingFace 下载

---

## 第 15 页：数据集与已复现任务

- 数据集：Xenium Human Breast Cancer
  - `Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad`：Slice 1，~164k 细胞，313 基因
  - `Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad`：Slice 2，~111k 细胞，313 基因
  - 每个切片包含：`adata.X`（表达）、`obsm['he']`（UNI H&E 嵌入）、`obsm['spatial']`（空间坐标）
  - `10x_breast_cancer/human_breast_cancer.h5ad`：NicheTrans 蛋白数据（用于 Fig.5 Omics Diagonal）
  - `xenium_big/`：Fig.4 百万细胞大切片数据
- 已复现任务：
  1. **Fig.2**：SpatialEx 单切片 H&E → 全转录组（双向跨切片评估）
  2. **Fig.3**：SpatialEx+ Panel Diagonal Integration（官方 150/163 gene split，Cycle baseline）
  3. **Fig.4**：SpatialEx+ 百万细胞大切片 Panel Diagonal
  4. **Fig.5 (Omics Diagonal)**：SpatialEx+ 转录组 ↔ 蛋白组（transcriptomics-proteomics）
- Fig.5c–f 代谢组场景：作为后续方向，尚未系统复现

---

## 第 16 页：复现成果总览 — 一图一表

- **左图**：复现结果对比柱状图
  - Fig.2：PCC / SSIM / CMD
  - Fig.3：PCC / SSIM / CMD
  - Fig.4：百万细胞 Panel Diagonal 的 PCC / SSIM / CMD
  - Fig.5：Omics Diagonal 的 PCC / SSIM / CMD
- **右表**：SpatialEx 复现结果总表

| 任务 | 方法 | 方向 | PCC | SSIM | CMD |
|------|------|------|-----|------|-----|
| Fig.2 | HGNN SpatialEx-512 | S1 / S2 | 0.257 / 0.273 | 0.419 / 0.425 | 0.205 / 0.207 |
| Fig.3 | HGNN-512 + Cycle | S1→B / S2→A | 0.275 / 0.301 | 0.308 / 0.332 | 0.334 / 0.267 |
| Fig.4 | HGNN-512 + Cycle | S1→B / S2→A | 0.008 / 0.013 | 0.439 / 0.433 | 0.328 / 0.383 |
| Fig.5 (Omics) | SpatialEx+ | protein→RNA / RNA→protein | 0.196 / -0.042 | 0.367 / 0.187 | 0.222 / 0.003 |

- 说明：表中只列官方 SpatialEx/SpatialEx+ 复现结果；GT、MNN 等改进方法在第 3–5 章介绍。

---

## 第 17 页：Fig.2 复现 — 与原文对比

- 任务：输入仅为 H&E，输出为全转录组 313 genes
- 协议：双向跨切片 train→test
  - 在 Slice A 上训练 H&E→表达，在 Slice B 全部细胞上评估
  - 反向再做一次
- 复现结果：

| 方法 | S1 PCC | S2 PCC | S1 SSIM | S2 SSIM | S1 CMD | S2 CMD |
|------|--------|--------|---------|---------|--------|--------|
| **HGNN SpatialEx (512)** | **0.257** | **0.273** | **0.419** | **0.425** | **0.205** | **0.207** |
| DeepPT | 0.268 | 0.276 | 0.357 | 0.366 | 0.271 | 0.281 |

- 与原文对比：
  - 论文 Fig.2c 报告 SpatialEx 在 PCC/SSIM/CMD 上均优于 DeepPT
  - 我们的复现中 DeepPT PCC 略高，但 HGNN SSIM/CMD 更好，空间结构保留更完整
  - 总体趋势与论文一致：HGNN 在结构保真度上占优

---

## 第 18 页：Fig.3 复现 — 与原文对比

- 任务：Panel Diagonal Integration，官方 150/163 gene split
- 方法：SpatialEx+ HGNN-512 + Cycle
- 复现结果：

| 编码器 | 监督 | S1 PCC | S2 PCC | S1 SSIM | S2 SSIM | S1 CMD | S2 CMD |
|--------|------|--------|--------|---------|---------|--------|--------|
| HGNN-512 | Cycle | 0.275 | 0.301 | 0.308 | 0.332 | 0.334 | 0.267 |
| DeepPT | — | 0.262 | 0.280 | 0.334 | 0.360 | 0.298 | 0.304 |

- 与原文对比：
  - 论文 Fig.3 展示了 SpatialEx+ 在 panel diagonal 上的性能
  - 我们的 Cycle baseline 在官方 split 上复现出相近数量级，并补充 DeepPT baseline 与 CMD 指标
  - SpatialEx+ PCC 略高，DeepPT SSIM 略高，两者互有胜负
  - 该 baseline 是后续 GT、MNN 改进的对照基准

---

## 第 19 页：Fig.4 与 Fig.5 (Omics Diagonal) 复现 — 与原文对比

### Fig.4：百万细胞大切片 Panel Diagonal

| 方向 | PCC | SSIM | CMD |
|------|-----|------|-----|
| Slice1 → panel B | 0.008 | 0.439 | 0.328 |
| Slice2 → panel A | 0.013 | 0.433 | 0.383 |

- 论文 Fig.4 展示 SpatialEx+ 在 ~900k 细胞大切片上的可扩展性
- 我们的复现：训练完成，评估时遇到 `coo_matrix` subscript bug；metrics 来自已有输出

### Fig.5 (Omics Diagonal)：转录组 ↔ 蛋白组

| 方向 | PCC | SSIM | CMD |
|------|-----|------|-----|
| protein → RNA | 0.196 | 0.367 | 0.222 |
| RNA → protein | -0.042 | 0.187 | 0.003 |

- 论文 Fig.5 主图以可视化为主，定量指标见 Extended Data Fig.8a；RNA→protein 仅 2 个 marker，任务过难；MNN 探索结果见 Appendix

---

## 第 20 页：复现中修复的 6 个关键 Bug

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| 1 | `preprocess.normalize_graph` | 未定义变量 `adj` | 图归一化失败 |
| 2 | `preprocess` | `'crs'` 拼写错误 | sparse 转换失败 |
| 3 | `Build_hypergraph_spatial_and_HE` | 默认返回 `coo` | 子图索引报错 |
| 4 | `Model_Plus.forward` | `agg_mtx` 维度不匹配 | forward 报错 |
| 5 | `SpatialExP.train` | `Regression` 维度不匹配 | Cycle 训练崩溃 |
| 6 | `Model_Plus / Regression` | BatchNorm batch=1 崩溃 | 小 batch 中断 |

- 所有 bug 均在官方原始仓库中验证；修复它们是后续实验的前提。

---

## 第 21 页：章节分隔 — 三、改进一：Graph Transformer 替代 HGNN

**过渡语**：复现跑通后，第一次改进尝试聚焦**网络架构**——能否用更强的 Graph Transformer 替换 HGNN，在不改变 Fig.3 Strict 协议的前提下提升 missing panel 预测？

**本章要回答的问题**：
- GT 在理论上比 HGNN 更强，换架构能否带来 Fig.3 增益？
- 若不能，瓶颈在空间编码器，还是在监督信号 / 输入模态？

---

## 第 22 页：改进一 — 动机与模块替换

**版式**：左文右表；本页不放实验结果图，专注「为什么要换 GT」。

### HGNN 的局限

- 固定权重超边聚合；无法按邻居特征自适应加权

$$
\mathbf{h}_i^{(l+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i)} \frac{1}{|\mathcal{N}(i)|} \mathbf{W}^{(l)} \mathbf{h}_j^{(l)} \right)
$$

### SpatialEx-GT 模块对照（`SpatialEx/model_improved.py`）

| 模块 | SpatialEx+ | SpatialEx-GT | 动机 |
|------|------------|--------------|------|
| 空间编码器 | HGNN 固定聚合 | GT 稀疏注意力 | 自适应权重 |
| 自监督 | DGI | MFP | 局部掩码重建 |
| 翻译器 | MLP | 轻量 MLP | 164k 细胞可训练 |
| 归一化 | BatchNorm | LayerNorm | batch=1 稳定 |

**讲点**：GT hidden=128（显存峰值），HGNN baseline=512 → 存在容量差异，后续结果需结合 Fig.2 对照解读。

---

## 第 23 页：GT 架构说明

**版式**：左图（纯结构 SVG）+ 右栏说明。

**左图**：`docs/image/fig3_diagnosis/12_gt_architecture.svg`（仅模块与数据流，无长文）

**右栏**：
- 模块对照表：HGNN vs GT
- 要点：UNI H&E + 空间坐标 → k-NN → GT Encoder；Panel / MFP / Cycle 三路损失

**讲点**：Cross-Attention Translator 因 $O(N^2)$ 显存退化为轻量 MLP；训练器 `SpatialExP_GT`。

---

## 第 24 页：Fig.3 结果 — GT vs HGNN（Cycle）

**版式**：上方结果表 + 下方柱状图 + 底部结论条。

| 编码器 | Hidden | S1 PCC | S2 PCC | S1 SSIM | S2 SSIM |
|--------|--------|-------:|-------:|--------:|--------:|
| HGNN-512 + Cycle | 512 | **0.275** | **0.301** | 0.308 | 0.332 |
| GT-128 + Cycle | 128 | 0.267 | 0.276 | **0.345** | **0.357** |

**配图**：`docs/image/fig3_diagnosis/14_gt_vs_hgnn_fig3_bar.svg`

**讲点**：PCC 上 HGNN 更高；SSIM 上 GT 略优（+0.04），但未形成 missing panel 预测优势。

---

## 第 25 页：Fig.3 多方法对照

**配图（全页）**：`docs/image/fig3_diagnosis/17_fig3_cycle_methods_bar.svg`

- DeepPT / HGNN+Cycle / GT+Cycle / MLP+Strict MNN 四方法 PCC+SSIM
- 定位：GT 与 HGNN 同属论文方法线；MLP+MNN 为后续改进参照

**讲点**：GT 未超过 HGNN，更未接近 MNN → 架构不是 Fig.3 主瓶颈。

---

## 第 26 页：Fig.3 Per-Gene — GT 是否系统性更好？

**配图（全页）**：`docs/image/fig3_diagnosis/16_gt_vs_hgnn_pergene_scatter.svg`

- 横轴 HGNN PCC，纵轴 GT PCC；对角线下方 = HGNN 更好
- Slice1：GT 赢 **62/163（38%）**；Slice2：GT 赢 **28/150（19%）**

**讲点**：GT 没有在基因层面系统性超越 HGNN，「基本持平」主要由 SSIM 局部优势支撑。

---

## 第 27 页：Fig.2 对照 — Encoder Ablation

**版式**：上方表格 + 下方柱状图。

| 方法 | S1 PCC | S2 PCC | S1 SSIM | S2 SSIM |
|------|-------:|-------:|--------:|--------:|
| HGNN SpatialEx (512) | **0.257** | **0.273** | **0.419** | **0.425** |
| GT-128 + MFP | 0.228 | 0.236 | 0.309 | 0.320 |
| GT-512 + MFP | 0.244 | 0.246 | 0.339 | 0.329 |
| GT-128 + DGI | 0.225 | 0.235 | 0.297 | 0.313 |

**配图**：`docs/image/fig3_diagnosis/15_fig2_encoder_ablation.svg`

**讲点**：纯 H&E 任务上 HGNN 仍最优；GT-512 加容量仍追不平；MFP≈DGI。

---

## 第 28 页：GT 改进小结与过渡

**版式**：左侧 5 条结论 + 右侧 `08_signal_contribution.png` 预告。

1. Fig.3：换 GT 无 PCC 优势；per-gene 胜率低
2. Fig.3 SSIM：GT 略优但未转化为预测精度
3. Fig.2：HGNN 全面领先 → GT 非 universally 更优
4. 根因：panel-to-panel 信号主导，H&E branch 弱
5. 下一步 → Branch Decomposition

**配图**：`docs/image/fig3_diagnosis/08_signal_contribution.svg`

---

## 第 29 页：章节分隔 — 四、诊断：Branch Decomposition 与信号来源分析

**过渡语**：网络不是瓶颈，我们开始显式诊断有效信号来源。

---

## 第 30 页：诊断动机 — 有效信号来自哪里？

**版式**：左文右公式；本页只讲「为什么要做 Branch Decomposition」。

### 背景问题

SpatialEx+ 同时存在两条预测路径：

- **H&E branch**：$X \to \hat{Y}_{\text{missing}}$
- **Panel branch**：$Y_{\text{measured}} \to \hat{Y}_{\text{missing}}$

改进一已证：换 GT 未带来 Fig.3 增益 → **瓶颈可能不在图网络，而在输入信号本身**。

### 本实验要回答

| 问题 | 若成立 → 含义 |
|------|----------------|
| H&E branch 单独能预测缺失 panel 吗？ | 能 → 换 encoder 应有帮助 |
| Panel branch 是否占主导？ | 是 → 应改进 panel-to-panel 监督 |
| 两分支融合能否互补？ | 能 → late fusion 应提升 |

**讲点**：这是**机制诊断**，不是再训一个更强模型；先把两条路径拆开量化。

---

## 第 31 页：Branch Decomposition — 实验设计

**版式**：全页流程图 + 底部 4 条说明。

**主图**：自动生成/更新的 `docs/image/branch_decomposition_design.svg`

- 上方：Slice 1 / Slice 2 输入（held-out 标红）
- 中间：两条并行分支
  - Branch 1（蓝）：H&E only → H&E encoder → 预测 missing panel
  - Branch 2（绿）：Panel only → Panel translator → 预测 missing panel
- 下方：Late Fusion（紫）把两分支结果融合

**底部说明**

1. **H&E branch**：只用形态嵌入 X 预测缺失 panel，测试 H&E 本身的信息量
2. **Panel branch**：只用已测 panel Y 预测缺失 panel，测试 panel-to-panel 信息量
3. **Late Fusion**：0.5 平均 / reliability-weighted，测试两分支能否互补
4. 机制诊断用 random gene split；定性结论对 official split 同样成立

**脚本**：`scripts/fig3/diagnose_spatialex_vs_cycle_outputs.py`

---

## 第 32 页：Branch Decomposition — 结果与结论

**版式**：上方结果表 + 下方三条结论（不放大图，留给下一页）。

| Variant | Slice1 PanelB PCC | Slice2 PanelA PCC |
|---------|------------------:|------------------:|
| H&E branch | -0.001 | 0.010 |
| Panel branch | **0.016** | **0.248** |
| 0.5 average (calibrated) | 0.011 | 0.177 |
| Reliability-weighted (calibrated) | -0.001 | 0.010 |

### 三条结论

1. **H&E branch ≈ 0**：H&E 不能形成稳定的 missing panel 先验
2. **Panel branch dominant**：有效信号来自 measured → missing panel
3. **Late fusion 有害**：Slice2 从 0.248 降至 0.177 → H&E 是噪声而非补充

**讲点**：因此 GT 无提升，根因是 **H&E 输入信息量低**；后续应改监督信号（→ Strict MNN）。

---

## 第 33 页：诊断图 — Branch Decomposition 可视化

**版式**：全页分组柱状图 + 底部 caption。

**主图**：自动生成/更新的 `docs/image/branch_decomposition_results.svg`

- 横轴：4 个 variant（H&E branch / Panel branch / 0.5 avg / Reliability-weighted）
- 纵轴：PCC
- 蓝条：Slice1 PanelB PCC；橙条：Slice2 PanelA PCC
- 关键信息：Panel branch 在 Slice2 显著高于 H&E branch；融合 variant 低于 Panel branch

**caption**：四 variant 对比验证「panel dominant、late fusion 有害」。

---

## 第 34 页：诊断图 — 信号贡献定量分析

- 配图：`docs/image/fig3_diagnosis/08_signal_contribution.svg`
- Slice2 方向：Panel branch 0.248 vs H&E 0.010 vs fusion 0.177
- 从另一角度验证「panel dominant、fusion 有害」

---

## 第 35 页：诊断图 — Per-Gene 预测散点

- 配图：`docs/image/fig3_diagnosis/03_per_gene_scatter.svg`
- 大多数基因上 MNN 监督优于 Cycle 监督
- 为改进二（Strict MNN）做铺垫

---

## 第 36 页：章节分隔 — 五、改进二：Strict MNN 伪标签替代 Cycle

**过渡语**：架构改进未带来增益后，第二次尝试聚焦**监督信号**——能否用跨切片 pseudo-label 替代 Cycle 自洽约束？

**本章要回答的问题**：
- Cycle 为什么无法恢复 missing panel？
- Strict MNN 如何在不用 held-out panel 的前提下构造可靠监督？
- 为何 MLP + Strict MNN 反而成为最优组合？

---

## 第 37 页：Cycle Consistency 的自洽陷阱

### Cycle 机制（SpatialEx+ 原监督）

- Slice 1：$Y_A^1 \xrightarrow{C_{A\to B}} \hat{Y}_B^1 \xrightarrow{C_{B\to A}} \tilde{Y}_A^1 \approx Y_A^1$
- Slice 2：对称地约束 $Y_B^2 \to \hat{Y}_A^2 \to \tilde{Y}_B^2 \approx Y_B^2$

### 问题：自洽 ≠ 正确

- Cycle 只保证 **panel 间往返一致**，不保证 $\hat{Y}_B^1 \approx Y_B^1$（真实 held-out panel）
- 若 H&E branch 弱或 $C_{A\to B}$ 学到幻觉，Cycle 会**加强幻觉**而非纠正
- 实验：MLP + Cycle only → PCC **0.005 / 0.013**，接近随机

**讲点**：Cycle 依赖“已测 panel → missing panel”映射本身可靠；Fig.3 下该映射缺乏外部锚点。

---

## 第 38 页：诊断图 — Cycle Self-Consistency Trap

- 配图：`docs/image/fig3_diagnosis/07_cycle_trap.svg`
- Cycle 让模型训练时自我闭环，测试时在 held-out panel 上暴露为随机或偏差

---

## 第 39 页：改进二动机 — Strict MNN 替代 Cycle

**版式**：左文右表；本页只讲「为什么要用 MNN 伪标签」。

### 核心思想

相邻切片组织连续 → 跨切片细胞存在**形态/空间对应**；用 MNN 建立可信配对，把参考切片 measured panel **转移**为 pseudo-label，替代 Cycle 自环。

### Strict 协议（全程不使用 held-out panel）

| 切片 | 训练/伪标签可用 | Held-out（eval only） |
|------|-----------------|------------------------|
| Slice 1 | $X_1$, $Y_A^1$ | $Y_B^1$ |
| Slice 2 | $X_2$, $Y_B^2$ | $Y_A^2$ |

**讲点**：pseudo-label 只能来自允许输入；$Y_B^1$、$Y_A^2$ 绝不参与 MNN 匹配或训练。

---

### 配图层次（第 40–42 页）

| 页 | 图文件 | 定位 |
|----|--------|------|
| **40** | `19_mnn_filter_algorithm.svg` | **单步原子** `build_mnn_pseudo`（Fwd/Rev/Mutual）；**不是** Step 1+2 综合图 |
| **41** | `20_strict_mnn_two_step_bridge.svg` | **Step 1 + Step 2 综合桥接图**（两步均调用上一页原子函数） |
| **42** | `13_mnn_architecture.svg` | **全流程**：Step 1 → Step 2 → MLP 训练（左图蓝框=Step1，橙框=Step2） |

---

## 第 40 页：MNN 单步原子 — build_mnn_pseudo

**版式**：左图（纯结构 SVG）+ 右栏说明。

**左图**：`docs/image/fig3_diagnosis/19_mnn_filter_algorithm.svg`（图 19）

**定位**：Strict MNN 的 **Step 1 / Step 2 均重复调用**此函数；本页**不是**两步综合图。

**右栏**：

对每个 query 细胞 $i$：

1. **归一化**：L2 normalize → 余弦相似度
2. **Forward**：$i$ 在 ref 中取 top-$mnn_k$ → $\mathcal{F}(i)$
3. **Reverse**：保留 $j \in \mathcal{F}(i)$ 且 $i$ 在 $j$ 的 top-$mnn_k$ 内
4. **Mutual**：$\mathcal{N}^{MNN}(i)=\{j: j\in\mathcal{F}(i),\ i\in\mathcal{R}(j)\}$
5. **Fallback**：若 $|\mathcal{N}^{MNN}(i)|=0$，用 $\mathcal{F}(i)$ 前 $k$ 个
6. **伪标签**：$\tilde{y}_i = \mathrm{mean}_{j\in\mathcal{N}^{MNN}(i)} y^{ref}_j$

默认：$k=5$，$mnn_k=20$。**实现**：`SpatialEx/SpatialEx_conditional_gt.py`

**讲点**：MNN 不是简单 kNN——要求**双向互为近邻**，过滤单向错误配对。

---

## 第 41 页：Strict MNN — Step 1 + Step 2 桥接（综合图）

**版式**：左侧简单流程图 + 右侧分步文字说明。

**左图**：自动生成/更新的 `docs/image/mnn_bridge_flowchart.svg`

- 上方：Slice 1 / Slice 2 输入（held-out 标红）
- 中间：Step 1（H&E Bridge，蓝）→ Step 2（Panel B Bridge，橙）
- 下方：MLP 训练框
- 核心信息：两次调用同一个 `build_mnn_pseudo`，全程不用 held-out

**右侧 — Step 1**（H&E Bridge，Slice1 pseudo-B）：

$$\tilde{Y}_B^1 = \mathrm{MNN}(X_1, X_2, Y_B^2)$$

- 目标：构造 Slice1 缺失 panel B 的 pseudo-label
- 匹配空间：UNI H&E 嵌入 $X_1 \leftrightarrow X_2$
- 转移目标：Slice2 已测的 $Y_B^2$

**右侧 — Step 2**（Panel B Bridge，Slice2 pseudo-A）：

$$\tilde{Y}_A^2 = \mathrm{MNN}(Y_B^2, \tilde{Y}_B^1, Y_A^1)$$

- 目标：构造 Slice2 缺失 panel A 的 pseudo-label
- 匹配空间：$Y_B^2$（Slice2 query）$\leftrightarrow$ $\tilde{Y}_B^1$（Step1 产物）
- 转移目标：Slice1 已测的 $Y_A^1$

> $\tilde{Y}_B^1$ 是 Step 1 **构造产物**，不是真实 $Y_B^1$。

**底部高亮**：核心在于两次调用同一个 `build_mnn_pseudo` 原子；Step 1 用形态找跨切片对应，Step 2 在 B-panel 空间 refine 对应。

---

## 第 42 页：Strict MNN 全流程 — Step 1 → Step 2 → MLP

**版式**：左图（纯结构 SVG）+ 右栏说明。

**左图**：`docs/image/fig3_diagnosis/13_mnn_architecture.svg`（**图 13：全流程**；左图蓝框=Step 1，橙框=Step 2）

**右栏 — Step 1（左图蓝框）**：

- $\tilde{Y}_B^1 = \mathrm{MNN}(X_1, X_2, Y_B^2)$
- H&E 跨切片匹配 → 构造 Slice1 pseudo-B

**右栏 — Step 2（左图橙框）**：

- $\tilde{Y}_A^2 = \mathrm{MNN}(Y_B^2, \tilde{Y}_B^1, Y_A^1)$
- B-panel 空间 refine → 构造 Slice2 pseudo-A

**右栏 — MLP 训练**：

- 输入：**仅 measured panel**（$Y_A^1$ / $Y_B^2$）
- 输出：missing panel 预测
- 损失：$\mathcal{L}=\|\hat{Y}_B^1-\tilde{Y}_B^1\|_2^2+\|\hat{Y}_A^2-\tilde{Y}_A^2\|_2^2$
- 训练器：`SpatialExP_ConditionalMLP`

### vs Cycle（同页右侧对照）

| 维度 | Cycle | Strict MNN |
|------|-------|------------|
| 监督来源 | 同切片自环 | 跨切片外部锚定 |
| MLP + 该监督 | 0.005 / 0.013 | **0.334 / 0.371** |

---

## 第 43 页：Fig.3 的搜索空间：编码器 × 监督信号

**版式**：上方说明 + 中间组合表 + 底部高亮。

**核心思想**：把 SpatialEx+ 解耦成「编码器」和「监督信号」两个可独立替换的模块，通过 shuffle 定位 Fig.3 的真正瓶颈。

**编码器**

- HGNN-512
- GT-128
- MLP

**监督信号**

- Cycle（同切片自环）
- Strict MNN（跨切片外部锚定）
- MNN + Cycle（验证 Cycle 是否干扰 MNN）
- Cycle only（去掉 MNN 的 ablation）

**组合表**

| 编码器 | 监督信号 | 实验意图 |
|--------|----------|----------|
| HGNN-512 | Cycle | 官方 SpatialEx+ baseline |
| GT-128 | Cycle | 改进一：换更强的图编码器 |
| MLP | Cycle only | Ablation：去掉 MNN 的外部监督 |
| MLP | Strict MNN | 改进二：换监督信号 |
| MLP | MNN + Cycle | 验证 Cycle 是否会干扰 MNN |
| GT-128 | Strict MNN | 同时换编码器 + 监督 |
| HGNN-512 | Strict MNN | 官方编码器 + 新监督 |

**底部高亮**

> 关键问题：瓶颈在编码器，还是在监督信号？
> 如果换 GT 无效而换 MNN 有效，说明监督信号才是 Fig.3 的主瓶颈。

---

## 第 44 页：Fig.3 主结果 — MLP + Strict MNN

**版式**：结果表 + 底部结论条。

| 编码器 | 监督 | S1 PCC | S2 PCC | S1 SSIM | S2 SSIM |
|--------|------|-------:|-------:|--------:|--------:|
| HGNN-512 | Cycle | 0.275 | 0.301 | 0.308 | 0.332 |
| MLP | Cycle only | 0.005 | 0.013 | 0.114 | 0.107 |
| **MLP** | **Strict MNN** | **0.334** | **0.371** | **0.374** | **0.398** |
| MLP | MNN + Cycle | 0.315 | 0.353 | 0.344 | 0.388 |

**讲点**：MNN + Cycle 略低于纯 MNN → Cycle 与外部监督冲突。

---

## 第 45 页：MNN 改进小结

### 为什么 Strict MNN 有效？

1. **外部锚定**：pseudo-label 来自跨切片真实 measured panel，而非自环
2. **空间信息已编码**：H&E MNN 配对形态/位置相似细胞；Panel B MNN 进一步配对表达相似细胞
3. **MNN 过滤降噪**：剔除单向 kNN 中的低质量跨切片配对
4. **任务匹配**：Fig.3 本质是 panel-to-panel 映射；MLP 足够，图模型额外聚合反而有害

### 与改进一的联系

- 改进一（GT）增强 H&E 空间编码 → 无效，因 H&E branch 弱
- 改进二（MNN）改善**监督信号** → 有效，因 panel branch dominant
- **核心洞察**：缺失 panel 任务中，监督信号质量 > 模型复杂度

**下一步配图**：
- `docs/image/fig3_diagnosis/02_mnn_sweep.svg` — 参数鲁棒性
- `docs/image/fig3_diagnosis/05_latent_alignment.svg` — Latent MNN 扩展

## 第 46 页：诊断图 — MNN 参数敏感性扫描

- 配图：`docs/image/fig3_diagnosis/02_mnn_sweep.svg`
- MNN 在一定超参数范围内鲁棒，不是凑出来的结果

---

## 第 47 页：跨切片 Latent Alignment 与 Latent MNN

- 配图：`docs/image/fig3_diagnosis/05_latent_alignment.svg`
- 比较 raw measured panel、PCA latent、CORAL aligned 三种 matching 空间：

| Matching space | Slice1 learned PCC | Slice2 learned PCC |
|----------------|-------------------:|-------------------:|
| raw measured panel | 0.015 | 0.264 |
| PCA latent (50-d) | 0.007 | **0.291** |
| CORAL aligned | 0.010 | 0.228 |

- PCA latent + MNN 在 Slice2 上进一步提升；CORAL 线性对齐反而下降

---

## 第 48 页：Official Split 主结果（PCC + SSIM + CMD）

| 编码器 | 监督 | Slice1 PCC | Slice2 PCC | Slice1 SSIM | Slice2 SSIM | Slice1 CMD | Slice2 CMD |
|--------|------|-----------:|-----------:|------------:|------------:|-----------:|-----------:|
| HGNN-512 | Cycle | 0.275 | 0.301 | 0.308 | 0.332 | 0.334 | 0.267 |
| DeepPT | — | 0.262 | 0.280 | 0.334 | 0.360 | 0.298 | 0.304 |
| GT-128 | Cycle | 0.267 | 0.276 | 0.345 | 0.357 | 0.296 | 0.289 |
| MLP | Cycle only | 0.005 | 0.013 | 0.114 | 0.107 | 0.780 | 0.788 |
| **MLP** | **Strict MNN** | **0.334** | **0.371** | **0.374** | **0.398** | **0.383** | **0.330** |
| MLP | MNN + Cycle | 0.315 | 0.353 | 0.344 | 0.388 | 0.355 | 0.323 |
| GT-128 | Strict MNN | 0.258 | 0.289 | 0.359 | 0.387 | 0.327 | 0.356 |
| HGNN-512 | Strict MNN | 0.234 | 0.273 | 0.072 | 0.055 | 0.347 | 0.369 |

- MLP + Strict MNN 显著最优；HGNN-512 + MNN 的 SSIM 极低（0.07 / 0.06）

---

## 第 49 页：诊断图 — PCC 分布对比

- 配图：`docs/image/fig3_diagnosis/10_pcc_distribution.svg`
- MLP + MNN 分布整体右移；Cycle 方法接近 0 或负值

---

## 第 50 页：诊断图 — Per-Gene 提升分布

- 配图：`docs/image/fig3_diagnosis/04_mnn_gain_distribution.svg`
- 大多数基因提升为正；78.7% 基因获得提升

---

## 第 51 页：Top Marker Genes 提升（HGNN+Cycle vs MLP+MNN）

**版式**：左侧柱状图 + 右侧 Top 5 表格 + 说明。

**左侧图**：`docs/image/top_marker_gains_hgnn_vs_mnn.svg`（由 `scripts/fig3/generate_top_marker_gains_chart.py` 根据 CSV 生成；样式仿 `09_top_marker_gains` 绿色横向柱状图）

- 横轴：$\Delta$PCC = MLP+MNN − HGNN+Cycle
- Top 10 Slice2 marker genes（完整数值见 `docs/a.md`）

**右侧表格**

| Gene | HGNN+Cycle | MLP+MNN | 提升 |
|------|-----------:|--------:|-----:|
| CPA3 | 0.079 | 0.578 | +0.498 |
| CTSG | 0.072 | 0.555 | +0.483 |
| CLEC14A | 0.330 | 0.638 | +0.308 |
| EGFL7 | 0.141 | 0.358 | +0.217 |
| CLDN5 | 0.046 | 0.262 | +0.216 |

**讲点**

- CPA3 / CTSG：肥大细胞 / 胰蛋白酶标记
- CLEC14A / EGFL7 / CLDN5：血管内皮相关
- MNN 的改进不只在平均指标上，也在真实空间表达结构上有视觉可辨的提升

---

## 第 52 页：结果图 — Slice2 Marker Gene 预测对比

- **配图**：`docs/image/fig3_diagnosis/11_marker_genes_slice2.jpg`（空间热力图，仿原 `visualize_marker_genes.py` 版式）
- **三列**：Ground truth | **HGNN+Cycle** | **MLP+MNN**（每行一个 marker，右侧 colorbar）
- **默认基因**：Top 4 ΔPCC（CPA3, CTSG, CLEC14A, EGFL7）；可改 `--genes CTLA4 PTPRC ...`
- **生成**：`python scripts/fig3/visualize_marker_genes_hgnn_vs_mnn.py`

---

## 第 53 页：章节分隔 — 六、讨论：为什么 MLP + MNN 反而最强？

**过渡语**：为什么简单 MLP 能打败复杂网络？

---

## 第 54 页：为什么 MLP + MNN 反而最强？

1. 任务本质是 panel-to-panel 分子映射，不是空间聚合
2. MNN 已通过跨切片 matching 编码了空间/形态信息
3. HGNN/GT 的额外空间聚合会 over-smooth 或引入 H&E 噪声
4. Cycle 与 MNN 冲突：自洽约束干扰可靠的外部监督
5. HGNN-512 + MNN 的 SSIM 极低，说明图模型在空间结构上严重失真
6. 当监督信号强时，简单 MLP 足够

---

## 第 54 页之二：为什么 MLP 能超越 MNN pseudo-label 本身？

**版式**：上方对比表 + 下方解释文字 + 右侧/底部流程示意

### 实验结果对比

| 构造方式 | pseudo-label 质量（S1 PCC / S2 PCC） | 训练后 MLP（S1 PCC / S2 PCC） | 提升 |
|---|---|---|---|
| raw cosine kNN | 0.178 / 0.252 | 0.327 / 0.381 | +0.149 / +0.129 |
| Strict MNN | 0.153 / 0.187 | 0.334 / 0.371 | +0.181 / +0.184 |

> pseudo-label 质量：直接把 kNN/MNN 平均结果当作预测，在 held-out panel 上评估的“ceiling”。
> 训练后 MLP：以 pseudo-label 为监督，学习 `measured panel → missing panel` 的映射后再预测。

### 解释

1. **pseudo-label 是带噪的 per-cell 估计**
   - 每个细胞的 pseudo-label 来自跨切片 H&E 或 panel 匹配的少数邻居平均
   - 细胞级噪声大；Slice2 还需通过 pseudo_B1 做两步桥接，误差进一步累积

2. **MLP 学到的是全局 panel-to-panel 映射**
   - 输入：同一切片的 measured panel（强信号）
   - 目标：pseudo-label（有噪监督）
   - 结果：MLP 把局部噪声平均掉，学到 `Y_A → Y_B` 的稳定相关结构

3. **MLP 绕过了两步桥接**
   - raw/MNN 构造 Slice2 的 pseudo_A2：`Y_B2 → pseudo_B1 → Y_A1`
   - MLP 直接学习：`Y_B2 → Y_A2`，避免中间误差放大

### 结论

MLP + Strict MNN 的强势不是来自模型复杂，而是来自「用简单模型把有噪但方向正确的 pseudo-label 蒸馏成稳定、可直接从 measured panel 使用的 panel-to-panel translator」。

---

## 第 55 页：结论

- 复现：跑通 SpatialEx/SpatialEx+（Fig.2–3），修复 6 个官方代码 bug
- 复现：补充 Fig.4 百万细胞与 Fig.5 Omics Diagonal 初步结果
- 复现：Fig.2 baseline 中 HGNN 在纯 H&E 任务上仍优于 GT
- 诊断：Branch Decomposition 揭示 H&E branch 弱、panel branch dominant
- 改进：Strict MNN 伪标签替代 Cycle，提供可靠外部监督
- 发现：MLP + Strict MNN 效果最佳（PCC 0.334 / 0.371，SSIM 0.374 / 0.398）
- 核心洞察：缺失 panel 的监督信号质量比模型复杂度更重要

---

## 第 56 页：局限与后续方向

| 局限 | 后续方向 |
|------|----------|
| Slice1 方向仍较弱（与 official split 信息桥不对称有关） | 加入位置编码的多模态 matching |
| 仅单一数据集验证 | 在更多癌种和组织上验证 |
| 未与 DeepPT 在 Fig.3 对比；CNN_Reg、Hist2ST 待实现 | 与更多 baseline 公平比较 |
| GT 显存限制，Cross-Attention Translator 退化为 MLP | 开发更轻量 GT 变体 |
| MNN pseudo-label 有噪声但未被建模 | MNN 配对不确定性建模 |
| Fig.4 评估脚本仍有 coo_matrix subscript bug | 修复评估脚本并系统扩展 Omics Diagonal |

---

## 第 58 页：Appendix — Fig.4 / Fig.5 上的 Strict MNN 探索

- 表格：Fig.5 Omics Diagonal MNN 结果（pseudo ceiling vs MLP + Strict MNN）
- 说明：Fig.4 百万细胞 MNN 尚未系统运行（计算/内存开销大），作为后续方向
- 结论：MNN 在 protein→RNA 略优，RNA→protein 因 marker 过少接近随机

---

## 第 59 页：附录 + 谢谢 / Q&A

- 可引用/展示的论文原图：
  - `Fig4`：百万细胞 Panel Diagonal 可扩展性
  - `Fig5`：Omics Diagonal Integration（转录组 ↔ 蛋白/代谢）
  - `Fig6`：ablation 与扩展分析
  - `Extended Data Fig.1–10`：方法细节与补充实验
- 讲点：
  - 论文 Fig.5–6 展示了 Omics Diagonal 等更复杂任务
  - 我们的 Strict MNN 思想可推广到这些场景
  - 未来可借鉴 Extended Data 中的更多分析
- 谢谢！欢迎各位老师提问
