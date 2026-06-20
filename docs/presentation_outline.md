# 答辩 PPT 大纲（44 页完整版）

> 对应文件：`docs/SpatialEx_Defense.pptx`  
> 目标页数：**44 页**  
> 风格：保留原有章节分隔页 + 深蓝/橙色配色 + 顶部蓝色横条
>
> 注：`docs/report/images/` 目录不存在，本大纲使用项目中实际可用的图片：
> - `docs/image/fig3_diagnosis/`（11 张诊断/实验图）
> - `docs/image/from_paper/`（论文原图 Fig.1–6 及 Extended Data Fig.1–10）
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

1. 背景与任务：Spatial Diagonal Integration & Fig.3
2. Fig.3 任务数学化：符号、Strict 协议、评价指标
3. 环境搭建与复现：Fig.2 baseline、Fig.3 跑通、修复的 6 个 bug
4. 改进一：Graph Transformer 替代 HGNN
5. 诊断：Branch Decomposition 与信号来源分析
6. 改进二：Strict MNN 伪标签替代 Cycle
7. 实验结果：Official Split、per-gene、空间可视化
8. 讨论：为什么 MLP + MNN 反而最强？
9. 结论、局限与展望

---

## 第 3 页：章节分隔 — 一、背景与任务

**过渡语**：先介绍我们要解决什么问题，以及这个问题为什么难。

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

## 第 5 页：论文原图 — 空间多组学整合概览

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
| Fig.3 任务 | Fig.4/5 等任务 |
| **本工作聚焦于此** | 本工作尚未复现 |

---

## 第 7 页：论文原图 — Fig.3 Panel Diagonal Integration

- 配图：`docs/image/from_paper/Fig3_41592_2025_2926_Fig3_HTML.png`
- 用途：展示原文 Fig.3 的实验设计和官方结果
- 讲点：原文使用两张相邻乳腺癌切片，官方方法通过 Cycle 约束做 cross-panel 预测

---

## 第 8 页：章节分隔 — 二、Fig.3 任务数学化

**过渡语**：把 Fig.3 任务用符号严格化，明确什么可用、什么不可用。

---

## 第 9 页：Fig.3 Strict 协议与评价指标

- 符号设定：
  - Slice 1：H&E 嵌入 $X_1$，已测 panel A $Y_A^1$，缺失 panel B $Y_B^1$
  - Slice 2：H&E 嵌入 $X_2$，已测 panel B $Y_B^2$，缺失 panel A $Y_A^2$
- 目标：$\hat{Y}_B^1 \approx Y_B^1$，$\hat{Y}_A^2 \approx Y_A^2$
- 严格限制：训练时不可用 held-out panel（$Y_B^1$、$Y_A^2$）
- 可用信号来源：
  1. H&E 形态先验
  2. 跨切片 pseudo-label
  3. 自监督约束（Cycle / DGI）
- 评价指标：
  - gene-level PCC（主指标）
  - SSIM（空间结构保真）
  - CMD（中心矩差异）

---

## 第 10 页：复现图 — Fig.3 Strict 协议数据划分

- 配图：`docs/image/fig3b.png`
- 左侧 Slice1：$X_1 + Y_A^1 \to$ predict $Y_B^1$
- 右侧 Slice2：$X_2 + Y_B^2 \to$ predict $Y_A^2$
- 红色框表示 held-out、训练不可用的 panel

---

## 第 11 页：章节分隔 — 三、环境搭建与复现

**过渡语**：先把官方代码跑通，并补充 Fig.2 baseline 对比。

---

## 第 12 页：SpatialEx / SpatialEx+ 概述

- SpatialEx：UNI + HGNN + DGI
  - UNI：病理图像预训练视觉大模型
  - HGNN：超图空间聚合
  - DGI：自监督对比损失
- SpatialEx+：加入 Omics Cycle Module
  - 两个 panel-specific encoder
  - Cycle 约束：$Y_A \to \hat{Y}_B \to Y_A' \approx Y_A$
- Cycle 问题：只保证自洽，不保证预测接近真实 missing panel

---

## 第 13 页：论文原图 — SpatialEx/SpatialEx+ 方法框架

- 配图：`docs/image/from_paper/Fig2_41592_2025_2926_Fig2_HTML.png`
- (a) 输入；(b) Hypergraph 构建；(c) HGNN + DGI；(d) Cycle Module

---

## 第 14 页：复现中修复的 6 个关键 Bug

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| 1 | `preprocess.normalize_graph` | 未定义变量 `adj` | 图归一化失败 |
| 2 | `preprocess` | `'crs'` 拼写错误 | sparse 转换失败 |
| 3 | `Build_hypergraph_spatial_and_HE` | 默认返回 `coo` | 子图索引报错 |
| 4 | `Model_Plus.forward` | `agg_mtx` 维度不匹配 | forward 报错 |
| 5 | `SpatialExP.train` | `Regression` 维度不匹配 | Cycle 训练崩溃 |
| 6 | `Model_Plus / Regression` | BatchNorm batch=1 崩溃 | 小 batch 中断 |

---

## 第 15 页：Fig.2 复现 — 任务设定

- Fig.2 与 Fig.3 的区别：
  - Fig.2：输入仅为 H&E，输出为全转录组（313 genes）
  - 不涉及 panel split 或随机留基因
  - 难度更高：模型必须 purely 从 H&E 推断表达
- 数据：Rep1 / Rep2，各含 UNI H&E 嵌入与 Xenium RNA 表达
- 协议：双向跨切片 train→test
  - Slice A 训练 → Slice B 全部细胞评估
  - 反向再做一次

---

## 第 16 页：Fig.2 复现 — 主结果

| 方法 | Slice1 PCC | Slice2 PCC | Slice1 SSIM | Slice2 SSIM | Slice1 CMD | Slice2 CMD |
|------|-----------:|-----------:|------------:|------------:|-----------:|-----------:|
| HGNN SpatialEx (512) | **0.257** | **0.273** | **0.419** | **0.425** | **0.205** | **0.207** |
| DeepPT | 0.268 | 0.276 | 0.357 | 0.366 | 0.271 | 0.281 |
| GT-512 + MFP | 0.244 | 0.246 | 0.339 | 0.329 | 0.234 | 0.241 |
| GT-128 + MFP | 0.228 | 0.236 | 0.309 | 0.320 | 0.239 | 0.252 |
| GT-128 + DGI | 0.225 | 0.235 | 0.297 | 0.313 | 0.238 | 0.248 |

- 结论：Fig.2 纯 H&E→omics 任务上 HGNN 仍优于 GT；Fig.3 上 GT≈HGNN 是因为 H&E branch 弱。

---

## 第 17 页：章节分隔 — 四、改进一：网络方面

**过渡语**：复现跑通后，先尝试改进网络架构，看是否能把 HGNN 换成更强的 GT。

---

## 第 18 页：改进一 — Graph Transformer 替代 HGNN

- 动机：HGNN 超边聚合权重固定；GT 可用稀疏自注意力学习自适应权重
- 实现：Graph Transformer + MFP 替代 HGNN + DGI
- Fig.3 结果（Cycle 监督下）：
  - HGNN-512 Cycle: 0.275 / 0.301
  - GT-128 Cycle: 0.267 / 0.276
- 结论：GT 与 HGNN 基本持平，网络架构不是 Fig.3 瓶颈

---

## 第 19 页：诊断图 — 信号贡献分解

- 配图：`docs/image/fig3_diagnosis/08_signal_contribution.png`
- measured panel 贡献占主导；H&E 贡献接近零甚至为负
- 为“网络不是瓶颈，监督信号才是关键”提供直接证据

---

## 第 20 页：GT 改进小结

- HGNN → GT 没有显著提升
- 结合 Fig.2 结果：在纯 H&E 任务上 HGNN 仍优于 GT
- 说明 Fig.3 上 GT≈HGNN 不是因为网络都够强，而是因为 H&E 这条输入本身信息量低
- 下一步：诊断 H&E 和 measured panel 各自的作用

---

## 第 21 页：章节分隔 — 五、有效信号诊断

**过渡语**：网络不是瓶颈，我们开始显式诊断有效信号来源。

---

## 第 22 页：诊断 — Branch Decomposition

- 实验设计：把输入拆成两条分支
  - H&E branch：只用 H&E 嵌入预测缺失 panel
  - Panel branch：只用 measured panel 预测缺失 panel
- 结果：

| Variant | Slice1 PCC | Slice2 PCC |
|---------|-----------:|-----------:|
| H&E branch | -0.001 | 0.010 |
| Panel branch | 0.016 | 0.248 |
| 0.5 average | 0.011 | 0.177 |
| Reliability-weighted | -0.001 | 0.010 |

- 结论：H&E branch 接近 0；panel branch dominant；late fusion 有害

---

## 第 23 页：诊断图 — Branch Decomposition 可视化

- 配图：`docs/image/fig3_diagnosis/01_branch_decomposition.png`
- H&E branch 散点接近随机；panel branch 有明显线性趋势

---

## 第 24 页：诊断图 — 信号贡献定量分析

- 配图：`docs/image/fig3_diagnosis/08_signal_contribution.png`
- 条形图/热图展示各信号来源贡献比例
- measured panel 贡献远高于 H&E

---

## 第 25 页：诊断图 — Per-Gene 预测散点

- 配图：`docs/image/fig3_diagnosis/03_per_gene_scatter.png`
- 大多数基因上 MNN 监督优于 Cycle 监督

---

## 第 26 页：章节分隔 — 六、改进二：监督信号方面

**过渡语**：H&E 和 Cycle 都不够强，转向跨切片 pseudo-label。

---

## 第 27 页：Cycle Consistency 的自洽陷阱

- Cycle 约束：$Y_A \to \hat{Y}_B \to Y_A' \approx Y_A$
- 问题：只保证自洽，不保证 $\hat{Y}_B \approx Y_B$
- 如果 $HE \to Y_A$ 学到幻觉，Cycle 会加强幻觉
- 实验：MLP + Cycle only PCC 仅 0.005 / 0.013，接近随机

---

## 第 28 页：诊断图 — Cycle Self-Consistency Trap

- 配图：`docs/image/fig3_diagnosis/07_cycle_trap.png`
- Cycle 让模型训练时自我闭环，测试时暴露为随机或偏差

---

## 第 29 页：改进二 — Strict MNN 伪标签

- 核心思想：用跨切片 Mutual Nearest Neighbor 构造 pseudo-label
- 严格限制：全程不使用 held-out panel
- 两步桥接：
  - Step 1：H&E MNN $X_1 \leftrightarrow X_2$，转移 $Y_B^2 \to \tilde{Y}_B^1$
  - Step 2：B-panel MNN $Y_B^2 \leftrightarrow \tilde{Y}_B^1$，转移 $Y_A^1 \to \tilde{Y}_A^2$
- MLP 输入 measured panel，监督为 strict MNN pseudo-label

---

## 第 30 页：流程图 — Strict MNN 伪标签流程

- 配图：`docs/image/fig3_diagnosis/06_mnn_pipeline.png`
- H&E MNN bridge → Panel B MNN bridge → 伪标签 → 监督 MLP

---

## 第 31 页：诊断图 — MNN 参数敏感性扫描

- 配图：`docs/image/fig3_diagnosis/02_mnn_sweep.png`
- MNN 在一定超参数范围内鲁棒，不是凑出来的结果

---

## 第 32 页：跨切片 Latent Alignment 与 Latent MNN

- 配图：`docs/image/fig3_diagnosis/05_latent_alignment.png`
- 比较 raw measured panel、PCA latent、CORAL aligned 三种 matching 空间：

| Matching space | Slice1 learned PCC | Slice2 learned PCC |
|----------------|-------------------:|-------------------:|
| raw measured panel | 0.015 | 0.264 |
| PCA latent (50-d) | 0.007 | **0.291** |
| CORAL aligned | 0.010 | 0.228 |

- PCA latent + MNN 在 Slice2 上进一步提升；CORAL 线性对齐反而下降

---

## 第 33 页：章节分隔 — 七、实验结果

**过渡语**：把 Strict MNN 接到不同模型上，全面评估效果。

---

## 第 34 页：Official Split 主结果（PCC + SSIM）

| 编码器 | 监督 | Slice1 PCC | Slice2 PCC | Slice1 SSIM | Slice2 SSIM |
|--------|------|-----------:|-----------:|------------:|------------:|
| HGNN-512 | Cycle | 0.275 | 0.301 | 0.308 | 0.332 |
| GT-128 | Cycle | 0.267 | 0.276 | 0.345 | 0.357 |
| MLP | Cycle only | 0.005 | 0.013 | 0.114 | 0.107 |
| **MLP** | **Strict MNN** | **0.334** | **0.371** | **0.374** | **0.398** |
| MLP | MNN + Cycle | 0.315 | 0.353 | 0.344 | 0.388 |
| GT-128 | Strict MNN | 0.258 | 0.289 | 0.359 | 0.387 |
| HGNN-512 | Strict MNN | 0.234 | 0.273 | 0.072 | 0.055 |

- MLP + Strict MNN 显著最优；HGNN-512 + MNN 的 SSIM 极低（0.07 / 0.06）

---

## 第 35 页：诊断图 — PCC 分布对比

- 配图：`docs/image/fig3_diagnosis/10_pcc_distribution.png`
- MLP + MNN 分布整体右移；Cycle 方法接近 0 或负值

---

## 第 36 页：诊断图 — Per-Gene 提升分布

- 配图：`docs/image/fig3_diagnosis/04_mnn_gain_distribution.png`
- 大多数基因提升为正；78.7% 基因获得提升

---

## 第 37 页：Top Marker Genes 提升

- 提升显著的 marker 基因：

| Gene | raw kNN PCC | MNN PCC | 提升 |
|------|------------:|--------:|-----:|
| CTLA4 | 0.132 | 0.332 | +0.200 |
| PTPRC | -0.019 | 0.135 | +0.154 |
| ESR1 | 0.197 | 0.323 | +0.126 |
| CLEC14A | 0.453 | 0.590 | +0.137 |

- 涉及免疫、上皮/基质、血管等生物学过程
- 配图：`docs/image/fig3_diagnosis/09_top_marker_gains.png`

---

## 第 38 页：结果图 — Slice2 Marker Gene 预测对比

- 配图：`docs/image/fig3_diagnosis/11_marker_genes_slice2.jpg`
- Ground Truth vs MLP+Strict MNN vs HGNN/GT+Cycle
- MNN 预测的热点区域与真实值对齐更好

---

## 第 39 页：章节分隔 — 八、讨论

**过渡语**：为什么简单 MLP 能打败复杂网络？

---

## 第 40 页：为什么 MLP + MNN 反而最强？

1. 任务本质是 panel-to-panel 分子映射，不是空间聚合
2. MNN 已通过跨切片 matching 编码了空间/形态信息
3. HGNN/GT 的额外空间聚合会 over-smooth 或引入 H&E 噪声
4. Cycle 与 MNN 冲突：自洽约束干扰可靠的外部监督
5. HGNN-512 + MNN 的 SSIM 极低，说明图模型在空间结构上严重失真
6. 当监督信号强时，简单 MLP 足够

---

## 第 41 页：结论

- 复现：跑通 SpatialEx/SpatialEx+，修复 6 个官方代码 bug
- 复现：补充 Fig.2 baseline，HGNN 在纯 H&E 任务上仍优于 GT
- 诊断：Branch Decomposition 揭示 H&E branch 弱、panel branch dominant
- 改进：Strict MNN 伪标签替代 Cycle，提供可靠外部监督
- 发现：MLP + Strict MNN 效果最佳（PCC 0.334 / 0.371，SSIM 0.374 / 0.398）
- 核心洞察：缺失 panel 的监督信号质量比模型复杂度更重要

---

## 第 42 页：局限与后续方向

| 局限 | 后续方向 |
|------|----------|
| Slice1 方向仍较弱（与 official split 信息桥不对称有关） | 加入位置编码的多模态 matching |
| 仅单一数据集验证 | 在更多癌种和组织上验证 |
| 未与 DeepPT 在 Fig.3 对比；CNN_Reg、Hist2ST 待实现 | 与更多 baseline 公平比较 |
| GT 显存限制，Cross-Attention Translator 退化为 MLP | 开发更轻量 GT 变体 |
| MNN pseudo-label 有噪声但未被建模 | MNN 配对不确定性建模 |
| 仅做了 Panel Diagonal | 扩展到 Omics Diagonal Integration |

---

## 第 43 页：附录 — 论文其他 Fig. 与扩展方向

- 可引用/展示的论文原图：
  - `Fig4`：Omics Diagonal Integration
  - `Fig5`：更多应用场景
  - `Fig6`：ablation 与扩展分析
  - `Extended Data Fig.1–10`：方法细节与补充实验
- Strict MNN 思想可推广到 Omics Diagonal 等更复杂任务

---

## 第 44 页：谢谢 / Q&A

- 谢谢！
- 请各位老师斧正 / 欢迎提问
