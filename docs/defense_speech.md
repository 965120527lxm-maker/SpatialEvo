# 答辩演讲稿（初稿）

> 对应 PPT：`docs/SpatialEx_Defense.pptx`（44 页）  
> 预计时长：约 40 分钟  
> 本稿按幻灯片顺序撰写，可根据实际节奏删减。

---

## 开场白（第 1–2 页，约 1 分钟）

各位老师好！我是李熹鸣，今天汇报的题目是《SpatialEx/SpatialEx+ 复现与改进——Fig.3 Panel Diagonal Integration 上的监督信号探索》。

这项工作基于 Nature Methods 2025 的论文 *High-parameter spatial multi-omics through histology-anchored integration*。我们不仅复现了原文的 SpatialEx 和 SpatialEx+，还针对 Fig.3 的 panel diagonal 任务做了一系列诊断和改进。最终我们发现一个有点反直觉的结论：**简单 MLP 加上 strict MNN 伪标签，反而比复杂的 HGNN 或 Graph Transformer 效果更好**。今天我按这个线索向各位老师汇报。

---

## 第一章：背景与任务（第 3–7 页，约 4 分钟）

### 第 3 页：章节分隔 — 一、背景与任务

首先介绍我们要解决什么问题。

### 第 4 页：背景

空间组学技术有一个根本性的权衡：分辨率、通量、panel 大小，这三者很难同时满足。比如 Xenium、MERFISH、CosMx 这些高分辨率技术，通常只能测几百个基因；而 Visium 这种大 panel 技术，分辨率又比较低。同一张组织切片上，我们很难同时拿到高分辨率、大 panel 和多组学信息。

于是一个自然的想法是：在相邻切片上分别测互补的信息，再通过计算整合。比如切片 1 测 panel A，切片 2 测 panel B；或者切片 1 测转录组，切片 2 测蛋白组。这就引出了 **Spatial Diagonal Integration**：两个切片没有共测的 omics 特征，需要跨切片补全。

这和传统的 horizontal integration（同特征空间）以及 vertical integration（同一细胞多模态）都不一样，它的难点在于没有共享特征，必须借助空间位置或形态先验来做桥接。

### 第 5 页：论文原图 Fig.1

这张图展示了原文的整体框架。左边是不同的空间组学平台，右边是 SpatialEx 作为 histology-anchored integration 的核心方法。我们的工作主要聚焦在右下角这个 Fig.3 的 panel diagonal 场景。

### 第 6 页：两种 Diagonal Integration

Diagonal integration 分为两类。一类是 **Panel Diagonal**：同一切片技术，但测不同的基因 panel，也就是 Fig.3 的任务。另一类是 **Omics Diagonal**：不同组学类型，比如转录组加蛋白，对应 Fig.4/5。我们这次主要聚焦 Panel Diagonal，因为它更直接地揭示了“缺失 panel 的监督信号”这个核心问题。

### 第 7 页：论文原图 Fig.3

Fig.3 使用了两张相邻的乳腺癌 Xenium 切片，Rep1 和 Rep2，各自测了互补的基因 panel。原文的 SpatialEx+ 通过 Cycle consistency 约束来做 cross-panel 预测。我们的目标是对这个设定做严格复现，并重新审视它的方法和结论。

---

## 第二章：Fig.3 任务数学化（第 8–10 页，约 3 分钟）

### 第 8 页：章节分隔

下面把 Fig.3 任务用符号严格化。

### 第 9 页：Strict 协议与评价指标

我们有两个相邻切片。Slice 1 有 H&E 特征 X1、已测 panel A YA1、缺失 panel B YB1；Slice 2 有 H&E 特征 X2、已测 panel B YB2、缺失 panel A YA2。我们的目标是预测 YB1 和 YA2。

关键在于 **strict 协议**：训练时绝对不能用 held-out panel，也就是 YB1 和 YA2。如果直接用它们监督，问题就退化为普通的有监督回归，失去研究意义。

所以模型只能依赖三种信号：H&E 形态先验、跨切片 pseudo-label、以及自监督约束比如 Cycle 或 DGI。评价指标主要用 gene-level PCC，辅以 SSIM 和 CMD。

### 第 10 页：Fig.3 Strict 协议数据划分

这张图很直观地展示了数据划分。Slice 1 用 X1 加 YA1 预测 YB1，Slice 2 用 X2 加 YB2 预测 YA2。红色框表示训练不可用的 held-out panel。

---

## 第三章：环境搭建与复现（第 11–16 页，约 6 分钟）

### 第 11 页：章节分隔

接下来介绍复现工作。我们首先要把官方代码跑通，并补充必要的 baseline 对比。

### 第 12 页：SpatialEx / SpatialEx+ 概述

SpatialEx 的框架是：用 UNI 这个病理图像预训练模型提取 H&E 嵌入，然后构建空间超图，用 HGNN 做空间聚合，再用 DGI 做自监督对比学习，最终预测基因表达。

SpatialEx+ 在此基础上加入了 Omics Cycle Module：两个 panel-specific encoder 分别预测 panel A 和 panel B，中间加入 mapping heads 实现 YA 和 YB 的循环映射。Cycle 约束是 YA 预测 YB 再预测回 YA，应该和原始 YA 接近。

但这里有一个潜在问题：Cycle 只保证自洽，不保证中间预测的 YB 接近真实 YB。

### 第 13 页：论文原图 Fig.2

这是原文的方法框架图。从左到右分别是输入、超图构建、HGNN+DGI 训练、以及 SpatialEx+ 的 Cycle Module。我们后续的网络改进主要围绕 HGNN 和 Cycle 这两个模块展开。

### 第 14 页：6 个关键 Bug

复现过程中我们修复了 6 个官方代码里的阻塞性 bug，涉及图归一化、coo/csr 类型错误、agg_mtx 维度不匹配、Regression 维度不匹配、以及 BatchNorm batch=1 崩溃等问题。这些 bug 都在官方原始仓库中验证过，是跑通 Fig.3 主实验的前提。

### 第 15 页：Fig.2 复现任务设定

除了 Fig.3，我们还补充了 Fig.2 的复现。Fig.2 和 Fig.3 有本质区别：Fig.2 输入只有 H&E，输出是全转录组 313 个基因，不涉及 panel split。这意味着模型必须纯粹从 H&E 推断表达，难度更高。

我们采用双向跨切片协议：在 Slice A 上训练，在 Slice B 全部细胞上评估，然后再反向做一次。

### 第 16 页：Fig.2 主结果

这是 Fig.2 的结果。可以看到，HGNN SpatialEx 在 SSIM 和 CMD 上明显最好，说明它保留空间结构和基因-基因关系的能力更强；DeepPT 的 PCC 略高；GT 系列无论是 128 维还是 512 维、MFP 还是 DGI，PCC 都低于 HGNN。

这个结果很重要：它说明在纯 H&E→omics 任务上，HGNN 仍然优于 GT。这与 Fig.3 上 GT 和 HGNN 持平的现象形成对照，暗示 Fig.3 的问题可能不在网络本身，而在于任务设定中 H&E 这条输入的信息量不够。

---

## 第四章：改进一——网络架构（第 17–20 页，约 5 分钟）

### 第 17 页：章节分隔

复现跑通后，我们首先怀疑网络架构是不是瓶颈，于是尝试把 HGNN 换成 Graph Transformer。

### 第 18 页：GT 替代 HGNN

动机很明确：HGNN 的超边聚合权重是固定的，由超图结构决定；而 Graph Transformer 可以用稀疏自注意力学习“哪些邻居更重要”，理论上更能适应不同组织区域的空间异质性。

实现上，我们用 Graph Transformer Encoder 替代 HGNN，用 Masked Feature Prediction 替代 DGI。但在 Cycle 监督下，HGNN-512 的 PCC 是 0.275/0.301，GT-128 是 0.267/0.276，基本持平，没有优势。

### 第 19 页：信号贡献分解

这张图定量展示了不同输入信号的贡献。可以看到，measured panel 的贡献占主导，H&E 的贡献接近零甚至为负。这直接支持了“网络不是瓶颈，监督信号才是关键”的判断。

### 第 20 页：GT 改进小结

结合 Fig.2 的结果，我们可以得出结论：把 HGNN 换成 GT 没有提升，不是因为图网络不够强，而是因为 Fig.3 任务中 H&E 这条输入本身信息量低。下一步应该显式诊断 H&E 和 measured panel 各自的作用。

---

## 第五章：有效信号诊断（第 21–25 页，约 4 分钟）

### 第 21 页：章节分隔

既然网络不是瓶颈，我们开始诊断真正有效的信号来自哪里。

### 第 22 页：Branch Decomposition

我们设计了一个 Branch Decomposition 实验：把模型输入显式拆成两条独立分支。H&E branch 只用 H&E 嵌入预测缺失 panel；Panel branch 只用同一切片的 measured panel 预测缺失 panel。我们还测试了简单平均融合和 reliability-weighted 融合。

结果非常清楚：H&E branch 的 PCC 接近 0，Panel branch 在 Slice2 上达到 0.248， late fusion 反而把性能拉低。这说明有效信号主要来自 measured panel 到 missing panel 的分子映射，H&E 在此任务中更像噪声而非补充。

### 第 23 页：Branch Decomposition 可视化

这张图更直观地展示了结果。H&E branch 的预测散点接近随机，Panel branch 则有明显的线性趋势。加入 H&E 后性能下降，说明 H&E 嵌入引入了与目标无关的方差。

### 第 24 页：信号贡献定量分析

从另一个角度验证同样的结论：measured panel 的贡献远高于 H&E，late fusion 无法提升性能。

### 第 25 页：Per-Gene 预测散点

这张图展示不同方法在每个基因上的预测相关性。大多数基因上，MNN 监督都优于 Cycle 监督，只有极少数基因 Cycle 略好。这说明不存在单一方法在所有基因上都最优。

---

## 第六章：改进二——监督信号（第 26–32 页，约 8 分钟）

### 第 26 页：章节分隔

H&E 和 Cycle 都不够强，我们转向第三种信号来源：跨切片 pseudo-label。

### 第 27 页：Cycle Consistency 的自洽陷阱

先解释一下为什么 Cycle 不够。Cycle 约束是 YA 预测 YB 再预测回 YA，要求它和原始 YA 接近。但问题是，这个循环只保证自洽，不保证中间的 YB 接近真实 YB。

如果模型学到一种幻觉映射能让循环闭合，Cycle loss 就会满意。我们做过实验：MLP + Cycle only 的 PCC 只有 0.005/0.013，接近随机。这说明 Cycle 本身几乎不提供有效监督。

### 第 28 页：Cycle Trap 图

这张图展示了 Cycle 的陷阱：训练时模型可以自我闭环，但测试时预测可能与真实 panel 无关。这是 Cycle 作为唯一监督信号的根本缺陷。

### 第 29 页：Strict MNN 伪标签

于是我们提出 Strict MNN 伪标签。核心思想是用跨切片的 Mutual Nearest Neighbor 构造 pseudo-label，并且全程不使用 held-out panel。

具体是两步桥接：第一步，在 H&E 嵌入空间做跨切片 MNN，把 Slice 2 已测的 YB2 转移给 Slice 1，得到 YB1 的伪标签；第二步，在 panel B 空间做 MNN，把 Slice 1 已测的 YA1 转移给 Slice 2，得到 YA2 的伪标签。然后用 MLP 输入 measured panel，以 strict MNN pseudo-label 作为监督。

### 第 30 页：MNN 流程图

这张图展示了两步桥接的流程：先用 H&E MNN bridge 建立形态-空间对应，再用 Panel B MNN bridge 建立表达对应，最后生成伪标签监督 MLP。

### 第 31 页：MNN 参数敏感性

这张图展示 MNN 超参数的敏感性扫描。可以看到 MNN 在一定范围内是鲁棒的，不是依赖某个特定超参数凑出来的结果。

### 第 32 页：Latent Alignment 与 Latent MNN

我们还尝试了不同的 matching 空间。在 raw measured panel 上，Slice2 学到 0.264；投影到 50 维 PCA latent 后，提升到 0.291；但用 CORAL 线性对齐反而下降到 0.228。这说明降噪后的低维空间有助于跨切片匹配，但过度对齐可能抹平生物学相关的切片间差异。Slice1 始终接近 0，说明这个方向的信息桥本身较弱。

---

## 第七章：实验结果（第 33–38 页，约 6 分钟）

### 第 33 页：章节分隔

下面我们把 Strict MNN 接到不同模型上，全面评估效果。

### 第 34 页：Official Split 主结果

这是最重要的结果表。可以看到，MLP + Strict MNN 在 PCC 和 SSIM 上都显著优于官方 HGNN/GT + Cycle：PCC 达到 0.334/0.371，SSIM 达到 0.374/0.398。

更有意思的是，加上 Cycle 后 MNN 性能反而下降，说明 Cycle 与 MNN 存在冲突。另外，HGNN-512 + MNN 的 SSIM 极低，只有 0.07/0.06，说明图模型在空间结构上严重失真；而 MLP + MNN 的空间保真最好。

### 第 35 页：PCC 分布

从分布上看，MLP + MNN 的 PCC 分布整体右移，Cycle 方法则接近 0 或负值。

### 第 36 页：Per-Gene 提升分布

大多数基因的 ΔPCC 为正，78.7% 的基因通过 MNN 获得提升。少数基因下降，说明仍有改进空间。

### 第 37 页：Top Marker Genes

提升最显著的 marker 基因包括 CTLA4、PTPRC、ESR1、CLEC14A，涉及免疫、上皮/基质、血管等生物学过程。例如 CTLA4 从 0.132 提升到 0.332，提升 0.2。这说明 MNN 的改进不仅在平均指标上，也在真实生物学关键基因上。

### 第 38 页：Marker Gene 空间可视化

这张图对比了 Ground Truth、MLP+Strict MNN 和 HGNN/GT+Cycle 的空间表达。MNN 预测的热点区域与真实值对齐更好，Cycle 预测则有过平滑或错位的现象。

---

## 第八章：讨论（第 39–44 页，约 5 分钟）

### 第 39 页：章节分隔

现在回答一个核心问题：为什么简单 MLP 能打败复杂网络？

### 第 40 页：为什么 MLP + MNN 最强

我们有六点解释：

第一，任务本质是 panel-to-panel 分子映射，不是空间聚合。同一切片上 measured panel 和 missing panel 之间存在生物学相关性。

第二，MNN 本身已经通过跨切片 matching 编码了空间和形态信息。H&E MNN 配对形态相似的细胞，Panel B MNN 进一步配对表达相似的细胞。

第三，HGNN/GT 的额外空间聚合会引入 H&E 噪声或 over-smoothing。Branch Decomposition 已经显示 H&E 分支接近 0。

第四，Cycle 与 MNN 冲突。MNN 是外部锚定的监督，Cycle 是自洽约束，当 MNN 可靠时，Cycle 会把模型拉向不一定真实的解。

第五，HGNN-512 + MNN 的 SSIM 极低，说明图模型在空间结构上严重失真；而 MLP + MNN 的 SSIM 最高。

第六，当监督信号强时，简单 MLP 足够。复杂网络需要大量有效监督才能发挥优势，而 Fig.3 strict 协议下监督稀缺且噪声大，MLP 更稳健。

### 第 41 页：结论

总结我们的工作：

我们成功复现了 SpatialEx/SpatialEx+，修复了 6 个官方代码 bug；补充了 Fig.2 baseline，发现 HGNN 在纯 H&E 任务上仍优于 GT；通过 Branch Decomposition 诊断出 Fig.3 的瓶颈不在网络架构，而在缺失 panel 的监督信号质量；提出 Strict MNN 伪标签替代 Cycle，配合轻量 MLP 取得显著提升。

核心洞察是：**缺失 panel 的监督信号质量比模型复杂度更重要**。

### 第 42 页：局限与后续方向

局限包括：Slice1 方向仍较弱，这与 official panel split 导致的信息桥不对称有关；只在单一数据集验证；未与 DeepPT 在 Fig.3 上对比，CNN_Reg、Hist2ST 等待实现；GT 的 Cross-Attention Translator 受显存限制退化为 MLP；MNN pseudo-label 的噪声尚未建模。

后续方向包括：加入位置编码的多模态 matching、在更多癌种验证、与更多 baseline 公平比较、开发更轻量的 GT 变体、MNN 配对不确定性建模、以及扩展到 Omics Diagonal Integration。

### 第 43 页：附录

论文 Fig.4–6 展示了 Omics Diagonal 等更复杂任务，Extended Data 有更多方法细节。我们的 Strict MNN 思想可以推广到这些场景。

### 第 44 页：谢谢

以上就是我的汇报，请各位老师批评指正！

---

## 答辩可能被问到的问题（备用）

1. **为什么 MNN 在 Slice2 比 Slice1 提升更大？**
   这与 official panel split 导致的信息桥不对称有关。Slice2 的 B-panel 信息桥更强，PCA latent 匹配还能进一步提升；Slice1 方向本身可预测信号较弱。

2. **如果 H&E 没用，为什么原文还要用 H&E？**
   H&E 在 Fig.2 纯 H&E→omics 任务上是有用的，HGNN 也优于 GT。但在 Fig.3 的 strict panel diagonal 设定下，measured panel 已经提供了更强的分子信号，H&E 变成了弱模态噪声。

3. **MNN pseudo-label 的噪声怎么处理？**
   当前没有显式建模，后续可以考虑对 MNN 配对置信度建模，低置信度区域降低权重或不强制学习。

4. **这个方法能推广到 Omics Diagonal 吗？**
   可以。核心思想仍然是跨切片 MNN 构造伪标签，只是 matching 空间从 panel 表达变成不同组学的低维表示。

5. **为什么 Cycle + MNN 会下降？**
   MNN 提供外部锚定监督，Cycle 提供自洽约束，两者优化目标不一致。当 MNN 已经可靠时，Cycle 会把预测拉离 MNN 目标。
