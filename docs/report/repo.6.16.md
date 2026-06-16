# SpatialExP Fig.3 Panel Diagonal Integration 复现与改进报告

> 作者：李熹鸣  
> 时间跨度：6.10–6.16


本报告聚焦 SpatialEx+ 在 Fig.3 panel diagonal integration 任务上的表现。该任务要求：两个相邻切片各自只测得一部分基因 panel，需利用 H&E 图像与已测 panel 预测对方切片中缺失的 panel。前一阶段（repo.6.9）已完成了 SpatialEx/SpatialEx+ 的代码复现与 bug 修复，并在 Xenium Human Breast Cancer 的两个切片上获得了单切片 H&E-to-omics 预测 PCC 约 0.31 的基线。本阶段则进一步追问：在跨切片 panel 整合场景下，SpatialEx+ 的有效信号究竟来自何处？

通过一系列拆解与对照实验，本文得到三个相互关联的观察。其一，在当前 Rep1/Rep2 与随机 150/163 gene split 的设置下，H&E 分支的跨切片预测能力很弱，两个方向的 PCC 都接近 0。其二，panel-to-panel 分支构成了主要有效信号，Slice2 方向可达到 PCC ≈ 0.25。其三，对 cross-slice matching 做 MNN 过滤、或在 PCA latent space 中再做 MNN，能够进一步提升存在信息桥的方向（Slice2 从约 0.238 提升到 0.291），但无法突破 Slice1 方向约 0.015 的硬瓶颈。由此可见，任务困难不仅来自预测器设计，更来自 no-co-measured setting 下缺失 panel 缺乏可靠监督信号这一根本约束。


## 一、问题设定

Fig.3 panel diagonal integration 考虑的是如下场景。设 Slice1 测得基因 panel A，缺失 panel B；Slice2 测得基因 panel B，缺失 panel A。目标是利用每张切片的 H&E 图像与已测 panel，预测对方切片中的缺失 panel，即

$$
(X_1, Y_A^1) \rightarrow \hat{Y}_B^1, \qquad (X_2, Y_B^2) \rightarrow \hat{Y}_A^2.
$$

由于本项目中两个切片实际测得的基因 panel 完全相同（均为 313 个基因），我们按照论文思路将基因随机划分为两个互补集合，从而模拟 no-co-measured 场景。需要指出的是，这一设定带来一个核心困难：在训练过程中，缺失 panel 的真实值 $Y_B^1$ 与 $Y_A^2$ 不可使用，否则将造成信息泄漏。因此，模型对缺失 panel 的学习必须依赖间接监督——要么是跨切片样本之间的匹配关系，要么是 cycle consistency 这类自洽约束。监督信号的可靠性，因而成为决定任务难度的关键。


## 二、从监督信号谈起

在 no-co-measured setting 下，不妨首先考察：缺失 panel 的监督信号可以从哪里获得？一条自然路径是构造跨切片 pseudo-label，另一条路径是借助 cycle consistency。本节通过 conditional MLP 与 conditional cycle 两组对照，检验这两条路径的可行性。

### 2.1 Pseudo-label 作为监督信号

若能在 Slice1 与 Slice2 之间建立可靠的细胞对应关系，则可用 Slice2 上已测的 panel B 为 Slice1 提供 pseudo-label $\tilde{Y}_B^1$，反之亦然。为检验这种监督信号的质量，我们训练了一个以 measured panel 为输入、以 pseudo-label 为目标的 conditional MLP。表 1 给出了直接 pseudo-label 与 learned MLP 的 PCC。

| 方法                    | Slice1 PanelB PCC | Slice2 PanelA PCC |
| ----------------------- | ----------------: | ----------------: |
| direct pseudo-label     |             0.014 |             0.252 |
| learned conditional MLP |             0.016 |             0.260 |

容易看出，Slice2 方向上 learned MLP 的 PCC（0.260）已非常接近 direct pseudo-label（0.252），而 Slice1 方向上两者都接近 0。这说明在当前设置下，conditional MLP 的表达能力并非瓶颈，瓶颈在于 pseudo-label 本身的质量。需要指出的是，direct pseudo-label 的具体数值会随近邻数 $k$ 与匹配策略略有变化，后文在 MNN 实验中将看到另一组 $k$ 下的结果。进一步地，H&E 跨切片 pseudo-label 的 PCC 接近 0，这意味着形态特征难以直接建立可靠的跨切片对应。因此，measured-panel pseudo-label 可视为该方向上的有效监督上限。

### 2.2 Cycle consistency 的诊断

另一条路径是不构造 pseudo-label，而是像 SpatialEx+ 那样，通过 cycle consistency 学习 panel 之间的翻译映射。我们实现了一个 conditional cycle 模型，其损失包括 cycle reconstruction、H&E anchor 与分布匹配。训练过程中 cycle loss 能够显著下降（panel-only 从 1.19 降到 0.20），但最终预测结果仍接近随机，如表 2 所示。

| 方法                           | Slice1 PanelB PCC | Slice2 PanelA PCC |
| ------------------------------ | ----------------: | ----------------: |
| measured_pseudo MLP            |             0.016 |             0.260 |
| conditional_cycle panel-only   |            −0.004 |             0.002 |
| conditional_cycle H&E + anchor |             0.004 |             0.007 |

这一现象提示我们，cycle consistency 存在一种 *self-consistency trap*：它只保证 $Y_A \rightarrow \hat{Y}_B \rightarrow Y_A'$ 的重建误差足够小，却不保证中间的 $\hat{Y}_B$ 与真实 $Y_B$ 一致。模型可以学到自洽但无生物学意义的中间表示。此外，H&E anchor 也没有带来帮助，因为 H&E 分支本身跨切片预测能力较弱，anchor 反而可能将输出锚定到较差的预测空间。

由此可见，在缺失 panel 没有直接监督的情况下，cycle consistency 单独不足以构成可靠的学习目标；而 pseudo-label 的质量则直接决定了 conditional predictor 的上限。


## 三、分支贡献的辨析

既然 supervision 信号是关键，那么 SpatialEx+ 内部两条路径——H&E-driven branch 与 omics-cycle（panel-to-panel）branch——各自贡献了多少？本节将最终预测拆解为单 branch 输出，并考察不同融合策略的效果。

### 3.1 单 branch 性能

我们分别训练了仅使用 H&E 图像的 H&E branch，以及仅使用已测 panel 的 panel-to-panel branch。表 3 给出了两者在两个方向上的 PCC。

| 分支         | Slice1 PanelB PCC | Slice2 PanelA PCC |
| ------------ | ----------------: | ----------------: |
| H&E branch   |            −0.001 |             0.010 |
| panel branch |             0.016 |             0.248 |

不难看出，H&E branch 在两个方向上的 PCC 均接近 0，而 panel branch 在 Slice2 方向上明显更强。这提示我们，在当前 Rep1/Rep2 与随机 gene split 的设置下，SpatialEx+ 内部真正起作用的信号可能主要来自 panel-to-panel 的分子映射，而非 H&E 形态特征。

### 3.2 融合的再考察

进一步地，我们将两个 branch 做分布校准后加权融合。结果却发现，简单平均把 Slice2 从 panel branch 的 0.248 拉低到 0.177；而可靠性加权则完全退化为 H&E branch。这一现象说明，多模态融合并不天然优于单模态。若某一模态在目标域中缺乏可迁移信息，融合操作可能更多地引入噪声，从而污染强信号。换言之，模态数量本身并不构成性能保证。


## 四、跨切片匹配的局限与修正

panel-to-panel branch 的性能依赖于跨切片细胞对应关系的质量。本节考察 raw cosine kNN 匹配的问题，并尝试用 mutual nearest neighbor（MNN）过滤来提升对应关系的可靠性。

### 4.1 MNN 过滤的基本想法

raw kNN 以单向最近邻为依据构造 pseudo-label，容易引入由批次噪声或表达稀疏性导致的低质量匹配。MNN 过滤则只保留“互为最近邻”的跨切片细胞对：若 Slice1 细胞 $i$ 把 Slice2 细胞 $j$ 列入前 $k$，且 $j$ 也把 $i$ 列入前 $k$，才认为该对应关系足够可信。对没有 MNN 的细胞，则回退到普通 top-$k$ kNN。

### 4.2 主结果

表 4 对比了 raw kNN 与 MNN 过滤后的 learned MLP 性能。

| 方法        | Slice1 PanelB PCC | Slice2 PanelA PCC |
| ----------- | ----------------: | ----------------: |
| raw kNN MLP |             0.014 |             0.238 |
| MNN kNN MLP |             0.015 |             0.265 |

由此可见，MNN 过滤在 Slice2 方向上带来了稳定提升，且 learned MLP（0.265）已超过 direct pseudo-label（约 0.205）。这说明 MLP 在 measured panel 输入上做了一定程度的 denoising 与非线性修正。进一步扫描多组 $(k, \text{mnn\_}k)$ 后发现，Slice2 的 learned MLP 稳定在 0.256–0.268 之间，而 Slice1 始终停留在 0.015 左右。这一不对称性再次说明：matching 方法的改进只能改善已经存在信息桥的方向，无法凭空创造缺失的生物学对应关系。

### 4.3 指标之外的证据

平均 PCC 可能掩盖基因层面的差异。进一步统计发现，Slice2 方向上有 78.7% 的基因因 MNN 而获得提升，平均提升约 0.027；提升最显著的基因包括 CTLA4、PTPRC、ESR1、CLEC14A 等，多与免疫、上皮/基质或血管功能相关。例如，CTLA4 的 PCC 从 0.132 提升到 0.332，PTPRC 从 −0.019 提升到 0.135，ESR1 从 0.197 提升到 0.323，CLEC14A 从 0.453 提升到 0.590。对这些 marker 的空间可视化也表明，MNN 预测的高表达区域更接近真实空间分布。这些观察共同说明，Slice2 方向的提升并非平均指标的偶然波动。


## 五、共享表示空间中的再考察

MNN 的效果取决于 matching 空间的选择。一个自然的延伸问题是：若把两片切片的 measured panel 投影到一个共享的低维空间，跨切片对应关系是否会变得更加稳定？本节考察 raw measured panel、PCA latent 与 CORAL 对齐三种空间。

### 5.1 三种 matching space 的对比

我们将两片切片的 measured panel 拼接后做 PCA，投影到 50 维共享空间，再在该空间中做 MNN；或先对 measured panel 做线性协方差对齐（CORAL），再做 MNN。MLP 输入仍使用原始 measured panel，以隔离 matching 空间本身的效果。结果如表 5 所示。

| Matching space     | Slice1 PanelB PCC | Slice2 PanelA PCC |
| ------------------ | ----------------: | ----------------: |
| raw measured panel |             0.015 |             0.264 |
| PCA latent (50-d)  |             0.007 |             0.291 |
| CORAL aligned      |             0.010 |             0.228 |

容易看出，PCA latent + MNN 将 Slice2 的 learned MLP 进一步提升到约 0.291，说明把切片投影到同一低维空间后，跨切片对应关系确实更稳定。而 CORAL 线性对齐反而有损，一个可能的原因是其过度校正了切片间的协方差差异，同时也抹平了部分有用的生物学信号。与此同时，Slice1 方向三种方法都接近 0，再次印证了该方向信息桥本身较弱，不是 matching space 的小修小补可以解决的。


## 六、若干讨论

基于以上实验，本节对观察到的现象做进一步讨论。

### 6.1 对原方法的理解

当前结果表明，在本文所考察的 breast cancer replicate、Rep1/Rep2、随机 150/163 gene split 设置下，SpatialEx+ 的有效性可能并非主要来自 H&E 形态特征的跨切片泛化，而更依赖 panel-to-panel branch 所提供的分子映射信号。H&E branch 即使在训练中参与，也未能学到对缺失 panel 有意义的跨切片先验。

### 6.2 对多模态融合的启示

分支拆解与融合实验提示我们，多模态融合并不天然优于单模态。当一个模态在目标域中缺乏可迁移信息时，late fusion 不仅难以带来互补增益，反而可能污染强信号。这一观察对设计跨模态整合模型具有一般意义：在融合之前，应先验证各模态在目标域上的独立有效性。

### 6.3 对 no-co-measured setting 的启示

在缺失 panel 没有真实监督的情况下，模型结构复杂度不能替代监督信号质量。Cycle consistency 可能陷入 self-consistency trap，而 pseudo-label 的质量又直接受制于 cross-slice matching 的可靠性。因此，no-co-measured 任务的关键瓶颈往往不在预测器本身，而在如何构造或提炼高质量的间接监督信号。

### 6.4 边界条件

需要指出的是，本文结论受限于当前数据设置。两个切片来自同一肿瘤的相邻 replicate，基因 panel 完全相同；panel 划分是随机的；且 Slice1 与 Slice2 之间存在明显的方向不对称性（Slice1 A→B 远弱于 Slice2 B→A）。因此，这些结论不应被泛化为所有数据集或所有 panel split。尤其地，Slice1 方向的持续瓶颈说明，panel 划分方式与生物信号桥接强度本身会显著影响任务难度。


## 七、复现入口

为便于复现与核查，表 6 列出了三类核心入口。完整脚本、中间结果与图像文件见仓库对应目录。

| 目标     | 入口文件                                | 说明                          |
| -------- | --------------------------------------- | ----------------------------- |
| 主实验   | `scripts/fig3/run_fig3_mnn_pseudo.py`   | MNN 过滤 pseudo-label + MLP   |
| 延伸实验 | `scripts/fig3/run_fig3_latent_mnn.py`   | PCA / CORAL 共享 latent + MNN |
| 图表生成 | `scripts/fig3/generate_fig3_figures.py` | 诊断图统一生成                |

本报告使用的主要图像文件包括：

- `docs/image/fig3_diagnosis/01_branch_decomposition.png`：分支贡献对比
- `docs/image/fig3_diagnosis/02_mnn_sweep.png`：MNN 参数敏感性
- `docs/image/fig3_diagnosis/03_per_gene_scatter.png`：每基因 PCC 散点
- `docs/image/fig3_diagnosis/04_mnn_gain_distribution.png`：MNN gain 分布
- `docs/image/fig3_diagnosis/05_latent_alignment.png`：共享 latent 对齐对比
- `docs/image/fig3_diagnosis/06_mnn_pipeline.png`：MNN 流程示意
- `docs/image/fig3_diagnosis/07_cycle_trap.png`：cycle self-consistency trap 示意
- `docs/image/fig3_diagnosis/08_signal_contribution.png`：信号来源拆解
- `docs/image/fig3_diagnosis/09_top_marker_gains.png`：top marker gain
- `docs/image/fig3_diagnosis/10_pcc_distribution.png`：PCC 分布对比
- `docs/image/fig3_diagnosis/11_marker_genes_slice2.jpg`：Slice2 marker 可视化（压缩版）
- `outputs/conditional/fig3_marker_visualization/marker_genes_slice2.png`：Slice2 marker 可视化（原始高分辨率 PNG）
