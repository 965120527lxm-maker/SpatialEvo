# 答辩演讲稿（逐页）

> 对应 PPT：`docs/SpatialEx_Defense_mine.pptx`（58 页，含附录）  
> 答辩人：李熹鸣  
> 预计时长：约 45–50 分钟（正文约 40 分钟 + 附录按需）  
> 说明：带「（指图）」处为配合幻灯片图片/表格的讲解提示。

---

## 第 1 页：封面

各位老师好！我是李熹鸣。今天汇报的题目是《SpatialEx / SpatialEx+ 复现与改进——Fig.3 Panel Diagonal Integration 上的探索》。

这项工作基于 Nature Methods 2025 的论文 *High-parameter spatial multi-omics through histology-anchored integration*。我们不仅复现了原文方法，还在 Fig.3 的 panel diagonal 任务上做了两条改进路线：一是用 Graph Transformer 替代 HGNN 做网络升级；二是用 Strict MNN 伪标签替代 Cycle 做监督升级。

最终的核心发现是：**在 strict 协议下，轻量 MLP 配合 Strict MNN，反而优于官方 HGNN + Cycle**。项目代码已开源在 GitHub 仓库 SpatialEvo。下面按 PPT 顺序向各位老师汇报。

---

## 第 2 页：答辩提纲

今天汇报分七部分：

第一，背景与任务——我会结合 Fig.1 到 Fig.5，说明 Spatial Diagonal Integration 的数学设定，展示对原文任务体系的理解。

第二，环境搭建与复现——展示 Fig.2、Fig.3、Fig.4 和 Omics Diagonal 的跑通情况，以及修复的 6 个官方 bug。

第三，探索一——Graph Transformer 替代 HGNN 的网络改进及结果。

第四，诊断——Branch Decomposition，分析 H&E 与 measured panel 各自贡献。

第五，探索二——Strict MNN 伪标签替代 Cycle 的监督改进及结果。

第六，Shuffle 实验——把编码器与监督信号解耦组合，全面评估瓶颈在哪。

第七，讨论、结论、局限与展望。

---

## 第 3 页：章节分隔 — 一、背景与任务

第一部分，先介绍我们要解决什么问题，以及 Fig.3 在整篇论文中的位置。

---

## 第 4 页：背景 — 空间组学的高参数困境

空间组学有一个根本性的「不可能三角」：**分辨率、通量、panel 大小很难同时满足**。

高分辨率平台，比如 Xenium、MERFISH、CosMx，可以在单细胞级别测空间表达，但通常只能测几十到几百个基因。大 panel 平台比如 Visium，基因数更多，但分辨率低，而且仍然不是全转录组。

因此，一个很自然的实验设计是：**在相邻切片上分别测互补的信息**——切片 1 测 panel A，切片 2 测 panel B；或者切片 1 测转录组，切片 2 测蛋白组——再通过计算做整合。

这就引出了 **Spatial Diagonal Integration**：两个切片**没有共测的 omics 特征**，不能直接做常规整合，必须借助 **H&E 形态** 或 **空间位置** 做桥接。这也是我们整篇工作的出发点。

---

## 第 5 页：论文三大任务

原文实际上包含三类相关但不同的任务，我们按难度和设定区分：

**第一，Panel Diagonal Integration（Fig.3）**——同一切片技术、不同基因 panel。相邻切片各测互补基因集，目标是互相预测缺失 panel。**这是我们本次工作的核心聚焦。**

**第二，Omics Diagonal Integration（Fig.4/5）**——不同组学类型，比如 RNA 加蛋白。相邻切片测不同模态，做跨模态预测。我们已完成复现，优化还在进行中。

**第三，H&E to omics Prediction（Fig.2）**——同一切片内，输入 H&E、输出全转录组。这是 **pure morphology → expression** 的有监督任务，我们把它作为理解 Fig.3 的 **对照 baseline**。

把这三类任务分开，很重要：Fig.2 有完整监督，Fig.3 在 strict 协议下缺失 panel 不可用于训练，问题难得多。

---

## 第 6 页：论文原图 Fig.2 — H&E to omics Prediction

（指图）这是 Fig.2 的任务设定。

在同一张组织切片内，每个细胞有 H&E 嵌入 **X ∈ R^(N×d)**，以及全转录组表达 **Y ∈ R^(N×313)**。目标是学习映射 **f: X → Ŷ**，使 Ŷ 接近 Y。

与 Fig.3 的关键区别是：**X 和 Y 在同一切片上同时观测**，属于标准有监督回归，不涉及 panel split，也不存在 held-out panel。

我们补充 Fig.2 baseline 的目的，是回答一个问题：当 H&E 单独作为输入时，HGNN 到底强不强？这有助于判断 Fig.3 上性能差，究竟是网络问题还是任务设定问题。

---

## 第 7 页：论文原图 Fig.3 — Panel Diagonal Integration

（指图）Fig.3 是我们最核心的任务。

**Slice 1** 有 H&E 嵌入 X1、已测 panel A 即 YA1，**缺失 panel B** 即 YB1。  
**Slice 2** 有 H&E 嵌入 X2、已测 panel B 即 YB2，**缺失 panel A** 即 YA2。

目标是预测缺失 panel：Slice 1 预测 YB1，Slice 2 预测 YA2。

这里有一个 **Strict 限制**：训练时 **绝对不能使用 held-out panel**，也就是 YB1 和 YA2。如果直接用它们做监督，问题就退化成普通多任务回归，失去 diagonal integration 的研究意义。

模型在 strict 协议下只能依赖三类信号：**H&E 形态先验**、**跨切片 pseudo-label**、以及 **自监督约束** 如 Cycle 或 DGI。

评价指标以 **gene-level PCC** 为主，辅以 **SSIM** 和 **CMD**，分别从相关性、空间结构和分布距离衡量预测质量。

---

## 第 8 页：论文原图 Fig.4 — Omics Diagonal Integration

（指图）Fig.4 把 diagonal 从「基因 panel 级」推广到「组学模态级」。

Slice 1 已测组学 A（如 RNA），缺失组学 B（如蛋白）；Slice 2 已测组学 B，缺失组学 A。同样要求 YB1、YA2 不参与训练。

Fig.3 是 panel-level diagonal，Fig.4 是 omics-level diagonal，数据规模也更大——原文展示百万细胞级可扩展性。

我们已复现 Fig.4 训练流程；同时，**Strict MNN 的跨切片 pseudo-label 思想可以推广到 Fig.4**，这也是后续工作方向之一。

---

## 第 9 页：论文原图 Fig.5 — 更多应用场景

（指图）Fig.5 进一步一般化：多个切片、多个平台、多个组织，每个切片只观测部分 omics、缺失部分 omics，目标是从 {Xi, Yi^obs} 预测 Ŷi^mis。

核心挑战不变：**不同切片没有共享特征空间**，必须借助 H&E 或空间位置桥接。

本工作聚焦 Fig.3，但所用诊断框架和改进思路——尤其是 Strict MNN——对 Fig.4、Fig.5 这类更复杂场景同样适用。

---

## 第 10 页：章节分隔 — 二、环境搭建与复现

第二部分，介绍如何把官方代码跑通，并建立后续实验的基础。

---

## 第 11 页：SpatialEx / SpatialEx+ 方法框架

（指图）SpatialEx 的框架可以概括为 **UNI + HGNN + DGI**：

- **UNI**：病理图像预训练视觉大模型，从 H&E patch 提取细胞级嵌入；
- **HGNN**：在细胞和 pseudo-spot 上构建空间超图，做邻居聚合；
- **DGI**：自监督对比损失，增强嵌入判别性；
- 最终从 H&E 嵌入预测基因表达。

**SpatialEx+** 在 SpatialEx 基础上加入 **Omics Cycle Module**：

- 两个 panel-specific encoder：HA 预测 panel A，HB 预测 panel B；
- 中间用 regression mapping heads 实现 YA ↔ YB 的循环映射；
- Cycle 约束要求「A → B → A'」与原始 A 接近。

但这里有一个潜在问题：**Cycle 只保证自洽，不保证中间的 B 接近真实 B**。这是我们后面要用 Strict MNN 替代 Cycle 的动机之一。

---

## 第 12 页：个人环境与 Conda 配置

实验环境如下：

- **硬件**：双卡 RTX 5090；HGNN-512 显存峰值约 20–24 GB。
- **软件**：Python 3.10，PyTorch 2.7.0+cu128，scanpy / anndata / scipy 等标准单细胞栈。
- **H&E 特征**：使用预提取 UNI 嵌入，存在 `obsm['he']`；若从原始 H&E 重提，需要 timm 和 UNI 权重。
- 本机已打包 Conda 环境 **`spatialex`**，保证 Fig.2–Fig.4 和 Omics 实验可复现。

---

## 第 13 页：数据集与已复现任务

**数据集**方面，Fig.3 使用两张相邻乳腺癌 Xenium 切片：

- Rep1：约 16.4 万细胞，313 基因，含 UNI 嵌入和空间坐标；
- Rep2：约 11.1 万细胞，313 基因，同样预处理；
- 官方 panel split 为 **150 / 163** 基因，保存在 `panel_selection_official.csv`。

**已复现任务**包括：

- Fig.2：单切片 H&E → 全转录组；
- Fig.3：SpatialEx+ Panel Diagonal，Cycle baseline；
- Fig.4：SpatialEx+ 百万细胞大切片 Panel Diagonal；
- Omics Diagonal：转录组 ↔ 蛋白组。

这说明我们不仅理解了 Fig.3，也建立了对原文整体实验体系的覆盖。

---

## 第 14 页：官方复现结果总览

（指表）这一页汇总 **官方 SpatialEx / SpatialEx+** 在我们环境中的复现结果，GT 和 MNN 等改进方法在后面章节展开。

- **Fig.2**：HGNN SpatialEx-512，PCC 约 0.26 / 0.27，SSIM 约 0.42，空间结构保留较好。
- **Fig.3**：HGNN-512 + Cycle，PCC 约 0.28 / 0.30，与原文同量级。
- **Fig.4**：百万细胞任务，当前 Cycle baseline 的 PCC 很低（约 0.008 / 0.013），我们后来发现是 **数据对齐错误** 导致，正在用官方 raw Xenium 重新训练；SSIM 仍在 0.43 左右。
- **Omics Diagonal**：protein → RNA 方向 PCC 约 0.20；RNA → protein 接近随机。

这张表的作用，是给后续所有改进实验提供一个 **统一的官方 baseline 参照**。

---

## 第 15 页：Fig.2 复现 — 与原文对比

Fig.2 的设定是：**输入仅 H&E，输出 313 个基因的全转录组**，采用双向跨切片 train→test 协议。

（指表）我们的复现结果：

- **HGNN SpatialEx (512)**：PCC 0.257 / 0.273，SSIM 0.419 / 0.425，CMD 0.205 / 0.207；
- **DeepPT**：PCC 略高，0.268 / 0.276，但 SSIM 和 CMD 不如 HGNN。

与原文 Fig.2c 对比：论文报告 SpatialEx 在 PCC、SSIM、CMD 上整体优于 DeepPT。我们复现中 DeepPT 的 PCC 略高，但 **HGNN 在空间结构指标 SSIM/CMD 上更好**，总体趋势与原文一致。

这个结果很重要：**在 pure H&E 任务上，HGNN 仍然有效**。这与 Fig.3 上「换 GT 无明显增益」形成对照，暗示 Fig.3 的瓶颈可能不在网络，而在 strict 协议下的监督信号。

---

## 第 16 页：Fig.3 复现 — 与原文对比

Fig.3 使用官方 **150 / 163 gene split**：Slice 1 测 panel A，Slice 2 测 panel B，方法为 **SpatialEx+ HGNN-512 + Cycle**。

（指表）复现结果：

- **HGNN-512 + Cycle**：PCC 0.275 / 0.301，SSIM 0.308 / 0.332，CMD 0.334 / 0.267；
- **DeepPT baseline**：PCC 0.262 / 0.280，SSIM 0.334 / 0.360。

与原文对比：SpatialEx+ 在 PCC 上略高，DeepPT 在 SSIM 上略高，两者互有胜负，但都在同一数量级。这组结果就是后续 **GT 替换** 和 **Strict MNN** 的对照基准。

---

## 第 17 页：Fig.4 与 Omics Diagonal 复现

**Fig.4** 是百万细胞级 Panel Diagonal。我们完成了 SpatialExP_Big 的 200 epoch 训练，但首次评估时遇到 `coo_matrix` 不可下标的问题，已修复 eval 脚本。

（指表）当前 Cycle baseline 的 PCC 约 0.008 / 0.013，明显异常；排查后发现是 **HE 嵌入与表达矩阵未按 barcode 对齐**，而非 split 错误。我们已改用官方 `Read_Xenium` 重建 loader，并启动重训。

**Omics Diagonal** 方面：

- protein → RNA：SpatialEx+ PCC 0.196；**MLP + Strict MNN** 略升至 0.202，SSIM 从 0.367 大幅提升到 0.915；
- RNA → protein：两种方法 PCC 均接近 0，说明该方向信息桥极弱。

这说明 **MNN 思想在 Omics 场景已有初步收益**，但不同方向难度差异很大。

---

## 第 18 页：复现中修复的 6 个关键 Bug

复现过程中，我们修复了 6 个阻塞性 bug，全部在官方原始仓库中验证过：

1. `normalize_graph` 使用未定义变量 `adj`；
2. `'crs'` 拼写错误导致 sparse matrix 转换失败；
3. `Build_hypergraph_spatial_and_HE` 默认返回 coo，后续代码期望 CSR；
4. `Model_Plus.forward` 中 `agg_mtx` 维度不匹配；
5. `SpatialExP.train` 中 Regression 维度不匹配；
6. BatchNorm 在 batch=1 时崩溃。

没有这些修复，Fig.3 主实验无法完整跑通。详细验证见 `docs/report/bug_verification_original_repo.md`。

---

## 第 19 页：章节分隔 — 四、改进一：网络方面

复现跑通后，我们首先怀疑：**是不是 HGNN 不够强？** 于是尝试用 Graph Transformer 替代 HGNN。

---

## 第 20 页：改进一 — 动机与模块替换

**HGNN 的局限**在于：每个细胞与其 k 个空间近邻构成超边，邻居做 **固定权重** 的均值聚合，无法根据特征相似性自适应调整「哪些邻居更重要」。

**改进思路**是用 **Graph Transformer**：在局部邻域内做稀疏自注意力；同时用 **MFP** 替代 DGI，用 **LayerNorm** 替代 BatchNorm，避免小 batch 崩溃。

（指表）模块对照：

| 模块 | HGNN 版 | GT 版 |
|------|---------|-------|
| 空间编码器 | 固定超边聚合 | 稀疏邻居注意力 |
| 自监督 | DGI | MFP |
| 翻译器 | MLP Regression | 轻量 MLP |
| 归一化 | BatchNorm | LayerNorm |

实现位于 `SpatialEx/model_improved.py` → `SpatialExP_GT`。受显存限制，GT hidden=128，HGNN baseline=512。

---

## 第 21 页：改进一 — Graph Transformer 模块对照（简表）

（指表）这一页是上一页模块替换的可视化摘要：编码器从固定聚合变为稀疏注意力，自监督从 DGI 变为 MFP，归一化从 BatchNorm 变为 LayerNorm。

核心信息就一句：**我们不是在原框架上小修小补，而是系统性地替换 SpatialEx+ 的空间编码和自监督模块**。

---

## 第 22 页：改进一 — 方法示意图

（指图）这一页展示 SpatialEx-GT 的整体结构：H&E 嵌入经 Graph Transformer 编码，配合 MFP 自监督，再接轻量 MLP 做 panel 预测。

各位老师可以把这张图理解为：**在保持 SpatialEx+ 整体训练流程不变的前提下，只替换「空间编码 + 自监督」两个模块**，以便公平对比 HGNN 与 GT。

---

## 第 23 页：Fig.3 结果 — GT vs HGNN（Official Split + Cycle）

（指表）在 Official Split + Cycle 监督下：

- **HGNN-512 + Cycle**：PCC 0.275 / 0.301，SSIM 0.308 / 0.332；
- **GT-128 + Cycle**：PCC 0.267 / 0.276，SSIM 0.345 / 0.357。

可以看到：**PCC 上 HGNN 两方向均更高**；GT 在 SSIM 上略优（约 +0.04），但 **没有转化为 missing panel 预测精度的提升**。

这说明：**在 Fig.3 strict 协议下，换更强的图编码器并没有带来实质收益**。下一步应该问：有效信号到底来自哪里？

---

## 第 24 页：Fig.3 多方法对照

（指图）这一页把 DeepPT、HGNN、GT、MLP+MNN 放在同一张对比图里。

关键信息：**GT + Cycle 没有超过 HGNN + Cycle，更远远落后于后面的 MLP + Strict MNN**。  
因此，改进一（换 GT）不是 Fig.3 的主线答案；它更像一次必要的 negative result，把我们引向监督信号诊断。

---

## 第 25 页：Fig.3 Per-Gene — GT 未系统性超越 HGNN

（指图）Per-gene 对比更清楚：对角线下方表示 HGNN 更好。

GT 只在 **62/163**（Slice1，38%）和 **28/150**（Slice2，19%）个基因上赢 HGNN，**没有系统性优势**。

这进一步支持：**Fig.3 的瓶颈不在编码器容量或注意力机制，而在任务可用的监督信号**。

---

## 第 26 页：Fig.2 对照 — 纯 H&E 任务上 HGNN 仍优于 GT

（指表）回到 Fig.2 pure H&E 任务：

- HGNN SpatialEx (512)：PCC 0.257 / 0.273，SSIM 0.419 / 0.425；
- GT-128 + MFP：PCC 0.228 / 0.236，SSIM 0.309 / 0.320；
- GT-512 + MFP：PCC 0.244 / 0.246，SSIM 0.339 / 0.329；
- GT-128 + DGI：与 MFP 几乎无差别。

结论：**在 Fig.2 上 HGNN 仍然更强**；GT 即使增大 hidden 到 512 也追不平；MFP vs DGI 几乎无差别 → **瓶颈不在自监督形式**。

这与 Fig.3 形成鲜明对比：**同一套 GT 模块，在 H&E 有完整监督的 Fig.2 上落后 HGNN，在 Fig.3 上与 HGNN 持平——说明 Fig.3 的问题不是「图网络弱」，而是「H&E 这条输入在 strict panel diagonal 下信息不足」。**

---

## 第 27 页：章节分隔 — 五、Branch Decomposition

既然换网络没用，我们开始诊断：**SpatialEx+ 的有效信号到底来自哪里？**

---

## 第 28 页：诊断动机

SpatialEx+ 有两条可能的预测路径：

- **H&E branch**：形态 → 缺失 panel；
- **Panel branch**：已测 panel → 缺失 panel。

改进一已经说明换 GT 没有 Fig.3 增益，因此瓶颈 **可能不在图网络**。

Branch Decomposition 要回答三个问题：

1. H&E branch 单独能否预测缺失 panel？——若能，换 encoder 应该有帮助；
2. Panel branch 是否占主导？——若是，应改进 panel-to-panel 监督；
3. 两分支融合能否互补？——若能，late fusion 应该提升。

做法是把两条路径 **拆开独立训练**，量化各自贡献。

---

## 第 29 页：Branch Decomposition 实验设计（示意图 1）

（指图）这一页展示实验设计：对 Slice 1 和 Slice 2 分别构造 H&E-only 和 Panel-only 两个分支。

H&E branch 输入 UNI 嵌入，Panel branch 输入 measured panel；两者独立预测 held-out panel，再与 fusion 策略对比。

---

## 第 30 页：Branch Decomposition 实验设计（示意图 2）

（指图）这一页补充 fusion 设定：简单平均、calibrated average、reliability-weighted 等。

重点不是 fusion 本身多复杂，而是验证：**如果 H&E 真的提供互补信息，fusion 应该稳定提升；如果没有提升甚至下降，就说明 H&E 是噪声。**

---

## 第 31 页：Branch Decomposition 结果与结论

（指表）结果非常清楚：

| Variant | Slice1 PanelB PCC | Slice2 PanelA PCC |
|---------|-------------------|-------------------|
| H&E branch | -0.001 | 0.010 |
| Panel branch | 0.016 | **0.248** |
| 0.5 average (calibrated) | 0.011 | 0.177 |
| Reliability-weighted | -0.001 | 0.010 |

结论：

1. **H&E branch ≈ 0**：H&E 不能形成稳定的 missing panel 先验；
2. **Panel branch dominant**：有效信号来自 measured → missing panel 的分子映射；
3. **Late fusion 有害**：Slice2 从 0.248 降到 0.177 → H&E 是噪声而非补充。

因此，**GT 无提升的根因是 H&E 输入信息量低**；后续应该改 **监督信号**，也就是 Strict MNN。

---

## 第 32 页：诊断图 — 信号贡献分解（柱状/汇总）

（指图）这一页用柱状图汇总不同输入设置的贡献：measured panel 占绝对主导，H&E 接近零甚至为负，late fusion 无法提升。

可以把这张图当作整篇工作的「转折点」：**网络不是瓶颈，监督信号才是关键。**

---

## 第 33 页：诊断图 — H&E vs Panel 散点

（指图）左图 H&E branch 的预测-真值散点接近随机；右图 Panel branch 有明显线性趋势。

加入 H&E 后性能下降，说明 H&E 嵌入引入了与目标无关的方差，**在 strict panel diagonal 设定下，measured panel 才是主信号。**

---

## 第 34 页：诊断图 — Per-Gene 预测散点

（指图）按基因比较 Cycle-only、MNN、MNN+Cycle 等方法。

大多数基因上 **MNN 监督优于 Cycle 监督**；只有少数基因 Cycle 略好。这说明不存在「一个方法在所有基因上都最优」，但 **MNN 在整体分布上明显更可靠**。

---

## 第 35 页：章节分隔 — 六、改进二：监督信号方面

H&E 和 Cycle 都不够强，我们转向第三类信号：**跨切片 pseudo-label**。

---

## 第 36 页：Cycle Self-Consistency Trap（定义）

先解释为什么 Cycle 不够。

Cycle 约束要求：YA 预测 YB，再预测回 YA'，且 YA' 应接近原始 YA。

**问题在于：Cycle 只保证自洽，不保证中间的 YB 接近真实 YB。**

模型可以学到一种「幻觉映射」让循环闭合，Cycle loss 也会下降，但测试时预测可能与真实 panel 无关。

类比：翻译模型把英语 → 法语 → 英语，句子通顺，不等于法语翻译准确。

实验证据：**MLP + Cycle only 的 PCC 仅 0.005 / 0.013，接近随机**，说明 Cycle 本身几乎不提供有效监督。

---

## 第 37 页：Cycle Self-Consistency Trap（示意图）

（指图）这一页展示 Cycle trap：训练时模型可以自我闭环，但测试时预测可能偏离真实生物学信号。

这是 Cycle 作为 **唯一监督信号** 的根本缺陷：**缺少外部锚定。**

---

## 第 38 页：改进二动机 — Strict MNN 替代 Cycle

相邻切片组织连续，因此跨切片细胞存在形态/空间对应关系。

**Strict MNN** 的核心思想是：用 MNN 建立可信配对，把参考切片上的 measured panel **转移** 为 pseudo-label，替代 Cycle 的自环约束。

Strict Fig.3 协议下：

- Slice 1 可用 X1、YA1；YB1 仅用于 eval；
- Slice 2 可用 X2、YB2；YA2 仅用于 eval；
- **YB1、YA2 绝不参与 MNN 匹配或训练**；pseudo-label 只能来自允许输入。

这保证了我们的改进 **不违反 strict 协议**。

---

## 第 39 页：MNN 如何构建

（指公式/流程）MNN 的实现是函数 `build_mnn_pseudo(query, ref, y_ref)`，位于 `SpatialEx_conditional_gt.py`。

步骤：

1. L2 normalize → 余弦相似度；
2. Forward：query 在 ref 中取 top-`mnn_k`；
3. Reverse：保留 ref 侧 j，且 query 也在 j 的 top-`mnn_k` 内；
4. Mutual：双向互为近邻 → 过滤单向错误配对；
5. 伪标签：Mutual 邻居的 `y_ref` 均值；
6. Fallback：无 Mutual 时用 Forward 前 k 个。

默认 **k=5, mnn_k=20**。Strict MNN 的 Step 1 和 Step 2 都调用同一个函数，保证实现一致。

---

## 第 40 页：Strict MNN 分两步

**Step 1 · H&E Bridge**  
目标：构造 Slice1 缺失 panel B 的 pseudo-label。  
匹配空间：UNI H&E 嵌入 X1 ↔ X2。  
转移目标：Slice2 已测的 YB2。

**Step 2 · Panel B Bridge**  
目标：构造 Slice2 缺失 panel A 的 pseudo-label。  
匹配空间：YB2（Slice2 query）↔ ŶB1（Step1 产物，**不是真实 YB1**）。  
转移目标：Slice1 已测的 YA1。

核心：**Step 1 用形态找跨切片对应，Step 2 在 B-panel 空间 refine 对应**。两次都是同一个 `MNN()` 函数，只是 matching 空间不同。

---

## 第 41 页：Fig.3 主结果 — MLP + Strict MNN

（指表）这是改进二最重要的结果：

| 编码器 | 监督 | S1 PCC | S2 PCC | S1 SSIM | S2 SSIM |
|--------|------|--------|--------|---------|---------|
| HGNN-512 | Cycle | 0.275 | 0.301 | 0.308 | 0.332 |
| MLP | Cycle only | 0.005 | 0.013 | 0.114 | 0.107 |
| **MLP** | **Strict MNN** | **0.334** | **0.371** | **0.374** | **0.398** |
| MLP | MNN + Cycle | 0.315 | 0.353 | 0.344 | 0.388 |

**MLP + Strict MNN 是当前最优**：PCC 和 SSIM 均显著超过官方 HGNN + Cycle。

MNN + Cycle 略低于纯 MNN，说明 Cycle 与外部监督存在冲突，而不是提供额外帮助。

---

## 第 42 页：章节分隔 — 七、Shuffle

接下来做 **Shuffle 实验**：把 Strict MNN 接到不同编码器上，全面评估瓶颈到底在编码器还是监督。

---

## 第 43 页：Fig.3 的搜索空间 — 编码器 × 监督信号

我们把 SpatialEx+ 解耦成两个可独立替换的模块：

- **编码器（Encoder）**：决定如何从输入提取特征；
- **监督信号（Supervision）**：决定 missing panel 的学习目标。

（指表）实验矩阵包括：

- HGNN-512 + Cycle：官方 baseline；
- GT-128 + Cycle：改进一；
- MLP + Cycle only：去掉 MNN 的 ablation；
- MLP + Strict MNN：改进二；
- MLP + MNN + Cycle：验证 Cycle 是否干扰 MNN；
- GT-128 + Strict MNN、HGNN-512 + Strict MNN：同时换编码器和监督。

这张表的设计目的，是 **正交地分离「网络」和「监督」两个因素**。

---

## 第 44 页：Official Split Fig.3 主结果（Shuffle 全表）

（指表）完整 Shuffle 结果：

| 编码器 | 监督 | S1 PCC | S2 PCC | S1 SSIM | S2 SSIM |
|--------|------|--------|--------|---------|---------|
| HGNN-512 | Cycle | 0.275 | 0.301 | 0.308 | 0.332 |
| GT-128 | Cycle | 0.267 | 0.276 | 0.345 | 0.357 |
| MLP | Cycle only | 0.005 | 0.013 | 0.114 | 0.107 |
| **MLP** | **Strict MNN** | **0.334** | **0.371** | **0.374** | **0.398** |
| MLP | MNN + Cycle | 0.315 | 0.353 | 0.344 | 0.388 |
| GT-128 | Strict MNN | 0.258 | 0.289 | 0.359 | 0.387 |
| HGNN-512 | Strict MNN | 0.234 | 0.273 | **0.072** | **0.055** |

三个关键结论：

1. **MLP + Strict MNN 全面最优**；
2. 换 GT 或 HGNN 加 MNN，PCC 都不如 MLP + MNN；
3. **HGNN-512 + MNN 的 SSIM 极低（0.07 / 0.06）**，空间结构严重失真，而 MLP + MNN 的 SSIM 最高。

这说明：**监督信号质量比模型复杂度更关键**；复杂图模型在弱监督下还可能伤害空间保真。

---

## 第 45 页：Top Marker Genes 提升 — HGNN+Cycle vs MLP+MNN

（指表）Slice 2 上提升最显著的 marker 基因：

| Gene | HGNN+Cycle | MLP+MNN | 提升 |
|------|------------|---------|------|
| CPA3 | 0.079 | 0.578 | +0.498 |
| CTSG | 0.072 | 0.555 | +0.483 |
| CLEC14A | 0.330 | 0.638 | +0.308 |
| EGFL7 | 0.141 | 0.358 | +0.217 |
| CLDN5 | 0.046 | 0.262 | +0.216 |

这些基因涉及 **免疫、上皮/基质、血管** 等生物学过程：CPA3/CTSG 与肥大细胞相关，CLEC14A/EGFL7/CLDN5 与血管内皮相关。

说明 MNN 的改进 **不只在平均 PCC 上，也在关键 marker 基因和空间表达结构上有可辨别的提升**。

---

## 第 46 页：章节分隔 — 八、讨论

下面回答一个核心问题：**为什么简单 MLP 能打败复杂网络？**

---

## 第 47 页：为什么 MLP + MNN 反而最强？

我们有六点解释：

1. **任务本质是 panel-to-panel 分子映射**，不是空间聚合；同切片上 measured panel 与 missing panel 有生物学相关性。
2. **MNN 已经编码了空间/形态信息**——H&E MNN 找形态相似细胞，Panel B MNN 进一步找表达相似细胞。
3. **HGNN/GT 的额外空间聚合会 over-smooth 或引入 H&E 噪声**——Branch Decomposition 已证明 H&E branch ≈ 0。
4. **Cycle 与 MNN 冲突**——MNN 是外部锚定监督，Cycle 是自洽约束，会把预测拉离 MNN 目标。
5. **HGNN + MNN 的 SSIM 极低**，空间结构失真；MLP + MNN 的 SSIM 最高，空间保真最好。
6. **当监督信号强时，简单 MLP 足够**——复杂网络需要大量有效监督才能发挥优势；strict 协议下 MLP 更稳健。

---

## 第 48 页：为什么 MLP 能超越 MNN pseudo-label 本身？

（指表）这是一个很好的追问：既然监督来自 MNN pseudo-label，为什么训练后的 MLP 比 pseudo-label 本身还强？

| 构造方式 | pseudo-label 质量 | 训练后 MLP | 提升 |
|----------|-------------------|------------|------|
| raw cosine kNN | 0.178 / 0.252 | 0.327 / 0.381 | +0.149 / +0.129 |
| Strict MNN | 0.153 / 0.187 | 0.334 / 0.371 | +0.181 / +0.184 |

解释：

- **pseudo-label 是带噪的 per-cell 估计**：每个细胞来自跨切片 kNN/MNN 的少数邻居平均；Slice2 还要经 pseudo_B1 做两步桥接，误差会累积。
- **MLP 学到的是全局 panel-to-panel 映射**：输入是同切片的 measured panel（强信号），目标是有噪 pseudo-label；模型会把噪声平均掉，直接学习 YB2 → YA2，绕过两步桥接。

结论：**MLP 把有噪但方向正确的 pseudo-label「蒸馏」成稳定的 translator。**

---

## 第 49 页：结论

总结四点：

1. **复现**：跑通 SpatialEx/SpatialEx+，修复 6 个官方 bug；补充 Fig.2 baseline，HGNN 在 pure H&E 任务上仍优于 GT。
2. **诊断**：Branch Decomposition 揭示 H&E branch 弱、panel branch dominant。
3. **改进**：Strict MNN 伪标签替代 Cycle，提供可靠的外部监督。
4. **发现**：**MLP + Strict MNN 效果最佳**（PCC 0.334 / 0.371，SSIM 0.374 / 0.398）。

核心洞察：**缺失 panel 的监督信号质量比模型复杂度更重要**——在 strict 协议下，**简单模型 + 好信号 > 复杂模型 + 弱信号**。

---

## 第 50 页：局限与后续方向

**局限**：

- Slice1 方向仍较弱，与 official split 信息桥不对称有关；
- 仅在单一乳腺癌数据集验证；
- 未与 DeepPT 在 Fig.3 上做系统对比；
- CNN_Reg、Hist2ST 等待实现；
- GT 受显存限制，Cross-Attention Translator 退化为 MLP；
- MNN pseudo-label 有噪声，尚未显式建模不确定性。

**后续方向**：

- 加入位置编码的多模态 matching；
- 在更多癌种和组织上验证；
- 与 DeepPT、CNN_Reg、Hist2ST 公平比较；
- 开发更轻量的 GT 变体；
- MNN 配对不确定性建模；
- 扩展到 Omics Diagonal Integration 和 Fig.4 百万细胞场景。

---

## 第 51 页：谢谢

以上就是我的汇报。感谢各位老师的聆听，请批评指正！

论文：*High-parameter spatial multi-omics through histology-anchored integration* (Nature Methods 2025)  
项目仓库：https://github.com/965120527lxm-maker/SpatialEvo

---

## 附录部分（第 52–58 页，答辩追问或时间充裕时使用）

### 第 52 页：Appendix 封面

下面是附录，主要回答参数敏感性、latent alignment 和更多可视化结果。如果老师感兴趣，我可以展开；否则可在问答环节选择性展示。

---

### 第 53 页：MNN 参数敏感性扫描

（指图）横轴是邻居数 k 或 mnn_k，纵轴是 gene-level PCC。

MNN 在较宽的参数范围内表现稳定，**不是依赖某个特定超参数「凑」出来的结果**。默认 k=5、mnn_k=20 位于平坦区域，鲁棒性较好。

---

### 第 54 页：跨切片 Latent Alignment 与 Latent MNN

（指表）我们比较了不同 matching 空间：

| Matching space | Slice1 learned PCC | Slice2 learned PCC |
|----------------|--------------------|--------------------|
| raw measured panel | 0.015 | 0.264 |
| PCA latent (50-d) | 0.007 | **0.291** |
| CORAL aligned | 0.010 | 0.228 |

PCA latent 在 Slice2 上进一步提升到 0.291，说明适度降噪有助于跨切片匹配；CORAL 线性对齐反而下降，可能 **过度抹平生物学相关的切片间差异**。

Slice1 始终接近 0，说明 **该方向的信息桥本身较弱**，与 official panel split 的不对称性一致。

---

### 第 55 页：Top Marker Genes 提升（附录版）

（指表）另一组 marker 基因提升：CTLA4 +0.200，PTPRC +0.154，ESR1 +0.126，CLEC14A +0.137，涉及免疫、激素、血管等通路。

与正文第 45 页互补，说明提升分布在多个生物学相关基因上，而非个别 outlier。

---

### 第 56 页：MNN & KNN — Slice2 Marker Gene 空间可视化

（指图）对比 Ground Truth、MLP+Strict MNN、HGNN/GT+Cycle 的空间表达图。

MNN 预测的热点区域与真实值 **对齐更好**；Cycle 预测常见 **过平滑或空间错位**。这是 SSIM 指标提升的视觉证据。

---

### 第 57 页：MNN & KNN — Per-Gene PCC 分布

（指图）小提琴图/箱线图比较各方法在所有基因上的 PCC 分布。

MLP + Strict MNN 的分布 **整体右移**；Cycle 方法集中在 0 附近甚至负值。说明改进是 **系统性的**，不是少数基因拉高平均。

---

### 第 58 页：MNN & KNN — Per-Gene 提升分布

（指图）横轴 ΔPCC（MNN − Cycle），纵轴基因密度。

**78.7% 的基因获得正提升**；少数基因下降，说明仍有改进空间，也提示未来可以做 gene-specific 或 uncertainty-weighted 训练。

---

## 附录：答辩常见问题（备用）

**Q1：为什么 MNN 在 Slice2 比 Slice1 提升更大？**  
A：与 official panel split 导致的信息桥不对称有关。Slice2 的 B-panel 信息桥更强；Slice1 方向可预测信号本身较弱。

**Q2：如果 H&E 没用，为什么原文还要用 H&E？**  
A：H&E 在 Fig.2 pure H&E→omics 任务上是有用的，HGNN 也优于 GT。但在 Fig.3 strict panel diagonal 下，measured panel 已是更强信号，H&E 变成弱模态噪声。

**Q3：MNN pseudo-label 的噪声怎么处理？**  
A：当前未显式建模；MLP 训练本身有一定降噪效果。后续可对 MNN 配对置信度加权。

**Q4：这个方法能推广到 Omics Diagonal 吗？**  
A：可以。核心仍是跨切片 MNN 构造 pseudo-label，matching 空间从 panel 表达换成不同组学的低维表示。Omics 实验中 protein→RNA 已有初步提升。

**Q5：为什么 Cycle + MNN 会下降？**  
A：MNN 提供外部锚定监督，Cycle 提供自洽约束，优化目标不一致；当 MNN 已较可靠时，Cycle 会把预测拉离 MNN 目标。

**Q6：Fig.4 的 PCC 为什么这么低？**  
A：首次复现使用了错误的数据对齐（HE npy 与 novae h5ad 按行截断拼接，未做 barcode join）。已改用官方 Read_Xenium 修复 loader 并重训；SSIM 正常说明模型结构可跑通，问题在数据而非方法本身。
