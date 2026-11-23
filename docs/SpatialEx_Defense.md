# SpatialEx / SpatialEx+ 答辩讲稿（逐页）

> 对应 PPT：`docs/SpatialEx_Defense_mine.pptx`（58 页）  
> 答辩人：李熹鸣  
> 正文时长：约 40 分钟（第 1–51 页）  
> 附录：答辩追问备用（第 52–58 页，不计入主时长）  
> 叙事主线：strict panel diagonal 中，缺失 panel 的监督从哪里来？

---

## 第 1 页：封面

> 建议时长：55 秒

各位老师好, 我是李熹鸣, 在过去的两周半时间内, 我对SpatialEx/P 项目进行了复现. 论文主要包括三大任务, 我针对其中的Panel Diagonal Integration进行了优化. 先说结论, 我提出的MLP+MNN方法 在Panel diagonal Integration 上表现稳定优于官方方法.



---

## 第 2 页：答辩提纲

> 建议时长：45 秒
> 
那么在接下来的报告中, 请允许我先行回顾论文的背景与任务, 复现的结果, 再依次介绍我改进思路的演变.

---

## 第 3 页：章节分隔 — 一、背景与任务

> 建议时长：15 秒

首先要介绍我们解决什么问题

---

## 第 4 页：背景 — 空间组学的高参数困境

> 建议时长：50 秒

空间组学有一个结构性矛盾：**分辨率、通量、panel 大小，三者很难同时满足。**

Xenium、MERFISH、CosMx 分辨率高，但通常只能测几十到几百个基因。Visium panel 更大，分辨率却低，而且仍然不是全转录组。

怎么办？一个现实做法是：**相邻切片测互补信息**——一切片测 panel A，一切片测 panel B；或者一切片测 RNA，一切片测蛋白——再靠计算整合。

这就引出 **Spatial Diagonal Integration**：两个切片**没有共测的 omics 特征**，不能走常规整合的路，必须借助 **H&E 形态** 或 **空间位置** 做桥接。

那么这座**桥是否可靠**呢?这就是我们的模型要完善的了。

---

## 第 5 页：论文三大任务

> 建议时长：55 秒

原文有三类任务。

**第一，Panel Diagonal Integration（Fig.3）**——同一切片技术、不同基因 panel。相邻切片各测互补基因集，互相预测缺失 panel。**这是我们本次工作的核心。**

**第二，Omics Diagonal Integration（Fig.4/5）**——不同组学，如 RNA 与蛋白。我们已完成复现，系统优化仍在进行。

**第三，H&E to omics Prediction（Fig.2）**——同一切片内，输入 H&E、输出全转录组。这是有完整监督的对照任务。

---

## 第 6 页：论文原图 Fig.2 — H&E to omics Prediction

> 建议时长：45 秒

请看 Fig.2。同一张切片内，每个细胞有 H&E 嵌入 **X ∈ R^(N×d)**，有全转录组 **Y ∈ R^(N×313)**。目标是学映射 **f: X → Ŷ**，使 Ŷ 接近 Y。

---

## 第 7 页：论文原图 Fig.3 — Panel Diagonal Integration

> 建议时长：65 秒

Fig.3 是整场汇报的核心。

**Slice 1**：H&E 嵌入 X1，已测 panel A（YA1），**缺失 panel B（YB1）**。  
**Slice 2**：H&E 嵌入 X2，已测 panel B（YB2），**缺失 panel A（YA2）**。

目标是预测 YB1 和 YA2。

模型只能依赖三类信号：H&E 形态先验、跨切片 pseudo-label、以及 Cycle 或 DGI 等自监督。

评价以 **gene-level PCC** 为主，辅以 **SSIM** 和 **CMD**——相关性、空间结构、分布距离，三个维度一起看。

---

## 第 8 页：论文原图 Fig.4 — Omics Diagonal Integration

> 建议时长：40 秒
Fig.3 是 panel-level diagonal；Fig.4 是 omics-level diagonal，规模也更大——原文强调百万细胞可扩展性。

---

## 第 9 页：论文原图 Fig.5 — Omics Diagonal Integration

> 建议时长：35 秒

Fig.5 进一步一般化：多切片、多平台，每个切片只观测部分 omics、缺失部分 omics。

核心挑战不变：**没有共享特征空间，必须借 H&E 或空间位置桥接。**

本工作聚焦 Fig.3，但诊断框架——先找监督、再谈模型——对 Fig.4、Fig.5 同样适用。当然，能否复制 Fig.3 的结论，还要在各自数据和协议下单独检验。

---

## 第 10 页：章节分隔 — 二、环境搭建与复现

> 建议时长：15 秒

问题清楚了，下一步是：原方法能不能跑通？

---

## 第 11 页：SpatialEx / SpatialEx+ 方法框架

> 建议时长：55 秒

原方法框架可以概括为两部分。

**SpatialEx = UNI + HGNN + DGI。** UNI 从 H&E 提嵌入；HGNN 在空间超图上做邻居聚合；DGI 做自监督对比；最终从 H&E 重建基因表达。

**SpatialEx+ 在此基础上加 Omics Cycle Module。** 两个 panel-specific encoder 分别预测 panel A 和 panel B；中间用 mapping heads 实现 YA ↔ YB 循环映射。


---

## 第 12 页：个人环境与 Conda 配置

> 建议时长：35 秒

个人实验环境如上所示, 机已打包 Conda 环境 **spatialex**.
---

## 第 13 页：数据集与已复现任务

> 建议时长：45 秒

Fig.3 使用两张相邻乳腺癌 Xenium 切片：**Rep1** 约 16.4 万细胞、313 基因；**Rep2** 约 11.1 万细胞、313 基因；均含 UNI 嵌入与空间坐标。官方 panel split 为 **150 / 163**，保存在 。

已复现任务包括：Fig.2 单切片 H&E→全转录组；Fig.3 SpatialEx+ Panel Diagonal（Cycle baseline）；Fig.4 SpatialExP_Big 百万细胞 Panel Diagonal；Omics Diagonal 转录组↔蛋白组。


---

## 第 14 页：官方复现结果总览

> 建议时长：50 秒

请看这张总表——只列 **官方 SpatialEx / SpatialEx+** 复现结果；GT、MNN 等改进在后面展开。

**Fig.2**：HGNN SpatialEx-512，PCC 约 0.26/0.27，SSIM 约 0.42——H&E 单独作输入，HGNN 有效。  
**Fig.3**：HGNN-512 + Cycle，PCC 0.275/0.301——这是后面所有对照的基准。  
**Fig.4**：Cycle baseline PCC 约 0.008/0.013——**此结果受数据对齐错误影响，重训中，不宜作为方法结论。** SSIM 约 0.43，说明 pipeline 能跑通。  
**Omics**：protein→RNA PCC 约 0.20；RNA→protein 接近随机——方向不对称，只能作初步观察。

这张表的作用：建立 **官方 baseline**，而不是宣布胜利。

---

## 第 15 页：Fig.2 复现 — 与原文对比

> 建议时长：50 秒

Fig.2 输入仅 H&E，输出 313 基因，双向跨切片 train→test。

**HGNN SpatialEx (512)**：PCC 0.257/0.273，SSIM 0.419/0.425，CMD 0.205/0.207。  
**DeepPT**：PCC 略高，0.268/0.276，但 SSIM、CMD 不如 HGNN。

与原文 Fig.2c 对比：论文报告 SpatialEx 整体优于 DeepPT。我们复现中 DeepPT PCC 略高，**HGNN 在空间结构指标上更好**，总体趋势与原文一致。

这个结果很重要：**在 pure H&E 任务上，HGNN 仍然有效。** 它否定了一种草率解释——「原文方法的网络整体不行」——同时把 Fig.3 的问题单独拎出来：同一条 HGNN，为什么在 Fig.3 上不够？

---

## 第 16 页：Fig.3 复现 — 与原文对比

> 建议时长：45 秒

Fig.3 使用官方 **150/163 gene split**，SpatialEx+ **HGNN-512 + Cycle**。

**HGNN-512 + Cycle**：PCC 0.275/0.301，SSIM 0.308/0.332，CMD 0.334/0.267。  
**DeepPT baseline**：PCC 0.262/0.280，SSIM 0.334/0.360。

与原文同量级，互有胜负：SpatialEx+ PCC 略高，DeepPT SSIM 略高。这组数字就是后面 GT 替换和 Strict MNN 的 **对照基准**。

问题不在「能不能复现」，而在：**还能不能更好？瓶颈在哪里？**

---

## 第 17 页：Fig.4 与 Omics Diagonal 复现

> 建议时长：50 秒

**Fig.4** 百万细胞 Panel Diagonal。我们完成 SpatialExP_Big 200 epoch 训练，修复了 eval 中 `coo_matrix` 不可下标的 bug。

当前 Cycle baseline：PCC 0.008/0.013，SSIM 0.439/0.433。PCC 异常低——排查后发现是 **Big*_uni.npy 与 novae h5ad 按行截断拼接，未做 barcode 对齐**。split 本身正确；问题在数据 loader。已改用官方 `Read_Xenium`，**重训进行中，最终指标待更新**。

**Omics Diagonal**（初步结果，不作主结论）：protein→RNA，SpatialEx+ PCC 0.196，MLP + Strict MNN 0.202，SSIM 从 0.367 升至 0.915；RNA→protein 两种方法均接近随机。说明 MNN 思想有探索价值，但 **方向难度差异极大，不能写成全面胜利**。

---

## 第 18 页：复现中修复的 6 个关键 Bug

> 建议时长：40 秒

复现过程中修复 6 个阻塞性 bug，均在官方原始仓库验证：`normalize_graph` 未定义变量；`'crs'` 拼写错误；超图返回 coo 而非 CSR；`agg_mtx` 维度不匹配；Regression 维度不匹配；BatchNorm batch=1 崩溃。

没有这些修复，Fig.3 主实验跑不通。它们是前提，不是贡献本身。详情见 `docs/report/bug_verification_original_repo.md`。

修复完成，我们就可以问：**在原方法能跑通的前提下，瓶颈究竟在哪里？**

---

## 第 19 页：章节分隔 — 四、改进一：网络方面

> 建议时长：15 秒

第一个自然怀疑：**是不是 HGNN 表达能力不够，才限制了缺失 panel 的预测？** 我们先从网络改起。

---

## 第 20 页：改进一 — 动机与模块替换

> 建议时长：55 秒

HGNN 的局限在于：k 近邻构成超边，邻居做 **固定权重** 均值聚合——不能根据特征相似性调整「谁更重要」。

改进思路：**Graph Transformer** 在局部邻域做稀疏自注意力；**MFP** 替代 DGI；**LayerNorm** 替代 BatchNorm，避免小 batch 崩溃。

编码器从固定聚合变为自适应注意力；自监督从 DGI 变为 MFP；翻译器保持轻量 MLP。实现于 `SpatialEx/model_improved.py` → `SpatialExP_GT`；受显存限制，GT hidden=128，HGNN baseline=512。

动机合理。但合理不等于有效——**事情是不是这样，还要由实验来回答。**

---

## 第 21 页：改进一 — Graph Transformer 模块对照（简表）

> 建议时长：30 秒

这张表是模块替换的摘要：编码器、自监督、归一化，三处系统性替换，而非小修小补。

要点只有一句：**我们在保持 SpatialEx+ 整体流程不变的前提下，只换「空间编码 + 自监督」，以便公平判断——瓶颈是否在编码器。**

---

## 第 22 页：改进一 — 方法示意图

> 建议时长：35 秒

请看结构图。H&E 嵌入经 Graph Transformer 编码，配合 MFP 自监督，再接轻量 MLP 做 panel 预测。

这张图说明我们改的是什么、没改的是什么。没改的是 strict 协议、panel split、Cycle 监督框架；改的是 **HGNN → GT** 这一条路。

如果瓶颈在编码器，GT 应该带来可见提升。我们看结果。

---

## 第 23 页：Fig.3 结果 — GT vs HGNN（Official Split + Cycle）

> 建议时长：50 秒

Official Split + Cycle 下：

**HGNN-512 + Cycle**：PCC 0.275/0.301，SSIM 0.308/0.332。  
**GT-128 + Cycle**：PCC 0.267/0.276，SSIM 0.345/0.357。

PCC 上 HGNN 两方向均更高；GT 在 SSIM 上略优约 0.04，**但没有转化为 missing panel 预测精度的提升。**

这个结果很重要。它 **否定** 了「主要矛盾在编码器」这一解释，同时 **把我们引向** 另一种可能：GT 主要在加工 H&E，而 H&E 在 Fig.3 strict 设定下，也许并不是主信号。

**网络改复杂了，问题没有解决。**

---

## 第 24 页：Fig.3 多方法对照

> 建议时长：40 秒

这张图把 DeepPT、HGNN、GT、MLP+MNN 放在一起。

最值得看的现象：**GT + Cycle 没有超过 HGNN + Cycle，更远远落后于 MLP + Strict MNN。**

它否定的是「换更强的图编码器就能解决 Fig.3」；它支持的是「还要往下追——有效信号到底来自哪里」。

改进一不是失败的白做，而是一次 **必要的 negative result**。

---

## 第 25 页：Fig.3 Per-Gene — GT 未系统性超越 HGNN

> 建议时长：40 秒

Per-gene 对比更细：对角线下方表示 HGNN 更好。

GT 只在 **62/163**（Slice1，38%）和 **28/150**（Slice2，19%）个基因上赢 HGNN——**没有系统性优势。**

个别基因谁高谁低，不能推翻整体判断。整体事实是：**换 GT 不能在基因层面稳定胜出。**

这就迫使我们继续追问：GT 加工的 H&E 信号，在这条任务上到底有多强？

---

## 第 26 页：Fig.2 对照 — 纯 H&E 任务上 HGNN 仍优于 GT

> 建议时长：55 秒

把 GT 放回 Fig.2 pure H&E 任务——同一切片有完整 Y 监督：

**HGNN SpatialEx (512)**：PCC 0.257/0.273，SSIM 0.419/0.425。  
**GT-128 + MFP**：PCC 0.228/0.236，SSIM 0.309/0.320。  
**GT-512 + MFP**：PCC 0.244/0.246——容量加大仍追不平 HGNN。  
**MFP vs DGI**：几乎无差别。

对比 Fig.3 和 Fig.2，矛盾很清楚：

- Fig.2 上，HGNN 仍然更强——**不是图网络整体不行**；  
- Fig.3 上，换 GT 无增益——**不是注意力没实现好**。

进一步分析，问题指向：**Fig.3 strict panel diagonal 下，H&E 这条输入的信息量可能不足。** GT 无提升，不是因为 GT 弱，而是因为它主要在处理一条弱信号。

到这里，继续堆编码器复杂度，意义有限。 **这就迫使我们改变思路——拆解信号来源。**

---

## 第 27 页：章节分隔 — 五、Branch Decomposition

> 建议时长：15 秒

既然改网络没有解决 Fig.3，我们就把输入拆开：**H&E 和 measured panel，各自到底贡献多少？** 这是整场汇报的关键转折。

---

## 第 28 页：诊断动机

> 建议时长：50 秒

SpatialEx+ 隐含两条预测路径：

**第一，H&E branch**：形态 → 缺失 panel。  
**第二，Panel branch**：已测 panel → 缺失 panel。

改进一已经说明：换 GT 没有 Fig.3 增益。瓶颈 **可能不在图网络**。

Branch Decomposition 要回答三个问题：

1. H&E 单独能否预测缺失 panel？——能，则换 encoder 应有帮助。  
2. Panel 是否占主导？——是，则应改 panel-to-panel 监督。  
3. 融合能否互补？——能，则 late fusion 应提升。

我们先不急着提出新方法，**先看问题究竟出在哪里。**

---

## 第 29 页：Branch Decomposition 实验设计（示意图 1）

> 建议时长：35 秒

请看设计图。Slice 1 和 Slice 2 分别构造 H&E-only 与 Panel-only 两个分支，独立预测 held-out panel。

这张图要表达的不是流程复杂，而是一个原则：**把混在一起的信号拆开，各自用 PCC 说话。**

---

## 第 30 页：Branch Decomposition 实验设计（示意图 2）

> 建议时长：35 秒

此外还测试简单平均、calibrated average、reliability-weighted 等融合策略。

关键不在 fusion 算法多精巧，而在一个反事实：**如果 H&E 真的提供互补信息，融合应该稳定提升；如果不升反降，H&E 就是噪声。**

---

## 第 31 页：Branch Decomposition 结果与结论

> 建议时长：60 秒

事实说明什么？

| Variant                  | Slice1 PanelB PCC | Slice2 PanelA PCC |
| ------------------------ | ----------------- | ----------------- |
| H&E branch               | -0.001            | 0.010             |
| Panel branch             | 0.016             | **0.248**         |
| 0.5 average (calibrated) | 0.011             | 0.177             |
| Reliability-weighted     | -0.001            | 0.010             |

**第一**，H&E branch ≈ 0——H&E 构不成稳定的 missing panel 先验。  
**第二**，Panel branch 占主导——有效信号来自 measured → missing panel 的分子映射。  
**第三**，Late fusion 有害——Slice2 从 0.248 降到 0.177；**H&E 不是补充，是噪声。**

到这里，问题已经比较清楚了：

> 在 Fig.3 的 strict panel diagonal 任务中，H&E 不是主要信息来源，measured panel 才是。

GT 无提升，不是因为注意力机制实现得不够好，而是因为它所加工的 H&E 信号本身就很弱。

**这就迫使我们改变思路——从改网络，转向改监督。**

---

## 第 32 页：诊断图 — 信号贡献分解

> 建议时长：40 秒

请看柱状汇总。measured panel 贡献占绝对主导；H&E 接近零甚至为负；late fusion 无法提升。

这张图是整场汇报的 **分水岭**：**不是网络太弱，而是 H&E 信号太弱；不是模型越复杂越好，而是信号是否可靠更重要。**

---

## 第 33 页：诊断图 — H&E vs Panel 散点

> 建议时长：40 秒

左图 H&E branch：预测-真值散点接近随机。右图 Panel branch：有明显线性趋势。

加入 H&E 后性能下降——H&E 嵌入引入了与目标无关的方差。

它否定「H&E 形态能稳定预测缺失 panel」；它支持「panel-to-panel 映射才是主战场」。

---

## 第 34 页：诊断图 — Per-Gene 预测散点

> 建议时长：35 秒

按基因看，大多数基因上 **MNN 监督优于 Cycle 监督**；只有少数基因 Cycle 略好。

这说明不存在单一方法在所有基因上都最优。但整体分布已经指向：**外部 pseudo-label 比 Cycle 自环更可靠。**

这为下一步 Strict MNN 做了铺垫。

---

## 第 35 页：章节分隔 — 六、改进二：监督信号方面

> 建议时长：15 秒

H&E 弱，measured panel 强——那原方法的 Cycle 监督可靠吗？**问题不在于模型能不能把循环做完，而在于循环中间的预测是否真实。**

---

## 第 36 页：Cycle Self-Consistency Trap（定义）

> 建议时长：55 秒

Cycle 约束：YA 预测 YB，再预测回 YA'，要求 YA' 接近 YA。

但 **Cycle 只保证自洽，不保证中间的 YB 接近真实 YB。** 模型可以学到一种「幻觉映射」让循环闭合，loss 也会下降，测试时预测却可能与真实 panel 无关。

类比：一句话中文→英文→中文，最后能还原原句，**不等于中间那句英文翻译正确。**

实验证据：**MLP + Cycle only 的 PCC 仅 0.005 / 0.013，接近随机。**

第三个判断由此成立：

> 缺失 panel 的学习，需要外部锚点，不能只靠模型自己证明自己。

---

## 第 37 页：Cycle Self-Consistency Trap（示意图）

> 建议时长：35 秒

请看示意图。训练时模型可以自我闭环；测试时预测可能偏离真实生物学信号。

Cycle 作为 **唯一监督** 的根本缺陷：**缺少外部锚定。** 这不是实现细节问题，是目标函数问题。

---

## 第 38 页：改进二动机 — Strict MNN 替代 Cycle

> 建议时长：50 秒

相邻切片组织连续，跨切片细胞存在形态和表达对应。Strict MNN 的核心：**用 MNN 建立可信配对，从参考切片转移已观测 panel，构造 pseudo-label。**

Strict 协议不能破：

- Slice 1 可用 X1、YA1；YB1 仅 eval。  
- Slice 2 可用 X2、YB2；YA2 仅 eval。  
- **YB1、YA2 绝不参与 MNN 匹配或训练。**

这不是让模型自圆其说，而是给 missing panel 一个 **来自切片外部的监督锚点。**

---

## 第 39 页：MNN 如何构建

> 建议时长：45 秒

MNN 的逻辑不复杂，核心是 **互为近邻** 才配对：

L2 归一化 → 余弦相似度 → Forward 取 top-`mnn_k` → Reverse 互检 → Mutual 保留双向近邻 → 对 ref 侧 `y_ref` 取均值得伪标签。无 Mutual 时回退 Forward 前 k 个。默认 **k=5, mnn_k=20**。

实现于 `SpatialEx_conditional_gt.py` 的 `build_mnn_pseudo`。Strict MNN 的 Step 1 和 Step 2 调用同一函数——区别只在 matching 空间，不在算法本身。

细节服务于原则：**只用互为近邻的高置信配对，减少跨切片错配。**

---

## 第 40 页：Strict MNN 分两步

> 建议时长：50 秒

**Step 1 · H&E Bridge**  
为 Slice1 构造 pseudo-B。匹配空间：UNI H&E 嵌入 X1 ↔ X2。转移目标：Slice2 已测的 YB2。

**Step 2 · Panel B Bridge**  
为 Slice2 构造 pseudo-A。匹配空间：YB2 ↔ ŶB1（Step1 产物，**不是真实 YB1**）。转移目标：Slice1 已测的 YA1。

两步的关系：Step 1 用形态找跨切片对应；Step 2 在 B-panel 空间 refine 对应。Step 2 的误差会累积——后面会讲到，MLP 训练对此有一定降噪作用。

---

## 第 41 页：Fig.3 主结果 — MLP + Strict MNN

> 建议时长：55 秒

实践检验的结果：

**HGNN-512 + Cycle**：PCC 0.275/0.301。  
**MLP + Cycle only**：0.005/0.013——接近随机，Cycle 单独几乎无效。  
**MLP + Strict MNN**：PCC **0.334/0.371**，SSIM **0.374/0.398**——当前最优。  
**MLP + MNN + Cycle**：0.315/0.353——**略低于纯 MNN**。

它否定「Cycle 能给 MNN 再加一层帮助」；它支持「外部锚定监督是主要增益来源」。

在当前数据集、official split 和 strict 协议下，**好的信号配上简单 MLP，明显胜过 HGNN + Cycle。**

---

## 第 42 页：章节分隔 — 七、Shuffle

> 建议时长：15 秒

MNN 有效，但还有一个问题必须回答：**是 MNN 好，还是 MLP 好？是监督好，还是编码器好？** Shuffle 实验就是来完成这条因果链的。

---

## 第 43 页：Fig.3 的搜索空间 — 编码器 × 监督信号

> 建议时长：45 秒

我们把 SpatialEx+ 解耦成两个可独立替换的模块：

**编码器**：如何从输入提取特征。  
**监督信号**：missing panel 的学习目标从哪里来。

实验矩阵覆盖：HGNN/GT/MLP × Cycle/MNN/MNN+Cycle。

如果瓶颈在编码器，**HGNN + MNN 或 GT + MNN 应该最强。** 如果瓶颈在监督，**MLP + MNN 应该最强，且 MNN 接到复杂编码器上不应更好。**

这是一个正交设计，用来定位 **主要矛盾**。

---

## 第 44 页：Official Split Fig.3 主结果（Shuffle 全表）

> 建议时长：60 秒

请看 Shuffle 全表——不逐格念，抓四个关键事实：

**第一**，**MLP + Strict MNN** 全面最优：PCC 0.334/0.371，SSIM 0.374/0.398。  
**第二**，GT + Strict MNN（0.258/0.289）、HGNN + Strict MNN（0.234/0.273）——加了 MNN 仍不如 MLP + MNN。  
**第三**，**HGNN-512 + MNN 的 SSIM 极低：0.072/0.055**——空间结构严重失真；MLP + MNN 的 SSIM 最高。  
**第四**，MNN + Cycle 低于纯 MNN——Cycle 与外部监督冲突。

如果瓶颈在编码器，事实不应如此。事实说明：

> 在当前数据集、official split 和 strict 协议下，决定结果的首先不是模型有多复杂，而是监督信号是否可靠。

---

## 第 45 页：Top Marker Genes 提升

> 建议时长：45 秒

平均 PCC 提升，是否只是少数基因拉高？看 marker 基因：

Slice2 上，CPA3 从 0.079 到 0.578（+0.498），CTSG +0.483，CLEC14A +0.308，EGFL7 +0.217，CLDN5 +0.216。涉及免疫、上皮/基质、血管等过程。

MNN 的改进 **不只在平均指标，也在生物学关键基因上。** 这不是一个「整体好看、局部全错」的结果。

---

## 第 46 页：章节分隔 — 八、讨论

> 建议时长：15 秒

实验结果已经给出。下面回答：**为什么简单 MLP 能打败复杂网络？** 这不是为 MLP 辩护，而是总结方法论的边界。

---

## 第 47 页：为什么 MLP + MNN 反而最强？

> 建议时长：60 秒

六点解释，对应六组独立观察：

**第一**，任务本质是 panel-to-panel 分子映射，不是空间聚合——同切片 measured panel 与 missing panel 有生物学相关性。

**第二**，MNN 已经编码空间/形态信息——H&E MNN 配形态相似细胞，Panel B MNN 配表达相似细胞；不是没用空间，而是 **MNN 已经用了，HGNN 再用 H&E 聚合是重复且引入噪声**。

**第三**，HGNN/GT 额外空间聚合会 over-smooth——Branch Decomposition 已证明 H&E branch ≈ 0。

**第四**，Cycle 与 MNN 冲突——MNN 是外部锚定，Cycle 是自洽约束；MNN 可靠时，Cycle 把预测拉离 MNN 目标。

**第五**，HGNN + MNN 的 SSIM 极低——复杂图模型在弱监督下伤害空间保真；MLP + MNN 的 SSIM 最高。

**第六**，监督强时简单 MLP 足够——复杂网络需要大量有效监督才能发挥优势；strict 协议下监督稀缺且带噪，MLP 更稳健。

**复杂并不自动等于有效。**

---

## 第 48 页：为什么 MLP 能超越 MNN pseudo-label 本身？

> 建议时长：50 秒

一个尖锐问题：监督来自 MNN，为什么训练后 MLP 比 pseudo-label 本身还强？

**Strict MNN pseudo-label 质量**：0.153/0.187。  
**训练后 MLP**：0.334/0.371。提升约 +0.18。

因为 pseudo-label 是 **带噪的 per-cell 估计**——每个细胞来自少数邻居平均；Slice2 还要经 pseudo_B1 两步桥接，误差累积。

MLP 学到的是 **全局 panel-to-panel 映射**：输入是同切片 measured panel（强信号），目标是有噪 pseudo-label；模型把噪声平均掉，直接学 YB2→YA2，绕过两步桥接。

结论：**MLP 把有噪但方向正确的 pseudo-label 蒸馏成稳定 translator。** 这不是 magic，是监督形态与模型容量匹配的结果。

---

## 第 49 页：结论

> 建议时长：60 秒

结论回答三个问题，不重复实验清单。

**我们做成了什么？**  
完成 SpatialEx/SpatialEx+ 复现，修复 6 个官方 bug；尝试 GT 替换 HGNN；用 Branch Decomposition 拆解信号；提出 Strict MNN 并在 Shuffle 实验中验证。

**我们发现了什么？**  
Fig.3 的主要瓶颈 **不是编码器容量，而是 missing-panel 监督的可靠性。** H&E 在 strict panel diagonal 下不是主信号；Cycle 自洽不等于外部正确；Strict MNN 提供的外部锚定，在当前设定下最有效。

**这件事说明什么？**  
在缺失模态和弱监督任务中，**先要解决「学习目标从哪里来」，再讨论「用什么模型学习」。**

最后一句，有边界地收束：

> 我们没有证明简单模型永远优于复杂模型。我们证明的是：**在监督信号尚不可靠的时候，继续堆叠模型复杂度，往往解决不了真正的问题。**

---

## 第 50 页：局限与后续方向

> 建议时长：50 秒

任何结论都有边界，我们主动说明局限：

**第一**，Slice1 方向仍较弱——与 official split 信息桥不对称有关，不是方法在 Slice2 有效、Slice1 必然同样有效。  
**第二**，仅在单一乳腺癌数据集、official 150/163 split、strict 协议下验证——**不能无限扩大为普遍规律。**  
**第三**，未与 DeepPT 在 Fig.3 做系统对比；CNN_Reg、Hist2ST 等待实现。  
**第四**，GT 受显存限制，Cross-Attention Translator 退化为 MLP——GT 的上限尚未完全探明，但现有结果已足以否定「编码器是 Fig.3 主瓶颈」。  
**第五**，MNN pseudo-label 有噪声，尚未显式建模不确定性。  
**第六**，Fig.4 首次 PCC 受数据对齐错误影响，重训进行中；Omics Diagonal 仅为初步结果。

后续：多模态 matching 加位置编码；更多癌种验证；与 DeepPT 等公平比较；MNN 不确定性建模；向 Omics Diagonal 和 Fig.4 扩展——**每项都要在对应 strict 协议下单独检验。**

---

## 第 51 页：谢谢

> 建议时长：20 秒

以上是我的汇报。感谢各位老师聆听，请批评指正。

---

# 附录：答辩追问备用（第 52–58 页）

> 不计入主时长；每页 30–60 秒，按需展开。

---

## 第 52 页：Appendix

> 建议时长：20 秒

下面是附录：参数敏感性、Latent MNN、更多可视化。老师若感兴趣我可以展开；否则留到问答环节。

---

## 第 53 页：MNN 参数敏感性扫描

> 建议时长：45 秒

**问：MNN 结果是不是调参调出来的？**

横轴 k 或 mnn_k，纵轴 gene-level PCC。MNN 在较宽参数范围内稳定，默认 k=5、mnn_k=20 位于平坦区域。**不是依赖某个特定超参数凑出来的。** 当然，换数据集、换 split，最优 k 可能需要重调——这是适用范围问题，不是鲁棒性造假。

---

## 第 54 页：跨切片 Latent Alignment 与 Latent MNN

> 建议时长：50 秒

**问：matching 空间有没有更好的选择？**

| Matching space     | Slice1 | Slice2    |
| ------------------ | ------ | --------- |
| raw measured panel | 0.015  | 0.264     |
| PCA latent (50-d)  | 0.007  | **0.291** |
| CORAL aligned      | 0.010  | 0.228     |

PCA latent 在 Slice2 略升；CORAL 反而降——可能过度抹平生物学相关的切片间差异。Slice1 始终接近 0——**该方向信息桥本身弱**，与 official split 不对称一致。Latent MNN 是探索方向，不是当前主方法。

---

## 第 55 页：Top Marker Genes 提升（附录版）

> 建议时长：40 秒

CTLA4 +0.200，PTPRC +0.154，ESR1 +0.126，CLEC14A +0.137——免疫、激素、血管相关。与正文第 45 页互补，说明提升分布在多个生物学基因上，不是个别 outlier 拉高平均。

---

## 第 56 页：MNN & KNN — Slice2 Marker Gene 空间可视化

> 建议时长：45 秒

Ground Truth 对比 MLP+Strict MNN 与 HGNN/GT+Cycle 的空间表达图。MNN 预测热点与真值对齐更好；Cycle 常见过平滑或错位。这是 SSIM 提升的视觉证据——**空间保真不是次要指标，它反映预测有没有生物学意义。**

---

## 第 57 页：MNN & KNN — Per-Gene PCC 分布

> 建议时长：40 秒

MLP + Strict MNN 的 PCC 分布整体右移；Cycle 方法集中在 0 附近甚至负值。说明改进是 **系统性的**，不是少数基因偶然拉高。

---

## 第 58 页：MNN & KNN — Per-Gene 提升分布

> 建议时长：40 秒

ΔPCC（MNN − Cycle）：**78.7% 基因获得正提升**；少数基因下降——说明仍有改进空间，也提示未来可做 gene-specific 或 uncertainty-weighted 训练。不能宣称「所有基因都变好」，但多数基因确实变好。

---

# 附录：高频问答（备用）

**Q1：为什么 MNN 在 Slice2 比 Slice1 提升更大？**  
A：official panel split 导致信息桥不对称。Slice2 的 B-panel 桥更强；Slice1 方向可预测信号本身弱。这是数据设定问题，不是方法在 Slice2 上「特别神」。

**Q2：H&E 没用，原文为什么还要 H&E？**  
A：Fig.2 上 H&E 有用，HGNN 优于 GT。Fig.3 strict panel diagonal 下，measured panel 已是更强信号，H&E 变成弱模态——**任务设定不同，信号主次不同。**

**Q3：MNN 噪声怎么处理？**  
A：当前未显式建模；MLP 训练有降噪效果。后续可对配对置信度加权。

**Q4：能推广到 Omics Diagonal 吗？**  
A：原则上可以，核心是跨切片 MNN 构造 pseudo-label。Omics 上 protein→RNA 有初步提升，RNA→protein 仍弱——**方向难度不同，需单独验证。**

**Q5：Cycle + MNN 为什么下降？**  
A：MNN 是外部锚定，Cycle 是自洽约束，优化目标不一致。

**Q6：Fig.4 PCC 为什么那么低？**  
A：首次复现 HE npy 与表达矩阵未按 barcode 对齐，不是 split 错误。loader 已修复，**重训进行中，最终指标待更新。** 不能把 0.008/0.013 写成方法结论。

**Q7：GT  hidden 只有 128，对比公平吗？**  
A：受显存限制。但 Fig.2 上 GT-512 仍不如 HGNN；Fig.3 上 GT+Cycle 也不如 HGNN+Cycle——**现有证据足以否定「编码器是 Fig.3 主瓶颈」，但不能宣称 GT 上限已完全探明。**

---

> **Fig.4 重训占位**：`experiments/fig4/spatialexp_big/outputs/metrics_spatialexp_big.csv` 更新后，请将第 14、17 页及相关问答中的 Fig.4 PCC 替换为新数值；SSIM/CMD 若变化不大可保留表述框架。
