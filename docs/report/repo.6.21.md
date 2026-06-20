# 答辩内容稿 / Defense Outline（40 min）
> 本稿作为梳理答辩内容的文件, 具体答辩文字稿件另起.
> 论文：*High-parameter spatial multi-omics through histology-anchored integration*（Nature Methods 2025）
> 任务：复现 SpatialEx/SpatialEx+，定位 bug，提出并验证与原文框架显著不同的改进算法。
> 答辩日期：6 月 21 日下午 5 点，线上，约 40 分钟。

---


## 0. 建议时间分配

| 章节                                          | 时间  | 内容                                  |
| --------------------------------------------- | ----- | ------------------------------------- |
| 1. 背景与任务                                 | 4 min | 问题定义、论文方法、Fig.3 任务        |
| 2. 环境搭建与复现                             | 6 min | 数据集、跑通任务、修复的 bug          |
| 3. Fig.3 的数学化描述                         | 4 min | 符号、约束、strict 协议               |
| 4. 改进一：网络架构（GT 替代 HGNN）           | 5 min | 动机、实现、效果                      |
| 5. 诊断：分支贡献分析（Branch Decomposition） | 4 min | H&E branch 弱、panel branch dominant  |
| 6. 改进二：监督信号（MNN 替代 Cycle）         | 8 min | Cycle trap、MNN 机制、Mutual 过滤     |
| 7. 实验结果与“MLP 反而最强”                   | 5 min | 官方 split 总表、per-gene、空间可视化 |
| 8. 结论与展望                                 | 4 min | 核心结论、局限、后续方向              |

答辩逻辑: 

```md
1. 任务背景与 strict 协议
   ↓
2. 第一次尝试：改进网络架构（GT 替代 HGNN）
   → 结果：基本持平，没有优势
   ↓
3. 诊断：分支贡献分析（Branch Decomposition）
   → 发现：H&E branch 接近 0，panel branch dominant
   → 结论：瓶颈不在网络架构，而在监督信号
   ↓
4. 第二次尝试：改进监督信号（Strict MNN 替代 Cycle）
   → Cycle self-consistency trap
   → Strict MNN 机制
   → 结果：显著提升
   ↓
5. 结果总表与排列组合解读
   ↓
6. 结论与展望
```

---

## 1. 背景与任务（第 1 章）

### 1.1 问题背景

**空间组学的高参数困境**
- 现有空间组学技术（Xenium、MERFISH、CODEX 等）在分辨率、通量、panel 大小之间存在权衡。
- 同一组织很难同时测得“高分辨率 + 大 panel + 多组学”。
- 替代方案：在相邻切片上分别测互补 panel / 互补组学，再通过计算整合。
- 这引出了 **spatial diagonal integration**：两个切片没有共测的 omics 特征，需要跨切片补全。

**两种 diagonal integration**
- **Panel diagonal integration（Fig.3）**：同一切片技术、不同基因 panel。
- **Omics diagonal integration**：不同组学类型（如转录组 + 蛋白）。

本次工作主要聚焦 **Fig.3 panel diagonal integration**。

### 1.2 论文方法速览

- **SpatialEx**：H&E foundation model（UNI）→ 细胞级 H&E 嵌入；构建空间超图；HGNN 编码；DGI 对比学习；预测单细胞 omics。
- **SpatialEx+**：在 SpatialEx 基础上加入 **omics cycle module**，通过跨切片、跨 panel 的循环一致性约束，实现无共测数据情况下的整合。

### 1.3 演讲核心信息

> 我们的探索分三步：
> 1. **网络架构**：把 HGNN 换成 Graph Transformer（GT），但发现效果基本持平；
> 2. **机制诊断**：通过 Branch Decomposition 发现，有效信号主要来自 panel-to-panel 分支，H&E 分支接近随机；
> 3. **监督信号**：提出 **Strict MNN 伪标签** 替代原 Cycle consistency，配合轻量 MLP 取得显著提升。
>
> 最终 surprising finding：**MLP + MNN 是所有模型里最强的**，说明 Fig.3 任务的瓶颈不是网络表达力，而是监督信号的质量。

---

## 2. 环境搭建与复现（第 2 章）

### 2.1 数据集

使用论文主图 Fig.3 的 **Xenium Human Breast Cancer** 数据：

| 文件                                                  | 内容                                         | 大小    |
| ----------------------------------------------------- | -------------------------------------------- | ------- |
| `Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad` | Slice 1，~313 基因，UNI H&E 嵌入             | ~990 MB |
| `Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad` | Slice 2，~313 基因，UNI H&E 嵌入             | ~674 MB |
| `data/panel_split_official.csv`                       | 官方 150/163 gene split（Panel A / Panel B） | -       |

两个切片各自包含：
- `adata.X`：基因表达矩阵；
- `adata.obsm['he']`：UNI 预训练 H&E 嵌入；
- `adata.obsm['spatial']`：细胞空间坐标。

### 2.2 跑通的任务

1. **SpatialEx（Fig.2 H&E→omics）**：Rep1/Rep2 跨切片协议，见 §2.4。
2. **SpatialEx+（Cycle baseline，Fig.3）**：使用官方 panel split，跑通 HGNN + Cycle。
3. **SpatialEx-GT（改进网络）**：Graph Transformer + MFP 替代 HGNN + DGI。
4. **Conditional MLP + MNN（改进监督）**：Strict MNN 伪标签 + 2-layer MLP。
5. **Conditional GT/HGNN + MNN**：把 MNN 伪标签接到图模型上，做对照。

### 2.3 发现的 Bug（已修复，见 docs/BUGFIXES.md）

| #   | 位置                                                         | 问题                                 | 修复                      |
| --- | ------------------------------------------------------------ | ------------------------------------ | ------------------------- |
| 1   | `preprocess.normalize_graph`                                 | 使用未定义变量 `adj`                 | `adj = H.copy()`          |
| 2   | `preprocess.Build_graph` / `Build_hypergraph_spatial_and_HE` | `return_type == 'crs'` 拼写错误      | 改为 `'csr'`              |
| 3   | `Build_hypergraph_spatial_and_HE`                            | 默认返回 `coo_matrix`，子图索引报错  | 默认改为 `csr`            |
| 4   | `model.Model_Plus.forward` + `SpatialExP.train`              | `agg_mtx` 与 `x_prime` 维度不匹配    | 增加 `selection` 切片     |
| 5   | `SpatialExP.train` 中 `Regression`                           | `agg_mtx` 与预测维度不匹配           | 传入前做 `selection` 切片 |
| 6   | `Model_Plus` / `Regression`                                  | `BatchNorm1d` 在 batch_size=1 时崩溃 | 替换为 `LayerNorm`        |

> 这些 bug 是官方代码运行时直接抛错的阻塞点，修复后才能在真实数据上跑通 SpatialEx+。

### 2.4 Fig.2 H&E→omics 复现与 baseline 对比

Fig.2 任务与 Fig.3 不同：**输入仅为 H&E 特征，输出为全转录组（313 genes）**，不涉及 panel split 或随机留基因。

**数据**：与 Fig.3 相同，`Human_Breast_Cancer_Rep1`（Slice1）与 `Rep2`（Slice2），各含 UNI H&E 嵌入（`obsm['he']`）与 Xenium 测得的 RNA 表达（`adata.X`）。

**训练/测试协议（cross-slice，非多折）**：

- **不是**随机留出部分细胞做多折交叉验证；
- **不是**随机留出部分基因（那是 Fig.3）；
- 采用 **双向跨切片** train→test：
  - 在 Slice A 上学习 H&E→表达，在 Slice B **全部细胞**上预测并评估；
  - 反向再做一次（B 训练 → A 测试）。

| 方法           | Slice1 训练来源   | Slice1 评估       | Slice2 训练来源   | Slice2 评估       |
| -------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| SpatialEx / GT | Rep2（module_HB） | Rep1 预测 vs 真实 | Rep1（module_HA） | Rep2 预测 vs 真实 |
| DeepPT         | Rep2（90% 细胞）  | Rep1 全部细胞     | Rep1（90% 细胞）  | Rep2 全部细胞     |

SpatialEx 训练时两个 slice 的模型 **各在本 slice 有监督**（H&E + 真实表达），但 **测试时交叉使用对侧模型**（`auto_inference`）。DeepPT 在训练 slice 内留 10% 做 validation early stopping，测试 slice 完全不参与训练。

**脚本与输出**（`experiments/fig2/`）：

| 实验           | 脚本                                           | 输出目录                           |
| -------------- | ---------------------------------------------- | ---------------------------------- |
| HGNN SpatialEx | `scripts/fig2/run_fig2_spatialex.py`           | `spatialex_breast_cancer/outputs/` |
| DeepPT         | `scripts/fig2/run_fig2_deeppt.py`              | `deeppt_breast_cancer/outputs/`    |
| GT-128 + MFP   | `scripts/fig2/run_fig2_gt.py`                  | `gt_breast_cancer/outputs/`        |
| GT-512 + MFP   | `scripts/fig2/run_fig2_gt.py --hidden_dim 512` | `gt512_breast_cancer/outputs/`     |
| GT-128 + DGI   | `scripts/fig2/run_fig2_gt_dgi.py`              | `gt_dgi_breast_cancer/outputs/`    |

**主结果（gene-level PCC / SSIM / CMD）**：

| 方法                     | Slice1 PCC | Slice2 PCC | Slice1 SSIM | Slice2 SSIM | Slice1 CMD | Slice2 CMD |
| ------------------------ | ---------: | ---------: | ----------: | ----------: | ---------: | ---------: |
| **HGNN SpatialEx (512)** |  **0.257** |  **0.273** |   **0.419** |   **0.425** |  **0.205** |  **0.207** |
| DeepPT                   |      0.268 |      0.276 |       0.357 |       0.366 |      0.271 |      0.281 |
| GT-512 + MFP             |      0.244 |      0.246 |       0.339 |       0.329 |      0.234 |      0.241 |
| GT-128 + MFP             |      0.228 |      0.236 |       0.309 |       0.320 |      0.239 |      0.252 |
| GT-128 + DGI             |      0.225 |      0.235 |       0.297 |       0.313 |      0.238 |      0.248 |

> **Fig.2 小结**：
> 1. **PCC**：DeepPT 略高（~0.27），HGNN SpatialEx 次之；GT 系列（128/512，MFP/DGI）均低于 HGNN SpatialEx；
> 2. **SSIM / CMD**：HGNN SpatialEx 明显最好（SSIM ~0.42，CMD ~0.21），说明空间结构与基因-基因关系保留更好；
> 3. **GT hidden dim**：512 维较 128 维 PCC 提升约 +0.01~0.02，但仍不及 HGNN-512；
> 4. **自监督形式**：GT 上 MFP vs DGI 几乎无差别，换 DGI 未带来 Fig.2 增益；
> 5. 与 Fig.3 结论形成对照：Fig.3 上 GT≈HGNN 是因为 H&E branch 弱；Fig.2 纯 H&E→omics 任务上 **HGNN 仍优于 GT**，说明 encoder 选择在该任务上仍有意义。

Fig.2 与 Fig.3 的 **关键区别**：Fig.2 没有 measured panel 作为分子锚点，模型必须 purely 从 H&E 推断 313 个基因，难度更高；因此 Fig.3 上「MLP+MNN 最强」的结论 **不能外推** 到 Fig.2。

---

## 3. Fig.3 任务的数学化描述（第 3 章）

### 3.1 符号设定

两个相邻切片：
- Slice 1：H&E 特征 $X_1 \in \mathbb{R}^{n_1 \times d_{he}}$，已测 panel $Y_A^1 \in \mathbb{R}^{n_1 \times |A|}$，缺失 panel $Y_B^1 \in \mathbb{R}^{n_1 \times |B|}$。
- Slice 2：H&E 特征 $X_2 \in \mathbb{R}^{n_2 \times d_{he}}$，已测 panel $Y_B^2 \in \mathbb{R}^{n_2 \times |B|}$，缺失 panel $Y_A^2 \in \mathbb{R}^{n_2 \times |A|}$。

**目标**：
$$
\hat{Y}_B^1 = f_1(X_1, Y_A^1), \qquad \hat{Y}_A^2 = f_2(X_2, Y_B^2).
$$

### 3.2 Strict Fig.3 协议

训练时 **绝对不可用** held-out panel：
- 不可用 $Y_B^1$；
- 不可用 $Y_A^2$。

因此模型不能直接学习 $Y_A^1 \to Y_B^1$ 或 $Y_B^2 \to Y_A^2$ 的真实映射，必须依赖：
1. H&E 提供的跨切片形态先验；
2. 跨切片 matching 构造的 pseudo-label；
3. 某种自监督 / 循环一致性约束。

### 3.3 评价指标

主要使用 **gene-level PCC**（Pearson correlation coefficient），辅以 SSIM、RMSE、CMD。

- Slice 1：评估 $\hat{Y}_B^1$ 与真实 $Y_B^1$ 的相关性；
- Slice 2：评估 $\hat{Y}_A^2$ 与真实 $Y_A^2$ 的相关性。

---

## 4. 改进一：网络架构——用 Graph Transformer 替代 HGNN（第 4 章）

### 4.1 动机

原 SpatialEx/SpatialEx+ 使用 **HGNN（Hypergraph Neural Network）**：
- 每个细胞作为一个节点；
- 每个细胞与其 $k$ 个空间近邻构成一条超边；
- 消息传递时，超边内邻居做固定加权聚合。

**问题**：HGNN 的聚合权重是固定的（由超图结构决定），无法根据特征相似性自适应调整。如果想“捕捉空间信息的同时，给更相似的邻居更高权重”，自然想到 **Graph Transformer**。

### 4.2 实现：SpatialEx-GT

在 `SpatialEx/model_improved.py` 中实现：
- **Graph Transformer Layer**：稀疏邻居注意力，仅对空间近邻计算 $QK^T$，复杂度 $O(N \cdot k)$ 而非 $O(N^2)$；
- **Masked Feature Prediction（MFP）**：替换 DGI，随机 mask 部分节点后重建 H&E 特征；
- **Cross-Attention Translator**：接口保留，但出于显存限制实现为轻量 MLP；
- 对应训练器：`SpatialEx/SpatialEx_improved.py` → `SpatialExP_GT`。

### 4.3 结果

在 Cycle 监督下：

| 编码器   | 监督  | Slice1 PCC | Slice2 PCC |
| -------- | ----- | ---------: | ---------: |
| HGNN-512 | Cycle |      0.275 |      0.301 |
| GT-128   | Cycle |      0.267 |      0.276 |

> GT 与 HGNN 基本持平，没有形成明显优势。
>
> **补充（Fig.2）**：在纯 H&E→全转录组任务上（§2.4），HGNN-512 仍优于 GT-128/512（PCC 0.257/0.273 vs 0.228~0.246），说明 GT 并非 universally 更优；Fig.3 上 GT≈HGNN 更可能与 H&E branch 弱、任务以 panel 信号为主有关。
>
> 但为什么更强的图网络没有用？我们需要先回答：**SpatialEx+ 内部的有效信号到底来自 H&E，还是来自已测 panel？**

---

## 5. 诊断：分支贡献分析（Branch Decomposition）（第 5 章）

### 5.1 问题

SpatialEx+ 同时存在两条路径：
- **H&E-driven branch**：$X \to \hat{Y}_{\text{missing}}$；
- **Panel-to-panel branch**：$Y_{\text{measured}} \to \hat{Y}_{\text{missing}}$。

如果 H&E branch 很强，换 GT 应该提升；如果 panel branch 是主要信号，那么 GT 帮助有限。

### 5.2 实验设计

将最终预测拆成两条独立路径，并比较单 branch、简单平均融合、reliability-weighted 融合：

$$
\hat{Y}_{B,X}^{1}=F_B(X_1), \qquad \hat{Y}_{A,X}^{2}=F_A(X_2)
$$

$$
\hat{Y}_{B,C}^{1}=C_{A\rightarrow B}(Y_A^1), \qquad \hat{Y}_{A,C}^{2}=C_{B\rightarrow A}(Y_B^2)
$$

### 5.3 结果

| Variant                           | Slice1 PanelB PCC | Slice2 PanelA PCC |
| --------------------------------- | ----------------: | ----------------: |
| H&E branch                        |            -0.001 |             0.010 |
| Panel branch                      |         **0.016** |         **0.248** |
| 0.5 average (calibrated)          |             0.011 |             0.177 |
| Reliability-weighted (calibrated) |            -0.001 |             0.010 |

> 注：以上数字来自随机 split 机制诊断实验，定性结论对 official split 同样成立。

### 5.4 结论

- **H&E branch 接近 0**：在当前 Fig.3 跨切片 setting 下，H&E 形态特征不能形成稳定的 missing panel 先验；
- **Panel branch dominant**：有效信号主要来自 measured panel → missing panel 的分子映射；
- **Late fusion 有害**：把 H&E 与 panel 拼接融合，反而把强信号拉低，说明 H&E 是噪声而非补充。

**因此**：GT 没有提升，不是因为图网络不够强，而是因为 **H&E 这条输入本身信息量低**。后续改进的重点应该是监督信号，而不是网络架构。

---

## 6. 改进二：监督信号——从 Cycle 到 Strict MNN（第 6 章）

### 6.1 原方法 Cycle Consistency 的依赖链条

SpatialEx+ 的 cycle loss 可以概括为：

- Slice 1：$Y_A^1 \xrightarrow{C_{A\to B}} \hat{Y}_B^1 \xrightarrow{C_{B\to A}} \tilde{Y}_A^1$，约束 $\tilde{Y}_A^1 \approx Y_A^1$。
- Slice 2：$Y_B^2 \xrightarrow{C_{B\to A}} \hat{Y}_A^2 \xrightarrow{C_{A\to B}} \tilde{Y}_B^2$，约束 $\tilde{Y}_B^2 \approx Y_B^2$。

这个循环的合理性依赖一条链：

$$
\text{HGNN 捕捉空间信息} \Rightarrow \text{HE} \to Y_A^1 \text{ 学得准确} \Rightarrow C_{A\to B} \text{ 可靠} \Rightarrow \hat{Y}_B^1 \text{ 接近真实}.
$$

**但是**：如果 $HE \to Y_A^1$ 学到了幻觉（hallucination），cycle 只是在 **加强这个幻觉**。cycle loss 下降只保证 $Y_A \to \hat{Y}_B \to Y_A'$ 自洽，不保证 $\hat{Y}_B \approx Y_B$。

我们把这一现象称为 **cycle self-consistency trap**。

实验验证：纯 Cycle-only MLP（无 MNN）结果接近随机：

| 模型             | Slice1 PCC | Slice2 PCC |
| ---------------- | ---------: | ---------: |
| MLP + Cycle only |      0.005 |      0.013 |

### 6.2 新监督信号：跨切片互补 + 近邻聚合 + Mutual 过滤

**直觉**：既然两个切片来自相邻组织，存在结构连续性，那么 Slice 1 的细胞在 Slice 2 中应该能找到“平行世界中的另一个自己”。

**直接互补的问题**：
- 双向一一映射太硬，切片间细胞不完全对齐；
- 批次效应和噪声会导致错误配对。

**改进：kNN 聚合 + Mutual Nearest Neighbor（MNN）过滤**
- 对 Slice 1 的每个细胞，在 Slice 2 的特征空间中找到 $k$ 个近邻；
- 只保留 **互为近邻** 的配对（mutual），剔除单向噪声匹配；
- 用保留邻居的目标 panel 表达加权平均，构造 pseudo-label。

### 6.3 Strict MNN 伪标签构建（Fig.3 合规）

必须保证不使用 held-out panel。我们设计了两步桥接：

**Step 1（Slice 1 缺失 Panel B 的伪标签）**：
- 用 H&E 特征做跨切片 MNN：$X_1 \leftrightarrow X_2$；
- 把 Slice 2 已测的 $Y_B^2$ 转移给 Slice 1，得到 $\tilde{Y}_B^1$。

**Step 2（Slice 2 缺失 Panel A 的伪标签）**：
- 用 B-panel 做跨切片 MNN：$Y_B^2 \leftrightarrow \tilde{Y}_B^1$；
- 把 Slice 1 已测的 $Y_A^1$ 转移给 Slice 2，得到 $\tilde{Y}_A^2$。

全程未使用 $Y_B^1$ 和 $Y_A^2$。

实现：`SpatialEx/SpatialEx_conditional_gt.py` → `build_strict_mnn_pseudo_labels`。

### 6.4 与文献的联系

MNN 的思想在单细胞整合中已被广泛使用，例如：
- Haghverdi et al., *Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors* (Nature Biotechnology, 2018)；
- 其核心是通过互惠关系过滤跨批次/跨样本的低质量匹配。

我们把这一思想迁移到 **无共测 panel 的空间 diagonal integration** 中，作为替代 cycle 的显式监督信号。

---

## 7. 实验结果与“MLP 反而最强”（第 7 章）

### 7.1 Official split 主结果

数据：`data/panel_split_official.csv`（150 A / 163 B）。

| 编码器   | 监督               | Slice1 PCC | Slice2 PCC | Slice1 SSIM | Slice2 SSIM |
| -------- | ------------------ | ---------: | ---------: | ----------: | ----------: |
| HGNN-512 | Cycle              |      0.275 |      0.301 |       0.308 |       0.332 |
| GT-128   | Cycle              |      0.267 |      0.276 |       0.345 |       0.357 |
| MLP      | Cycle only         |      0.005 |      0.013 |       0.114 |       0.107 |
| **MLP**  | **Strict MNN**     |  **0.334** |  **0.371** |   **0.374** |   **0.398** |
| MLP      | Strict MNN + Cycle |      0.315 |      0.353 |       0.344 |       0.388 |
| GT-128   | Strict MNN         |      0.258 |      0.289 |       0.359 |       0.387 |
| HGNN-512 | Strict MNN         |      0.234 |      0.273 |       0.072 |       0.055 |

> **关键发现**：
> 1. MLP + Strict MNN 在两个方向上都显著优于原 HGNN/GT + Cycle；
> 2. 加上 Cycle 后 MNN 性能反而下降，说明 Cycle 与 MNN 存在冲突；
> 3. 同监督下 MLP > GT > HGNN，图模型没有带来优势。

### 7.2 为什么 MLP 反而最强？

我们提出以下解释：

1. **任务本质是跨 panel 分子映射，不是空间聚合**：
   - 输入是已测 panel $Y_A^1$（或 $Y_B^2$），输出是缺失 panel $Y_B^1$（或 $Y_A^2$）；
   - 跨切片的空间/形态信息已经通过 MNN 伪标签编码进来了；
   - 此时再让 HGNN/GT 在空间上做额外聚合，反而可能把不同细胞类型的信号平均掉（over-smoothing）。

2. **MNN 本身携带了空间信息**：
   - H&E 桥和 B-panel 桥都建立在跨切片最近邻上；
   - 这些最近邻天然反映了空间连续性和生物学相似性；
   - MLP 学习的是“在 MNN 伪标签监督下的 panel-to-panel 映射”，足够完成任务。

3. **Cycle 与 MNN 冲突**：
   - MNN 提供的是“外部锚定的监督”；
   - Cycle 提供的是“自洽约束”；
   - 当 MNN 已经可靠时，Cycle 会把模型拉向自洽但不一定真实的解，导致性能下降。

4. **图模型在当前数据上反而是噪声源**：
   - 我们之前做过 branch decomposition：H&E branch 单独预测 PCC 接近 0；
   - 把 H&E 与 panel 拼接喂给 HGNN/GT，会把弱模态的噪声带入强模态信号中；
   - 这也是为什么 conditional HGNN/GT + MNN 不如纯 panel-only MLP。
   - 注意 HGNN-512 + MNN 的 SSIM 极低（0.07 / 0.06），说明它在空间结构上严重失真；MLP + MNN 的 SSIM 最高（0.37 / 0.40），空间保真也最好。

### 7.3 Per-gene 与空间可视化

- Slice 2 上 **78.7% 的基因**（118/150）通过 MNN 获得提升；
- 平均提升：Slice2 +0.0267 PCC；
- 提升显著的 marker genes：`CTLA4`、`PTPRC`、`ESR1`、`CLEC14A`。

示例（Slice 2）：

| Gene    | raw kNN PCC | MNN PCC |   提升 |
| ------- | ----------: | ------: | -----: |
| CTLA4   |       0.132 |   0.332 | +0.200 |
| PTPRC   |      -0.019 |   0.135 | +0.154 |
| ESR1    |       0.197 |   0.323 | +0.126 |
| CLEC14A |       0.453 |   0.590 | +0.137 |

对应图：`docs/image/fig3_diagnosis/11_marker_genes_slice2.jpg`。

> 这些基因涉及免疫、上皮/基质、血管等生物学过程，说明 MNN 的改进不只在平均指标上，也在真实空间表达结构上有视觉可辨的提升。

### 7.4 进一步尝试：Latent MNN

我们还尝试了把 matching 空间从 raw panel 投影到 PCA latent：

| Matching space     | Slice1 learned PCC | Slice2 learned PCC |
| ------------------ | -----------------: | -----------------: |
| raw measured panel |              0.015 |              0.264 |
| PCA latent (50-d)  |              0.007 |          **0.291** |
| CORAL aligned      |              0.010 |              0.228 |

- PCA latent + MNN 在 Slice2 上进一步提升到 0.291，说明降噪后的低维空间有助于跨切片匹配；
- CORAL 线性对齐反而下降，可能过度抹平了生物学相关的切片间差异；
- Slice1 始终接近 0，说明该方向的 panel 信息桥本身较弱，不是模型问题。

---

## 8. 结论与展望（第 8 章）

### 8.1 核心结论

1. **复现层面**：成功配置环境、跑通 SpatialEx/SpatialEx+，并修复官方代码中 5-6 个阻塞性 bug。
2. **机制诊断**：Fig.3 任务的瓶颈不是网络架构，而是 **缺失 panel 的监督信号质量**。
3. **算法改进**：
   - 网络侧：Graph Transformer 替代 HGNN，效果持平，未形成优势；
   - 监督侧：提出 **Strict MNN 伪标签**，替代原 Cycle consistency，取得显著提升。
4. ** surprising finding**：**MLP + Strict MNN** 在所有架构中表现最好（Slice1 0.334，Slice2 0.371），说明当监督信号可靠时，简单模型即可超过复杂图网络。

### 8.2 局限

1. **Slice1 方向仍然较弱**：即使在 MNN 下，Slice1 PCC 也只在 0.33 左右，远低于 Slice2 的 0.37；这与 official panel split 导致的信息桥不对称有关。
2. **仅在单一数据集验证**：Xenium Human Breast Cancer Rep1/Rep2；其他组织、其他 panel split 的泛化性待验证。
3. **Fig.2 baseline 已补充 DeepPT 与 GT 系列**（§2.4）；CNN_Reg、Hist2ST 等待实现。
4. **GT 的 Cross-Attention Translator 受显存限制退化为 MLP**：完整 $O(N^2)$ cross-attention 在数十万细胞切片上难以实现。

### 8.3 后续方向

1. **更好的跨切片 matching cost**：结合 H&E 形态、空间距离、基因表达的多模态 matching；
2. **生物学驱动的 panel split**：对比 random split 与功能相关/共表达 split，区分“任务本身难”与“模型不足”；
3. **多组学扩展**：把 MNN 伪标签方法应用到转录组-蛋白（transcriptomics-proteomics）diagonal integration；
4. **不确定性估计**：对 MNN 配对的置信度建模，低置信度区域不强制学习；
5. **可解释性分析**：哪些基因对 MNN 敏感，哪些对空间聚合敏感。

### 8.4 收尾一句话

> “在 SpatialEx+ 的 Fig.3 任务中，我们原本以为更强的图网络会带来提升，但实验告诉我们：**当缺失 panel 没有真实标签时，监督信号的质量比模型复杂度更重要。** Strict MNN 提供了一种简单、可解释、且显著优于原 Cycle baseline 的替代方案。”

---

## 附录：推荐答辩用图

| 图号 | 文件                                                    | 说明                                      |
| ---- | ------------------------------------------------------- | ----------------------------------------- |
| 1    | `docs/image/fig3_diagnosis/01_branch_decomposition.png` | Branch decomposition：H&E vs panel branch |
| 2    | `docs/image/fig3_diagnosis/07_cycle_trap.png`           | Cycle self-consistency trap 示意图        |
| 3    | `docs/image/fig3_diagnosis/06_mnn_pipeline.png`         | Strict MNN 伪标签流程                     |
| 4    | `docs/image/fig3_diagnosis/02_mnn_sweep.png`            | MNN 参数敏感性                            |
| 5    | `docs/image/fig3_diagnosis/03_per_gene_scatter.png`     | Per-gene PCC 散点（raw kNN vs MNN）       |
| 6    | `docs/image/fig3_diagnosis/11_marker_genes_slice2.jpg`  | Slice2 marker gene 空间可视化             |
| 7    | `docs/image/fig3_diagnosis/05_latent_alignment.png`     | PCA/CORAL latent MNN 对比                 |

---

## 附录：运行命令速查（答辩时可能被问到）

```bash
# Fig.2 HGNN SpatialEx
python scripts/fig2/run_fig2_spatialex.py \
  --out_dir experiments/fig2/spatialex_breast_cancer/outputs

# Fig.2 DeepPT
python scripts/fig2/run_fig2_deeppt.py \
  --out_dir experiments/fig2/deeppt_breast_cancer/outputs

# Fig.2 GT-512 + MFP
python scripts/fig2/run_fig2_gt.py --hidden_dim 512 \
  --out_dir experiments/fig2/gt512_breast_cancer/outputs

# Strict MNN + MLP（Fig.3 当前最佳）
python scripts/fig3/run_fig3_mnn_pseudo.py \
  --panel_csv data/panel_split_official.csv \
  --out_dir outputs/conditional/fig3_mnn_pseudo_strict_official

# HGNN + Cycle（论文 baseline）
python scripts/fig3/run_fig3_panel_split.py \
  --model spatialexp \
  --panel_csv data/panel_split_official.csv \
  --out_dir outputs/conditional/fig3_spatialexp_official

# GT + Strict MNN
python scripts/fig3/run_fig3_panel_split.py \
  --model conditional_gt_mnn --hidden_dim 128 \
  --panel_csv data/panel_split_official.csv \
  --out_dir outputs/conditional/fig3_conditional_gt_mnn_strict_official

# MLP + Strict MNN + Cycle
python scripts/fig3/run_fig3_panel_split.py \
  --model conditional_mnn_cycle_mlp \
  --panel_csv data/panel_split_official.csv \
  --out_dir outputs/conditional/fig3_mnn_cycle_strict_official
```
