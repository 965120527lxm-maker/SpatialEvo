# SpatialEx / SpatialEx+ 非官方复现与改进仓库

> 本仓库基于论文 *High-Parameter Spatial Multi-Omics through Histology-Anchored Integration*（Nature Methods, 2025）进行非官方复现、bug 修复与算法改进。
>
> 原仓库：[KEAML-JLU/SpatialEx](https://github.com/KEAML-JLU/SpatialEx)  
> 本仓库：[965120527lxm-maker/SpatialEvo](https://github.com/965120527lxm-maker/SpatialEvo)  
> 论文链接：[Nature Methods](https://www.nature.com/articles/s41592-025-02926-6)

---

## 目录

- [安装与依赖](#安装与依赖)
- [Quick Start](#quick-start)
- [已修复的 Bug](#已修复的-bug)
- [架构改进](#架构改进)
- [核心结论](#核心结论)
- [相关文档](#相关文档)
- [引用](#引用)
- [联系方式](#联系方式)

---

## 安装与依赖

### 推荐环境

- Python 3.10
- PyTorch ≥ 2.0（根据 CUDA 版本选择）
- CUDA 11.8 或 12.x

### 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n spatialex python=3.10 -y
conda activate spatialex

# 2. 安装 PyTorch（请根据你的 CUDA 版本选择）
# 例如 CUDA 12.x：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 安装其他核心依赖
pip install -r requirements.txt

# 4. 以 editable 模式安装本包
pip install -e .
```

### 核心依赖

```
anndata==0.8.0
scanpy==1.9.3
numpy==1.23.5
pandas==2.0.3
scipy==1.10.1
scikit-learn==1.3.2
cellpose==3.0.10
scikit-image==0.21.0
timm==1.0.8
huggingface-hub==0.24.6
```

> ⚠️ `requirements.txt` 中 PyTorch 版本为 2.3.1+cu118（对应 CUDA 11.8）。如果你的 CUDA 版本不同，请先安装匹配 CUDA 版本的 PyTorch，再安装 `requirements.txt` 中的其余包。

---

## Quick Start

### 1. 准备数据

下载 10x Xenium Human Breast Cancer 预处理后数据（已含 UNI H&E 嵌入），放置于 `data/`：

```
data/
├── Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad
└── Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad
```

### 2. 运行 Fig.3 panel diagonal integration 主实验

#### 2.1 MLP + Strict MNN（当前最佳）

```bash
python scripts/fig3/run_fig3_mnn_pseudo.py \
  --panel_csv data/panel_split_official.csv \
  --out_dir outputs/conditional/fig3_mnn_pseudo_strict_official
```

#### 2.2 官方 SpatialEx+ Cycle baseline

```bash
python scripts/fig3/run_fig3_panel_split.py \
  --model spatialexp \
  --panel_csv data/panel_split_official.csv \
  --out_dir outputs/conditional/fig3_spatialexp_official
```

#### 2.3 GT + Strict MNN

```bash
python scripts/fig3/run_fig3_panel_split.py \
  --model conditional_gt_mnn --hidden_dim 128 \
  --panel_csv data/panel_split_official.csv \
  --out_dir outputs/conditional/fig3_conditional_gt_mnn_strict_official
```

### 3. 生成诊断图

```bash
python scripts/fig3/generate_fig3_figures.py
```

生成图片保存至 `docs/image/fig3_diagnosis/`。

---

## 已修复的 Bug

在复现官方代码的过程中，我们定位并修复了 **6 个阻塞性 bug**，完整记录见 [`docs/BUGFIXES.md`](docs/BUGFIXES.md)。

| #   | 位置                                              | 问题                                               | 修复                        |
| --- | ------------------------------------------------- | -------------------------------------------------- | --------------------------- |
| 1   | `preprocess.py` `normalize_graph`                 | 使用未定义变量 `adj`                               | `adj = H.copy()`            |
| 2   | `preprocess.py`                                   | `return_type == 'crs'` 拼写错误                    | 改为 `'csr'`                |
| 3   | `preprocess.py` `Build_hypergraph_spatial_and_HE` | 默认返回 `coo_matrix`，不支持子图索引              | 默认改为 `csr`              |
| 4   | `model.py` + `SpatialEx.py` `Model_Plus.forward`  | `agg_mtx` 与 `x_prime` 维度不匹配                  | 增加 `selection` 参数并切片 |
| 5   | `SpatialEx.py` `SpatialExP.train`                 | `Regression.forward` 中 `agg_mtx` 与预测维度不匹配 | 传入前对 `selection` 切片   |
| 6   | `model.py` `Model_Plus` / `Regression`            | `BatchNorm1d` 在 batch_size=1 时崩溃               | 替换为 `LayerNorm`          |

完整修复记录见 [`docs/BUGFIXES.md`](docs/BUGFIXES.md)。

---

## 架构改进

本仓库在官方 SpatialEx/SpatialEx+ 基础上做了两类改进尝试：

### 1. 网络架构改进：SpatialEx-GT

用 Graph Transformer + Masked Feature Prediction 替代官方 HGNN + DGI：

| 模块       | 官方 SpatialEx+ | 改进 SpatialEx-GT                      |
| ---------- | --------------- | -------------------------------------- |
| 空间编码器 | HGNN            | Graph Transformer（稀疏邻居注意力）    |
| 自监督     | DGI             | Masked Feature Prediction              |
| 组学翻译器 | MLP Regression  | Cross-Attention Translator（轻量实现） |

实现文件：
- `SpatialEx/model_improved.py`
- `SpatialEx/SpatialEx_improved.py`

### 2. 监督信号改进：Strict MNN 伪标签

针对 Fig.3 任务，我们发现官方 Cycle consistency 存在 **self-consistency trap**，并提出用 **Mutual Nearest Neighbor（MNN）伪标签**作为替代监督信号：

- Slice 1 缺失 Panel B：H&E 跨切片 MNN 桥接，转移 Slice 2 的 $Y_B^2$；
- Slice 2 缺失 Panel A：B-panel 跨切片 MNN 桥接，转移 Slice 1 的 $Y_A^1$；
- 全程不使用 held-out panel，符合 Fig.3 strict 协议。

实现文件：
- `SpatialEx/SpatialEx_conditional_mlp.py`（MLP + MNN）
- `SpatialEx/SpatialEx_conditional_gt.py`（GT + MNN）
- `SpatialEx/SpatialEx_conditional_hgnn.py`（HGNN + MNN）

---

## 核心结论

### 论文任务复现与改进情况

论文主要提出了三类任务：

| 论文任务 | 对应方法 | 本仓库状态 | 说明 |
|----------|---------|-----------|------|
| **H&E-to-omics 预测** | SpatialEx（Fig.2） | 部分复现 | 单切片 H&E→omics 训练链路可跑通，但未系统对比论文中的 DeepPT、CNN_Reg 等 baseline |
| **Panel diagonal integration** | SpatialEx+（Fig.3） | 已复现 + 改进 | 在 Xenium Human Breast Cancer Rep1/Rep2 上跑通官方 Cycle baseline，并提出 Strict MNN 替代方案 |
| **Omics diagonal integration** | SpatialEx+ | 未复现 | 转录组-蛋白、代谢组-转录组等多组学任务尚未开展 |

**未复现/待补充内容**：
- 论文 Fig.2 中的 DeepPT、CNN_Reg、Hist2ST、THItoGene 等 baseline；
- 其他组织（human colon、human skin、mouse colon 等）的 H&E-to-omics 实验；
- 超百万细胞大尺度数据（`SpatialExP_Big`）的完整验证；
- Fig.3 之外的 omics diagonal integration 任务。

### Fig.3 official split 主结果

在 Xenium Human Breast Cancer Rep1/Rep2、`data/panel_split_official.csv`（150 A / 163 B）strict 协议下，主要结果如下（gene-level PCC）：

| 编码器   | 监督               | Slice1 PCC | Slice2 PCC | Slice1 SSIM | Slice2 SSIM |
| -------- | ------------------ | ---------: | ---------: | ----------: | ----------: |
| HGNN-512 | Cycle              |      0.275 |      0.301 |       0.308 |       0.332 |
| GT-128   | Cycle              |      0.267 |      0.276 |       0.345 |       0.357 |
| MLP      | Cycle only         |      0.005 |      0.013 |       0.114 |       0.107 |
| **MLP**  | **Strict MNN**     |  **0.334** |  **0.371** |   **0.374** |   **0.398** |
| MLP      | Strict MNN + Cycle |      0.315 |      0.353 |       0.344 |       0.388 |
| GT-128   | Strict MNN         |      0.258 |      0.289 |       0.359 |       0.387 |
| HGNN-512 | Strict MNN         |      0.234 |      0.273 |       0.072 |       0.055 |

**关键发现**：

1. **监督信号比网络架构更重要**：GT 替代 HGNN 效果基本持平，说明 Fig.3 任务的瓶颈不在网络表达力；
2. **Cycle consistency 存在自洽陷阱**：纯 Cycle-only MLP 接近随机，无法恢复真实 missing panel；
3. **Strict MNN 是更有效的监督**：MLP + Strict MNN 显著优于官方 HGNN/GT + Cycle；
4. **MLP 反而最强**：当 MNN 已经把跨切片空间/形态信息编码进伪标签后，简单的 panel-to-panel MLP 超过复杂图网络；
5. **Cycle 与 MNN 冲突**：MNN + Cycle 性能低于单独 MNN，说明自洽约束会干扰可靠的外部监督。

详细结果与机制分析见 [`docs/report/repo.6.20.md`](docs/report/repo.6.20.md) 和 [`docs/report/repo.6.16.md`](docs/report/repo.6.16.md)。

---

## 相关文档

| 文档                                                                                             | 内容                                 |
| ------------------------------------------------------------------------------------------------ | ------------------------------------ |
| [`docs/BUGFIXES.md`](docs/BUGFIXES.md)                                                           | 6 个官方代码 bug 的详细修复记录      |
| [`docs/report/bug_verification_original_repo.md`](docs/report/bug_verification_original_repo.md) | 原始仓库 bug 验证报告                |
| [`docs/A100_RUN_GUIDE.md`](docs/A100_RUN_GUIDE.md)                                               | A100 / CUDA 11.8 环境配置与运行指南  |
| [`docs/report/repo.6.16.md`](docs/report/repo.6.16.md)                                           | Fig.3 机制诊断报告（含 11 张诊断图） |
| [`docs/report/repo.6.20.md`](docs/report/repo.6.20.md)                                           | Official split 实验结果总表与解读    |
| [`docs/report/repo.6.21.md`](docs/report/repo.6.21.md)                                           | 40 分钟答辩演讲大纲                  |
| [`docs/README_REPRODUCTION.md`](docs/README_REPRODUCTION.md)                                     | 早期复现说明与合成数据验证           |

---

## 引用

如果本仓库对你的工作有帮助，请引用原论文：

```bibtex
@article{liu2025spatial,
  title={High-parameter spatial multi-omics through histology-anchored integration},
  author={Liu, Yonghao and Wang, Chuyao and Wang, Zhikang and Chen, Liang and Li, Zhi and Song, Jiangning and Zou, Qi and Gao, Rui and Qian, Bin-Zhi and Feng, Xiaoyue and Guan, Renchu and Yuan, Zhiyuan},
  journal={Nature Methods},
  volume={23},
  pages={373--386},
  year={2025},
  publisher={Nature Publishing Group}
}
```

---

## 联系方式

- 原论文作者：Yonghao Liu, Chuyao Wang 等（见论文作者列表）
- 原仓库：[KEAML-JLU/SpatialEx](https://github.com/KEAML-JLU/SpatialEx)
- 本复现/改进仓库：[965120527lxm-maker/SpatialEvo](https://github.com/965120527lxm-maker/SpatialEvo)

如有问题，欢迎通过 GitHub Issues 交流。
