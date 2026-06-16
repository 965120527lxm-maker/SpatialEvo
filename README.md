# SpatialEx / SpatialEx+ 复现与改进仓库

> 非官方复现与算法诊断 fork，基于论文 *High-Parameter Spatial Multi-Omics through Histology-Anchored Integration*（Nature Methods, 2025）。
>
> 原仓库：[KEAML-JLU/SpatialEx](https://github.com/KEAML-JLU/SpatialEx)  
> 本仓库：[965120527lxm-maker/SpatialEvo](https://github.com/965120527lxm-maker/SpatialEvo)

[![Nature Methods](https://img.shields.io/badge/Published-Nature%20Methods-blue)](https://www.nature.com/articles/s41592-025-02926-6)
[![BioRxiv](https://img.shields.io/badge/Preprint-BioRxiv-green)](https://www.biorxiv.org/content/10.1101/2025.02.23.639721v2.abstract)

---

## 内容提要

本仓库围绕 **SpatialEx / SpatialEx+** 做三件事：

1. **官方代码复现与 bug 修复**：在 `SpatialEx/preprocess.py`、`SpatialEx/model.py`、`SpatialEx/SpatialEx.py` 中修复了多个会导致官方实现无法运行的关键 bug，使 `SpatialExP` 可在本机环境跑通。
2. **架构改进尝试**：增加了基于 **Graph Transformer + Masked Feature Prediction (MFP) + Cross-Attention Translator** 的改进模型（`SpatialExP_GT`），用于和官方 HGNN/DGI/MLP 方案做对照。
3. **Fig.3 panel diagonal integration 机制诊断**：系统拆解了 no-co-measured 设置下的监督信号来源，指出在 Xenium Human Breast Cancer Rep1/Rep2、随机 150/163 gene split 条件下，有效信号主要来自 **measured-panel → missing-panel 的分子映射**，H&E branch 跨切片泛化有限；并验证了 MNN / PCA latent MNN 对 matching 质量的修正作用。

---

## 目录

- [仓库结构](#仓库结构)
- [安装与依赖](#安装与依赖)
- [快速复现：Fig.3 panel diagonal integration 诊断](#快速复现fig3-panel-diagonal-integration-诊断)
- [已修复的 bug](#已修复的-bug)
- [架构改进尝试](#架构改进尝试)
- [核心结论摘要](#核心结论摘要)
- [相关文档](#相关文档)
- [引用](#引用)
- [联系方式](#联系方式)

---

## 仓库结构

```
SpatialEx/
├── SpatialEx/                      # 核心包
│   ├── SpatialEx.py                # 官方 SpatialEx / SpatialEx+ 训练器（已修复 bug）
│   ├── SpatialEx_improved.py       # 改进版 SpatialExP_GT 训练器
│   ├── model.py                    # 官方模型（已修复 bug）
│   ├── model_improved.py           # Graph Transformer / MFP / Cross-Attention 改进模型
│   ├── preprocess.py               # 图构建与预处理（已修复 bug）
│   ├── utils.py
│   └── __init__.py
├── scripts/                        # 实验脚本
│   ├── fig3/                       # Fig.3 panel diagonal integration 诊断脚本
│   │   ├── run_fig3_mnn_pseudo.py
│   │   ├── run_fig3_latent_mnn.py
│   │   └── generate_fig3_figures.py
│   └── ...                         # 其他复现/对比脚本
├── curriculum/                     # 从 01 到 15 的渐进式实现章节
├── data/                           # 本地数据（需自行下载/软链）
├── docs/
│   ├── report/repo.6.16.md         # Fig.3 机制诊断报告（含 8 张诊断图）
│   ├── report/repo.6.9.md          # 前期复现与 bug 修复报告
│   ├── BUGFIXES.md                 # 详细 bug 修复记录
│   ├── A100_RUN_GUIDE.md           # A100 / 多卡运行指南
│   └── README_REPRODUCTION.md      # 早期复现说明
├── tutorials/                      # 官方教程 notebook
├── requirements.txt
├── setup.py
└── README.md                       # 本文件
```

---

## 安装与依赖

本机当前环境：

- Python 3.10
- PyTorch 2.7.0+cu128
- 2 × NVIDIA GeForce RTX 5090（31.36 GiB）

### 从源码安装

```bash
conda create -n spatialex python=3.10
conda activate spatialex
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

### 关键依赖（参考）

```
anndata==0.8.0
scanpy==1.9.3
numpy==1.23.5
pandas==2.0.3
cellpose==3.0.10
scikit-image==0.21.0
scikit-learn==1.3.2
scikit-misc==0.2.0
torch>=2.0
huggingface-hub==0.24.6
timm==1.0.8
torchvision>=0.18
```

> ⚠️ 由于官方代码依赖较老，建议逐个安装并检查兼容性。若在 `torch>=2.7` / `CUDA 12.8` 上遇到类型或稀疏矩阵相关告警，可参考 `docs/BUGFIXES.md` 和 `docs/A100_RUN_GUIDE.md` 中的兼容性说明。

---

## 快速复现：Fig.3 panel diagonal integration 诊断

### 1. 准备数据

下载 10x Xenium Human Breast Cancer 预处理后 h5ad（含 UNI 特征），放置于 `data/`：

```
data/Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad
data/Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad
```

### 2. 运行 MNN pseudo-label 主实验

```bash
python scripts/fig3/run_fig3_mnn_pseudo.py
```

- 比较 `raw kNN` 与 `MNN` pseudo-label
- 训练 measured-panel conditional MLP
- 输出 `outputs/conditional/mnn_pseudo_results.csv`

### 3. 运行 latent MNN 延伸实验

```bash
python scripts/fig3/run_fig3_latent_mnn.py
```

- 比较 `raw measured panel`、`PCA latent`、`CORAL aligned` 三种 matching space
- 输出 `outputs/conditional/latent_mnn_results.csv`

### 4. 生成诊断图

```bash
python scripts/fig3/generate_fig3_figures.py
```

生成图片保存至 `docs/image/fig3_diagnosis/`，对应报告中的图 1–8。

---

## 已修复的 bug

复现官方代码时，在 `preprocess.py`、`model.py`、`SpatialEx.py` 中修复了以下关键 bug（完整记录见 [`docs/BUGFIXES.md`](docs/BUGFIXES.md)）：

| Bug | 位置 | 问题 | 修复 |
| --- | --- | --- | --- |
| 未定义变量 `adj` | `preprocess.py` `normalize_graph` | 函数参数为 `H`，体内使用未定义的 `adj` | `adj = H.copy()` |
| `return_type` 拼写错误 | `preprocess.py` | `'crs'` 应为 `'csr'` | 统一改为 `'csr'` |
| 默认返回 `coo_matrix` | `preprocess.py` `Build_hypergraph_spatial_and_HE` | 子图索引需要 `csr_matrix` | 默认参数改为 `return_type='csr'` |
| `agg_mtx` 维度不匹配 | `SpatialEx.py` + `model.py` `Model_Plus.forward` | `x_prime` 包含整个 ROI，`agg_mtx` 仅覆盖 selection | 增加 `selection` 参数并切片 |
| `Regression.forward` 维度不匹配 | `SpatialEx.py` `SpatialExP.train` | `panel_1b/panel_2a` 维度与 `agg_mtx` 不匹配 | 传入前对 `selection` 切片 |

此外，针对本机 `PyTorch 2.7 + CUDA 12.8 + RTX 5090` 环境，还做了若干兼容性微调，例如稀疏矩阵类型推断、混合精度训练适配等。

---

## 架构改进尝试

在官方 SpatialEx+ 基础上，新增了 `SpatialExP_GT`（`SpatialEx_improved.py` / `model_improved.py`），核心改动：

| 模块 | 官方 SpatialEx+ | 改进 SpatialEx-GT | 备注 |
| --- | --- | --- | --- |
| 空间编码器 | HGNN | **Graph Transformer** | 自注意力替代固定超边聚合 |
| 自监督信号 | DGI | **Masked Feature Prediction (MFP)** | 掩码预测替代全局对比 |
| 组学翻译器 | MLP Regression | **Cross-Attention Translator** | 交叉注意力替代全连接映射 |
| 归一化 | BatchNorm | **LayerNorm** | 更适合 Transformer/序列建模 |

**当前状态**：在合成数据上改进模型相对官方版本有明显提升；但在真实 Xenium 数据上，由于 RTX 5090 单卡显存峰值限制，改进模型只能以 `hidden_dim=128` 运行，与官方 `hidden_dim=512` 基本持平。该改进分支保留在仓库中，可作为后续调参与 scaling 的起点。

---

## 核心结论摘要

基于 Xenium Human Breast Cancer Rep1 / Rep2、随机 150/163 gene split 的实验：

| 阶段 | Slice1 PanelB PCC | Slice2 PanelA PCC |
| --- | ---: | ---: |
| raw kNN direct transfer | 0.010 | 0.190 |
| raw kNN learned MLP | 0.014 | 0.238 |
| MNN pseudo-label + MLP | 0.015 | 0.265–0.268 |
| PCA latent + MNN + MLP | 0.007 | **0.291** |
| H&E branch only | ≈ 0 | 0.010 |
| conditional cycle | ≈ 0 | ≈ 0 |

主要结论：

1. **有效信号来源**：在当前设置下，SpatialEx+ 对 missing panel 的有效预测主要来自 measured panel 到 missing panel 的分子映射，而非 H&E 形态特征。
2. **多模态融合需谨慎**：H&E branch 较弱时，late fusion 会拉低 panel branch 性能。
3. **cycle consistency 陷阱**：cycle loss 下降不等于真实 missing panel 预测准确，模型可能学到自洽但无生物学意义的中间表示。
4. **matching 修正有效但有边界**：MNN 与 PCA latent MNN 能显著提升 Slice2 方向，但 Slice1 方向始终接近随机，说明 matching 只能改善已有信息桥，不能创造缺失的生物学对应关系。

完整论证见 [`docs/report/repo.6.16.md`](docs/report/repo.6.16.md)。

---

## 相关文档

- [`docs/report/repo.6.16.md`](docs/report/repo.6.16.md)：Fig.3 panel diagonal integration 机制诊断报告
- [`docs/BUGFIXES.md`](docs/BUGFIXES.md)：官方代码 bug 修复详细记录
- [`docs/A100_RUN_GUIDE.md`](docs/A100_RUN_GUIDE.md)：A100 / 多卡 / 大显存运行指南
- [`docs/README_REPRODUCTION.md`](docs/README_REPRODUCTION.md)：早期复现与改进模型说明
- [`curriculum/`](curriculum/)：从 01 到 15 的渐进式实现章节

---

## 引用

若使用本仓库，请同时引用原论文：

```bibtex
@article{liu2025high,
  title={High-Parameter Spatial Multi-Omics through Histology-Anchored Integration},
  author={Liu, Yonghao and Wang, Chuyao and Wang, Zhikang and Chen, Liang and Li, Zhi and Song, Jiangning and Zou, Qi and Gao, Rui and Qian, Binzhi and Feng, Xiaoyue and Guan, Renchu and Yuan, Zhiyuan},
  journal={Nature Methods},
  year={2025}
}
```

---

## 联系方式

- 本 fork 维护：李熹鸣（965120527lxm-maker）
- 原论文作者：见上方 bibtex

如有问题，欢迎通过 GitHub Issues 或邮件联系。
