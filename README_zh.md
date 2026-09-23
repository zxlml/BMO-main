<div align="center">

[English](README.md) | 简体中文

# BMO：一阶双层极小极大优化

**Fine-grained Analysis on the Stability and Generalization for First-order Bilevel Minimax Optimization**

本仓库提供了验证一阶双层极小极大算法理论泛化界的官方实现，并研究其与实际设置
（如最大内层迭代数、步长调度、元样本规模）之间的联系。

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-17%20passed-brightgreen.svg)](tests/test_bmo.py)

</div>

## 📰 新闻

- **[2026-09]** 以 `bmo_core` 包的形式重新实现了论文中的三个一阶求解器（**SSGDA**、**TSGDA-1**、**TSGDA-2**），并通过单元测试（17/17 通过）；发布了复现论文图 1–4 趋势的仿真实验。

## ✨ 概述

我们研究面向鲁棒数据（去噪 / 生成）的双层极小极大框架：**上层**在一小撮干净元集上
元学习样本权重，**下层**在带噪训练样本上求解 GAN 式极小极大问题。

**亮点**

- 🔬 **算法与论文严格对应** —— 实现了论文中的全部三个算法：
  - **SSGDA**（算法 1）：单步随机梯度下降-上升，步长按外层索引；
  - **TSGDA-1**（算法 2）：内循环 `K` 步交替更新 `y`/`z`；
  - **TSGDA-2**（算法 3）：内循环 `K` 步 `y` 更新 + `Q` 步 `z` 更新。
- 🚫 **纯一阶算法** —— 不使用超梯度 / 展开的伪网络。借鉴一阶双层优化的工程设计，
  上层采用对偶平均动量并配合梯度裁剪，支持多种步长衰减调度
  （`fixed` / `exp` / `inv` / `inv_sqrt`）。
- 📐 **与理论一致的步长索引** —— 条件 `γ^k ≤ c/((k+1)L)` 在 TSGDA-1/2 中按*内层*索引计
  （每个外层步重置），在 SSGDA 中按*外层*步计。
- 📊 **泛化指标** —— 全程记录加权与 *raw*（排除元网络影响）的训练/验证/测试误差，
  以及泛化差距 `GAP = test − val`。
- 🧪 **仿真协议** —— 数据由自然帧加性噪声生成，元集/测试集规模有限，与论文仿真设置一致。

## 🏗️ 项目结构

```text
BMO/
├── bmo_core/
│   ├── data_generator.py    # 仿真数据：带噪副本、元集/测试集增广
│   ├── networks.py          # ConvGenerator / ConvDiscriminator / WeightNet（1-10-10-1）
│   ├── bmo_task.py          # 双层目标函数、评估、指标历史
│   └── solvers.py           # SSGDA / TSGDA-1 / TSGDA-2（一阶）
├── train_test_BMO.py        # 三个求解器的单次训练入口
├── run_experiment.py        # 论文实验 A（图 1/4）与 B（图 2/3）
├── run_expA_split.py        # 实验 A 的单次 (m1, seed) 运行驱动
├── plot_expA.py             # 汇总实验 A 多次运行并绘图
├── tests/test_bmo.py        # 17 个单元测试（更新规则、K=1 等价性、收敛性）
├── results/                 # 仿真实验的 CSV + PNG 输出
└── Create_real_data/        # Chaplin 帧数据管线
```

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/zxlml/BMO-main.git BMO
cd BMO
pip install -r requirements.txt
```

### 2. 数据

仿真数据基于自然视频帧（卓别林影片帧）。请从
[Chaplin's Frames (MFCGAN)](https://github.com/zhigang-yao/MFCGAN)
下载并放置于 `Create_real_data/Chaplin/frames/` 目录下。

### 3. 训练

```bash
# TSGDA-1（默认），实验 A 配置
python train_test_BMO.py --solver TSGDA-1 --T 5 --K 300 --img_size 32 \
    --width_g 64 --width_d 4 --gamma1 0.003 --gamma_schedule fixed \
    --eta 0.005 --eta_schedule exp --eta_decay 0.95

# SSGDA
python train_test_BMO.py --solver SSGDA --T 600 --gamma1 0.01

# TSGDA-2
python train_test_BMO.py --solver TSGDA-2 --T 5 --K 100 --Q 50
```

### 4. 单元测试

```bash
python -m pytest tests/ -q   # 17 通过
```

## 📊 实验

复现论文的仿真研究（结果输出至 `results/` 目录）：

```bash
# 实验 A（图 1/4）：TSGDA-1，GAP 与误差随内层迭代 k 的变化（不同元集大小 m1）
python run_experiment.py --exp A --img_size 32 --T 5 --repeats 2

# 实验 B（图 2/3）：SSGDA，GAP 与误差随外层迭代 T 的变化（不同步长调度）
python run_experiment.py --exp B --img_size 32 --T 600 --gamma1 0.01

# 实验 A 的大规模运行可拆分为单次 (m1, seed) 任务：
python run_expA_split.py --m1 500 --seed 100
python plot_expA.py
```

### 仿真数据上的关键结论

| 现象 | 结果 | 论文趋势 |
|---|---|---|
| GAP 随内层迭代 `k` 增长（m1=500） | 单调增长，+0.012 → +0.023（raw +0.031 → +0.069） | ✅ 图 1 |
| 元集越大 ⇒ GAP 越小 | m1=1000：GAP +0.002 → +0.005（远小于 m1=500） | ✅ O(T/m1) 界 |
| 验证误差平台 | 对抗平衡区内为 0.2–0.3 | ✅ 图 3a |
| 固定 η 不稳定 | 训练误差方差最大（0.34 ± 0.19），收敛快但波动剧烈 | ✅ 图 3b |
| 衰减 η 达到论文误差区间 | val 0.206 / test 0.182（γ=0.01，T=600） | ✅ 图 3 |

> **说明：** **SSGDA** 的系统性 GAP 增长（图 3c）需要论文级规模（160×160 图像、
> T=50–200）；在小规模 CPU 运行中下层学习过慢、无法停留在平衡区。**TSGDA-1** 在可行
> 规模下即可复现 GAP 增长。

**推荐的平衡区配置：** `--width_g 64 --width_d 4 --noise_std 0.1
--gamma1 0.003 --gamma_schedule fixed --eta 0.005 --eta_schedule exp --eta_decay 0.95`

## 📖 参考文献

如果您觉得本仓库对您有帮助，也请考虑引用本实现所基于的以下工作。

<details open>
<summary><b>文献列表</b></summary>

> [1] Meta-Weight-Net: Learning an Explicit Mapping for Sample Weighting (NeurIPS 2019)

```bibtex
@article{shu2019meta,
  title={Meta-weight-net: Learning an explicit mapping for sample weighting},
  author={Shu, Jun and Xie, Qi and Yi, Lixuan and Zhao, Qian and Zhou, Sanping and Xu, Zongben and Meng, Deyu},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
```

> [2] Manifold Fitting with CycleGAN (PNAS 2024)

```bibtex
@article{yao2024manifold,
  title={Manifold fitting with CycleGAN},
  author={Yao, Zhigang and Su, Jiaji and Yau, Shing-Tung},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  number={5},
  pages={e2311436121},
  year={2024},
  publisher={National Academy of Sciences}
}
```

</details>

## 🙏 致谢

- [First_Order_BMO](https://github.com/) —— 一阶双层优化工程设计（对偶平均、梯度裁剪）；
- [SiPBA](https://github.com/qichaosustech/SiPBA) —— 一阶双层求解器参考实现；
- [Meta-Weight-Net](https://github.com/xjtushujun/Meta-weight-net) —— 元样本加权框架；
- [MFCGAN](https://github.com/zhigang-yao/MFCGAN) —— Chaplin 帧数据来源。

## 📄 许可证

本项目基于 [MIT License](LICENSE) 发布。
