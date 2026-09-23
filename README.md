<div align="center">

English | [简体中文](README_zh.md)

# BMO: First-order Bilevel Minimax Optimization

**Fine-grained Analysis on the Stability and Generalization for First-order Bilevel Minimax Optimization**

This repository provides the official implementation for validating the theoretical
generalization bounds of first-order bilevel minimax algorithms and investigating their
connections to practical settings, e.g., max inner iterations, step-size schedules and
meta sample sizes.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-17%20passed-brightgreen.svg)](tests/test_bmo.py)

</div>

## 📰 News

- **[2026-09]** Reimplemented all three first-order solvers (**SSGDA**, **TSGDA-1**, **TSGDA-2**) as a clean `bmo_core` package with unit tests (17/17 passed); released simulation experiments that reproduce the trends of Figures 1–4 of the paper.

## ✨ Overview

We study a bilevel minimax framework for robust data (denoising / generation): the
**upper level** meta-learns instance weights on a small clean meta set, and the
**lower level** solves a GAN-style minimax problem on noisy training samples.

**Highlights**

- 🔬 **Faithful algorithm–paper matching** — all three algorithms of the paper are implemented:
  - **SSGDA** (Algorithm 1): single-step stochastic gradient descent-ascent, outer-indexed step sizes;
  - **TSGDA-1** (Algorithm 2): inner loop of `K` alternating `y`/`z` steps;
  - **TSGDA-2** (Algorithm 3): inner loop of `K` `y`-steps followed by `Q` `z`-steps.
- 🚫 **Purely first-order** — no hyper-gradient / unrolled pseudo-network. Following the
  engineering design of first-order bilevel implementations, the upper level uses
  dual-averaging momentum with gradient clipping, and supports decaying step-size
  schedules (`fixed` / `exp` / `inv` / `inv_sqrt`).
- 📐 **Theory-consistent step-size indexing** — the condition `γ^k ≤ c/((k+1)L)` is indexed by
  the *inner* loop position (reset at every outer step) for TSGDA-1/2, and by the *outer*
  step for SSGDA.
- 📊 **Generalization metrics** — weighted and *raw* (weight-net-excluded) train/validation/test
  errors, plus the generalization gap `GAP = test − val`, logged throughout training.
- 🧪 **Simulation protocol** — data generated from natural frames with additive noise and
  finite meta/test sets, matching the paper's simulation setup.

## 🏗️ Project Structure

```text
BMO/
├── bmo_core/
│   ├── data_generator.py    # simulation data: noisy copies, meta/test augmentation
│   ├── networks.py          # ConvGenerator / ConvDiscriminator / WeightNet (1-10-10-1)
│   ├── bmo_task.py          # bilevel objective, evaluation, metric history
│   └── solvers.py           # SSGDA / TSGDA-1 / TSGDA-2 (first-order)
├── train_test_BMO.py        # single-run training entry for all three solvers
├── run_experiment.py        # paper experiments A (Fig. 1/4) & B (Fig. 2/3)
├── run_expA_split.py        # single (m1, seed) driver for Experiment A
├── plot_expA.py             # aggregate Experiment A runs into curves
├── tests/test_bmo.py        # 17 unit tests (update rules, K=1 equivalence, convergence)
├── results/                 # CSV + PNG outputs of the simulation experiments
└── Create_real_data/        # Chaplin frames data pipeline
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/zxlml/BMO-main.git BMO
cd BMO
pip install -r requirements.txt
```

### 2. Data

The simulation data is built from natural video frames (Charlie Chaplin's frames).
Download them from [Chaplin's Frames (MFCGAN)](https://github.com/zhigang-yao/MFCGAN)
and place them under `Create_real_data/Chaplin/frames/`.

### 3. Training

```bash
# TSGDA-1 (default), Experiment-A configuration
python train_test_BMO.py --solver TSGDA-1 --T 5 --K 300 --img_size 32 \
    --width_g 64 --width_d 4 --gamma1 0.003 --gamma_schedule fixed \
    --eta 0.005 --eta_schedule exp --eta_decay 0.95

# SSGDA
python train_test_BMO.py --solver SSGDA --T 600 --gamma1 0.01

# TSGDA-2
python train_test_BMO.py --solver TSGDA-2 --T 5 --K 100 --Q 50
```

### 4. Unit Tests

```bash
python -m pytest tests/ -q   # 17 passed
```

## 📊 Experiments

Reproduce the paper's simulation studies (results are written to `results/`):

```bash
# Experiment A (Fig. 1/4): TSGDA-1, GAP & errors vs. inner iterations k, per meta size m1
python run_experiment.py --exp A --img_size 32 --T 5 --repeats 2

# Experiment B (Fig. 2/3): SSGDA, GAP & errors vs. outer iterations T, per step-size schedule
python run_experiment.py --exp B --img_size 32 --T 600 --gamma1 0.01

# Large-scale runs of Experiment A can be split into single (m1, seed) jobs:
python run_expA_split.py --m1 500 --seed 100
python plot_expA.py
```

### Key Findings on Simulation Data

| Observation | Result | Paper trend |
|---|---|---|
| GAP vs. inner iterations `k` (m1=500) | monotonically grows, +0.012 → +0.023 (raw +0.031 → +0.069) | ✅ Fig. 1 |
| Larger meta set ⇒ smaller GAP | m1=1000: GAP +0.002 → +0.005 (≪ m1=500) | ✅ O(T/m1) bound |
| Validation error plateau | 0.2–0.3 in the adversarial balance region | ✅ Fig. 3a |
| Fixed η instability | largest train-error variance (0.34 ± 0.19), fast but erratic convergence | ✅ Fig. 3b |
| Decaying η reaches the paper's error range | val 0.206 / test 0.182 (γ=0.01, T=600) | ✅ Fig. 3 |

> **Note:** the systematic GAP growth of **SSGDA** (Fig. 3c) requires the paper-scale setup
> (160×160 images, T=50–200); on small-scale CPU runs the lower level learns too slowly to
> remain in the balance region. **TSGDA-1** reproduces GAP growth already at feasible scale.

**Recommended balance-region configuration:** `--width_g 64 --width_d 4 --noise_std 0.1
--gamma1 0.003 --gamma_schedule fixed --eta 0.005 --eta_schedule exp --eta_decay 0.95`

## 📖 Reference

If you find this repository useful, please also consider citing the works below, which this
implementation builds upon.

<details open>
<summary><b>Bibliography</b></summary>

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

## 🙏 Acknowledgements

- [First_Order_BMO](https://github.com/) — first-order bilevel engineering design (dual averaging, gradient clipping);
- [SiPBA](https://github.com/qichaosustech/SiPBA) — reference for first-order bilevel solvers;
- [Meta-Weight-Net](https://github.com/xjtushujun/Meta-weight-net) — meta instance-weighting framework;
- [MFCGAN](https://github.com/zhigang-yao/MFCGAN) — Chaplin frames data source.

## 📄 License

This project is released under the [MIT License](LICENSE).
