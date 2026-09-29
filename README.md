<div align="center">

English | [简体中文](README_zh.md)

# Fine-Grained Stability and Generalization Analysis of Bilevel Minimax Optimization: From First-Order Solvers to Hypergradient-Corrected Momentum Methods

This repository implements and validates three families of bilevel minimax optimization (BMO)
algorithms for robust data processing (denoising / generation): the **first-order baselines**
from the paper, a **first-order improved hypergradient-aided solver (HSGDA)**, and a
**second/higher-order stochastic momentum solver (MS-BMO)**. Each family comes with a
one-click run script, its own experiments, and tests.

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2604.20115-red.svg)](https://arxiv.org/abs/2604.20115)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4%2B-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-17%20passed-brightgreen.svg)](tests/test_bmo.py)

</div>

## 📰 News

- **[2026-09]** Released three algorithm families with one-click scripts:
  first-order baselines (`run_first_order.sh`), first-order improved **HSGDA** (`run_hsgda.sh`),
  and second/higher-order **MS-BMO** (`run_msbmo.sh`); a master `run.sh` dispatches all of them.
- **[2026-09]** Reimplemented all three first-order solvers (**SSGDA**, **TSGDA-1**, **TSGDA-2**)
  as a clean `bmo_core` package with unit tests (17/17 passed); released simulation experiments
  that reproduce the trends of Figures 1–4 of the paper.

## ✨ Overview

We study a bilevel minimax framework for robust data (denoising / generation): the
**upper level** meta-learns instance weights on a small clean meta set, and the
**lower level** solves a GAN-style minimax problem on noisy training samples.

```text
min_x  E_xi f(x, y*(x), z*(x); xi)
s.t.   (y*(x), z*(x)) ∈ argmin_y argmax_z E_zeta g(x, y, z; zeta)
```

with `x` = upper-level weight network, `y` = generator `G` (lower-level minimization),
`z` = discriminator `D` (lower-level maximization).

## 🧮 Algorithms

| Family | Algorithm | Implementation | Key mechanism | Entry |
|---|---|---|---|---|
| First-order baseline | **SSGDA** (Alg. 1) | `bmo_core/solvers.py` | single-timescale, one alternating y/z step per outer step, outer-indexed step sizes | `run_first_order.sh` |
| First-order baseline | **TSGDA-1** (Alg. 2) | `bmo_core/solvers.py` | inner loop of `K` alternating y/z steps, inner-indexed step sizes | `run_first_order.sh` |
| First-order baseline | **TSGDA-2** (Alg. 3) | `bmo_core/solvers.py` | inner loop of `K` y-steps followed by `Q` z-steps | `run_first_order.sh` |
| First-order improved | **HSGDA** | `hsgda/hsgda_core.py` | hypergradient-aided SSGDA: saddle-point and adjoint tracking with provable contraction on an analytically verifiable synthetic problem | `run_hsgda.sh` |
| Second/higher-order improved | **MS-BMO** | `bmo_core/solver_ms.py` | lower-level STORM same-sample corrected momentum; saddle-point inverse estimation via a fresh Hessian oracle with damped CG (`M^{-1}P`); upper-level full hypergradient oracle with STORM correction; single-loop, single-timescale | `run_msbmo.sh` |

**Engineering notes shared by the first-order baselines**

- 🚫 **Purely first-order** — no hypergradient / unrolled pseudo-network; the upper level uses
  dual-averaging momentum with gradient clipping, and supports decaying step-size schedules
  (`fixed` / `exp` / `inv` / `inv_sqrt`).
- 📐 **Theory-consistent step-size indexing** — the condition `γ^k ≤ c/((k+1)L)` is indexed by
  the *inner* loop position (reset at every outer step) for TSGDA-1/2, and by the *outer* step
  for SSGDA.
- 📊 **Generalization metrics** — weighted and *raw* (weight-net-excluded) train/validation/test
  errors, plus the generalization gap `GAP = test − val`, logged throughout training.
- 🧪 **Simulation protocol** — data generated from natural frames with additive noise and
  finite meta/test sets, matching the paper's simulation setup.

## 🏗️ Project Structure

```text
ref_code/
├── bmo_core/
│   ├── data_generator.py    # simulation data: noisy copies, meta/test augmentation
│   ├── networks.py          # ConvGenerator / ConvDiscriminator / WeightNet (1-10-10-1)
│   ├── bmo_task.py          # bilevel objective, evaluation, metric history
│   ├── solvers.py           # [Family 1] SSGDA / TSGDA-1 / TSGDA-2 (first-order baselines)
│   └── solver_ms.py         # [Family 3] MS-BMO (second-order momentum + Hessian/CG)
├── hsgda/
│   ├── hsgda_core.py        # [Family 2] HSGDA + baselines on an analytic synthetic BMO problem (numpy)
│   ├── run_hsgda_experiments.py  # E1–E4: convergence, K, m1, step-size schedules
│   ├── plot_hsgda_figures.py     # aggregate figures (PDF/PNG)
│   └── smoke_test.py        # analytic correctness + convergence smoke test
├── train_test_BMO.py        # single-run training entry for the three first-order solvers
├── run_experiment.py        # paper experiments A (Fig. 1/4) & B (Fig. 2/3)
├── run_expA_split.py        # single (m1, seed) driver for Experiment A
├── plot_expA.py             # aggregate Experiment A runs into curves
├── run_ms_experiments.py    # MS-BMO experiments: conv / m1 / T / lr / sched
├── test_msbmo_smoke.py      # MS-BMO smoke test (toy manifold data)
├── tests/test_bmo.py        # 17 unit tests (update rules, K=1 equivalence, convergence)
├── run.sh                   # master script: deps + all three families
├── run_first_order.sh       # [Family 1] one-click script
├── run_hsgda.sh             # [Family 2] one-click script
├── run_msbmo.sh             # [Family 3] one-click script
├── results/                 # CSV + PNG outputs of the simulation experiments
└── Create_real_data/        # Chaplin frames data pipeline
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/zxlml/BMO-main.git BMO
cd BMO/ref_code
pip install -r requirements.txt
```

### 2. Data

The simulation data is built from natural video frames (Charlie Chaplin's frames).
Download them from [Chaplin's Frames (MFCGAN)](https://github.com/zhigang-yao/MFCGAN)
and place them under `Create_real_data/Chaplin/frames/`. The HSGDA experiments are
purely synthetic (numpy only) and need no external data.

### 3. One-click runs

Every family has its own script; each accepts a mode argument
(`full` by default, or `quick` for a small-scale smoke run):

```bash
# Master script: install deps, then run all three families (full or quick)
bash run.sh                 # full
bash run.sh quick           # quick smoke of everything

# Or run a single family
bash run.sh full first_order
bash run.sh full hsgda
bash run.sh full msbmo

# Or run one family's script directly
bash run_first_order.sh           # SSGDA / TSGDA-1 / TSGDA-2: tests + training + Exp A & B
bash run_hsgda.sh                 # HSGDA: smoke test + E1–E4 + plotting
bash run_msbmo.sh                 # MS-BMO: smoke test + conv/m1/T/lr/sched
bash run_hsgda.sh quick           # small-scale version of any script
```

### 4. Manual training (first-order solvers)

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

### 5. Unit Tests

```bash
python -m pytest tests/ -q   # 17 passed
```

## 📊 Experiments

All outputs (CSV + PNG) are written to `results/` (first-order & MS-BMO) and
`hsgda/results/` (HSGDA).

**Family 1 — first-order baselines** (`run_experiment.py`):

```bash
# Experiment A (Fig. 1/4): TSGDA-1, GAP & errors vs. inner iterations k, per meta size m1
python run_experiment.py --exp A --img_size 32 --T 5 --repeats 2

# Experiment B (Fig. 2/3): SSGDA, GAP & errors vs. outer iterations T, per step-size schedule
python run_experiment.py --exp B --img_size 32 --T 600 --gamma1 0.01

# Large-scale runs of Experiment A can be split into single (m1, seed) jobs:
python run_expA_split.py --m1 500 --seed 100
python plot_expA.py
```

**Family 2 — HSGDA** (`hsgda/run_hsgda_experiments.py`):

```bash
python hsgda/run_hsgda_experiments.py --exp probe   # quick tuning probe
python hsgda/run_hsgda_experiments.py --exp conv    # E1: convergence vs SSGDA/TSGDA-1
python hsgda/run_hsgda_experiments.py --exp tk      # E2: inner K & outer T (Fig. 1 protocol)
python hsgda/run_hsgda_experiments.py --exp m1      # E3: meta size m1 vs 1/m1 gap (Fig. 2)
python hsgda/run_hsgda_experiments.py --exp eta     # E4: step size & schedules (Fig. 3/4)
python hsgda/run_hsgda_experiments.py --exp all     # E1–E4
python hsgda/plot_hsgda_figures.py                  # aggregate figures
```

**Family 3 — MS-BMO** (`run_ms_experiments.py`):

```bash
python run_ms_experiments.py --exp conv   # convergence: MS-BMO vs SSGDA vs TSGDA-1
python run_ms_experiments.py --exp m1     # meta set size (Fig. 1 protocol)
python run_ms_experiments.py --exp T      # number of outer iterations
python run_ms_experiments.py --exp lr     # step size eta (Fig. 3 protocol)
python run_ms_experiments.py --exp sched  # step-size schedules (Fig. 4 protocol)
python run_ms_experiments.py --exp all    # everything (add --quick for a smoke run)
```

## 📚 References

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

## 🔖 Cite This Paper

If you find this work useful, please cite:

```bibtex
@article{zhang2026stability,
  title={On the Stability and Generalization of First-order Bilevel Minimax Optimization},
  author={Zhang, Xuelin and Yuan, Peipei},
  journal={arXiv preprint arXiv:2604.20115},
  year={2026},
  url={https://arxiv.org/abs/2604.20115}
}
```
