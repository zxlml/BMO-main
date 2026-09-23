# -*- coding: utf-8 -*-
"""诊断脚本：核对论文图1 的过拟合趋势（val 下降、test 上升、gap 增长）。

机制假设：
  val = 奇数帧干净版（内容出现在带噪训练集中），test = 偶数帧干净版（内容未出现）。
  当 G 足够强（D 无法"全部判真"）且 gamma 固定不衰减（下层持续学习）时，
  D 会记忆训练帧的特异内容 -> val 误差下降、test 误差上升 -> GAP 随迭代增长。
"""
import os
import sys
import argparse
import time
import itertools

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from bmo_core import generate_simulation_data, make_dataloaders, TSGDA1
from run_experiment import build_task, init_nets, FRAMES_DIR


def run_config(device, width_g, width_d, gamma, eta, T, K, img_size, seed=100,
               noise_std=0.1, gamma_schedule='fixed', gamma_decay=1.0, m1=100,
               fake_noise=None):
    """fake_noise: 生成假样本时叠加的噪声幅度；None 时与数据噪声一致，
    设为 0 则假样本干净（噪声不成为假样本线索，迫使 D 依赖内容记忆化）。"""
    args = argparse.Namespace(img_size=img_size, latent_dim=16, width_g=width_g,
                              width_d=width_d,
                              noise_std=noise_std if fake_noise is None else fake_noise)
    data = generate_simulation_data(FRAMES_DIR, img_size=img_size, num_copies=20,
                                    noise_std=noise_std, m1=m1, seed=seed)
    loaders = make_dataloaders(data, batch_size=64, device=device)
    task = build_task(args, device)
    init_nets(task, seed)
    s = TSGDA1(task, eta=eta, gamma1=gamma, gamma2=gamma,
               eta_schedule='exp', eta_decay=0.95,
               gamma_schedule=gamma_schedule, gamma_decay=gamma_decay,
               beta_momentum=0.9, grad_clip=10.0, seed=seed)
    t0 = time.time()
    hist = s.run(loaders['train'], loaders['meta'], loaders['test'],
                 T=T, K=K, eval_ks=(10, 100, 200, K))
    # raw（未加权）指标已由 evaluate 记录到 history
    h = hist.records
    print(f'-- width_g={width_g} width_d={width_d} gamma={gamma} eta={eta} '
          f'img={img_size} sigma={noise_std} gsched={gamma_schedule} m1={m1} '
          f'T={T} K={K} ({time.time()-t0:.0f}s)')
    print('  step |  train |   val_w |  test_w |  gap_w |  val_raw | test_raw | gap_raw')
    for i in range(len(h['step'])):
        print(f"  {h['step'][i]:4d} | {h['train_error'][i]:.4f} | "
              f"{h['val_error'][i]:.4f} | {h['test_error'][i]:.4f} | "
              f"{h['gap'][i]:+.4f} | {h['val_error_raw'][i]:.4f} | "
              f"{h['test_error_raw'][i]:.4f} | {h['gap_raw'][i]:+.4f}")
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--T', type=int, default=6)
    p.add_argument('--K', type=int, default=300)
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--width_g', type=int, default=32)
    p.add_argument('--width_d', type=int, default=16)
    p.add_argument('--gamma', type=float, default=0.003)
    p.add_argument('--eta', type=float, default=0.005)
    p.add_argument('--noise_std', type=float, default=0.1)
    p.add_argument('--m1', type=int, default=100)
    p.add_argument('--fake_noise', type=float, default=None)
    p.add_argument('--gamma_schedule', default='fixed')
    p.add_argument('--gamma_decay', type=float, default=1.0)
    p.add_argument('--grid', action='store_true', help='运行预设网格')
    args = p.parse_args()
    device = 'cpu'

    if args.grid:
        grid = [
            # (width_g, width_d, gamma, eta)
            (32, 16, 0.003, 0.005),
            (64, 16, 0.003, 0.005),
            (64, 16, 0.005, 0.005),
        ]
        for (wg, wd, g, e) in grid:
            run_config(device, wg, wd, g, e, args.T, args.K, args.img_size)
    else:
        run_config(device, args.width_g, args.width_d, args.gamma, args.eta,
                   args.T, args.K, args.img_size, noise_std=args.noise_std,
                   gamma_schedule=args.gamma_schedule,
                   gamma_decay=args.gamma_decay, m1=args.m1,
                   fake_noise=args.fake_noise)


if __name__ == '__main__':
    main()
