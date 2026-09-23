# -*- coding: utf-8 -*-
"""实验A拆分驱动：单次 (m1, seed) 运行，保存 detail CSV（规避后台超时）。"""
import os
import sys
import argparse

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch

from run_experiment import run_once, RESULTS_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--solver', default='TSGDA-1')
    p.add_argument('--m1', type=int, required=True)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--T', type=int, default=5)
    p.add_argument('--K', type=int, default=300)
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--latent_dim', type=int, default=16)
    p.add_argument('--width_g', type=int, default=64)
    p.add_argument('--width_d', type=int, default=4)
    p.add_argument('--noise_std', type=float, default=0.1)
    p.add_argument('--fake_noise', type=float, default=None)
    p.add_argument('--num_copies', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--eval_ks', type=int, nargs='+',
                   default=[10, 100, 150, 200, 250, 300])
    p.add_argument('--eta', type=float, default=0.005)
    p.add_argument('--gamma1', type=float, default=0.003)
    p.add_argument('--gamma2', type=float, default=0.003)
    p.add_argument('--gamma_schedule', default='fixed')
    p.add_argument('--gamma_decay', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=0.9)
    p.add_argument('--grad_clip', type=float, default=10.0)
    args = p.parse_args()

    device = 'cpu'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = run_once(args, device, args.seed, K=args.K, m1=args.m1,
                  eta_mode='exp', eta_decay=0.95)
    df['m1'] = args.m1
    out = os.path.join(RESULTS_DIR, f'expA_detail_m{args.m1}_s{args.seed}.csv')
    df.to_csv(out, index=False)
    last = df.iloc[-1]
    print(f'[A] m1={args.m1} seed={args.seed} val={last["val_error"]:.4f} '
          f'test={last["test_error"]:.4f} gap={last["gap"]:.4f} -> {out}')


if __name__ == '__main__':
    main()
