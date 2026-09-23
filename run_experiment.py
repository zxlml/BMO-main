# -*- coding: utf-8 -*-
"""论文实验复现入口（在仿真数据上验证 SSGDA / TSGDA-1 / TSGDA-2）。

实验A（论文图1）：TSGDA-1，固定 T，变化内层迭代 K 与元集大小 m1，
    记录验证误差（元集）、测试误差与泛化 GAP 随 K 的变化。
实验B（论文图2/3）：SSGDA（K=1），变化外层迭代 T 与步长调度
    fixed / decaying(0.95) / faster-decaying(0.85)，
    记录 GAP 与验证误差随 T 的变化。

用法：
    python run_experiment.py --exp A --img_size 32 --T 20 --repeats 2   # 快速
    python run_experiment.py --exp B --img_size 32 --T 200 --repeats 1
    python run_experiment.py --exp A --paper_scale                      # 论文规模
结果输出到 results/ 目录（CSV + PNG）。
"""
import os
import sys
import argparse
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bmo_core import (generate_simulation_data, make_dataloaders,
                      WeightNet, ConvGenerator, ConvDiscriminator,
                      BMOTask, SOLVERS)

FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'Create_real_data', 'Chaplin', 'frames')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def build_task(args, device):
    netG = ConvGenerator(latent_dim=args.latent_dim, img_size=args.img_size,
                         width=args.width_g).to(device)
    netD = ConvDiscriminator(img_size=args.img_size, width=args.width_d).to(device)
    weight_net = WeightNet(hidden_size=10, num_layers=2).to(device)  # 论文附录A: 1-10-10-1
    # fake_noise: 假样本叠加噪声幅度；默认与数据噪声一致，设为 0 可避免
    # "噪声纹理"成为假样本线索（保证噪声不破坏内容区分）。
    fake_noise = getattr(args, 'fake_noise', None)
    task = BMOTask(netG, netD, weight_net,
                   noise_std=args.noise_std if fake_noise is None else fake_noise,
                   latent_dim=args.latent_dim, device=device)
    return task


def init_nets(task, seed):
    """每次重复实验前重置网络权重（独立重复实验）。"""
    torch.manual_seed(seed)
    for m in (task.netG, task.netD, task.weight_net):
        for p in m.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
            else:
                torch.nn.init.zeros_(p)


def run_once(args, device, seed, K, m1, eta_mode, eta_decay):
    """跑一次完整训练，返回历史记录 DataFrame。"""
    data = generate_simulation_data(FRAMES_DIR, img_size=args.img_size,
                                    num_copies=args.num_copies,
                                    noise_std=args.noise_std,
                                    m1=m1, seed=seed)
    loaders = make_dataloaders(data, batch_size=args.batch_size, device=device)
    task = build_task(args, device)
    init_nets(task, seed)

    solver_cls = SOLVERS[args.solver]
    solver = solver_cls(task,
                        eta=args.eta, gamma1=args.gamma1, gamma2=args.gamma2,
                        eta_schedule=eta_mode, eta_decay=eta_decay,
                        gamma_schedule=args.gamma_schedule, gamma_decay=args.gamma_decay,
                        beta_momentum=args.beta, grad_clip=args.grad_clip, seed=seed)

    if args.solver == 'SSGDA':
        hist = solver.run(loaders['train'], loaders['meta'], loaders['test'],
                          T=args.T, eval_every=args.eval_every)
    elif args.solver == 'TSGDA-1':
        hist = solver.run(loaders['train'], loaders['meta'], loaders['test'],
                          T=args.T, K=K, eval_ks=args.eval_ks)
    else:
        hist = solver.run(loaders['train'], loaders['meta'], loaders['test'],
                          T=args.T, K=K, Q=K, eval_ks=args.eval_ks)
    df = pd.DataFrame(hist.records)
    df['seed'] = seed
    df['K'] = K
    df['m1'] = m1
    df['eta_mode'] = eta_mode
    return df


def finalize_record(df, args, K, m1, eta_mode):
    """取该次运行最后一个评估点作为最终误差。"""
    row = df.iloc[-1]
    return dict(solver=args.solver, K=K, m1=m1, eta_mode=eta_mode,
                train_error=row['train_error'], val_error=row['val_error'],
                test_error=row['test_error'], gap=row['gap'], seed=row['seed'])


def summarize(df_detail, keys):
    agg = df_detail.groupby(keys)[['train_error', 'val_error', 'test_error', 'gap']]
    return agg.agg(['mean', 'std']).reset_index()


def save_plots(summary, out_prefix, xkey, title):
    metrics = [('val_error', 'Errors in Meta Set'),
               ('test_error', 'Errors in Test Set'),
               ('gap', 'Generalization Gap')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (metric, label) in zip(axes, metrics):
        for (key_vals, sub) in summary.groupby([k for k in keys if k != xkey]):
            sub = sub.sort_values(xkey)
            mean = sub[(metric, 'mean')]
            std = sub[(metric, 'std')].fillna(0)
            label_txt = '/'.join(str(v) for v in (key_vals if isinstance(key_vals, tuple) else (key_vals,)))
            ax.errorbar(sub[xkey], mean, yerr=std, marker='o', capsize=3, label=label_txt)
        ax.set_xlabel(xkey)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_prefix + '.png', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', choices=['A', 'B'], default='A')
    p.add_argument('--solver', default=None, help='覆盖默认求解器')
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--latent_dim', type=int, default=16)
    p.add_argument('--width_g', type=int, default=64, help='生成器通道宽度')
    p.add_argument('--width_d', type=int, default=16, help='判别器通道宽度')
    p.add_argument('--noise_std', type=float, default=0.1)
    p.add_argument('--fake_noise', type=float, default=None,
                   help='假样本噪声幅度；默认与 --noise_std 一致，0 表示假样本不加噪')
    p.add_argument('--num_copies', type=int, default=20)
    p.add_argument('--m1', type=int, default=None,
                   help='元集大小（实验B 默认 None=17 张自然帧）')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--T', type=int, default=20, help='外层迭代数（快速模式）')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--eval_every', type=int, default=10)
    p.add_argument('--eval_ks', type=int, nargs='+', default=[10, 100, 150, 200, 250, 300])
    p.add_argument('--eta', type=float, default=1e-3)
    p.add_argument('--gamma1', type=float, default=1e-3)
    p.add_argument('--gamma2', type=float, default=1e-3)
    p.add_argument('--gamma_schedule', default='exp')
    p.add_argument('--gamma_decay', type=float, default=0.95)
    p.add_argument('--schedules', nargs='+', default=None,
                   help='实验B 仅运行指定调度（Fixed/Decaying/Faster Decaying）')
    p.add_argument('--beta', type=float, default=0.9)
    p.add_argument('--grad_clip', type=float, default=10.0)
    p.add_argument('--paper_scale', action='store_true', help='按论文规模运行（耗时长）')
    args = p.parse_args()

    if args.paper_scale:
        args.img_size = 160
        args.batch_size = 64
        args.T = 50 if args.exp == 'A' else 200
        args.repeats = 10
        args.eta = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    print(f'设备: {device} | 实验: {args.exp} | 求解器: {args.solver}')

    if args.exp == 'A':
        # 论文图1/4: TSGDA-1, 固定 T, 记录内层 k ∈ eval_ks 处的指标并对
        # 外层步与重复实验取平均；x 轴为内层迭代 k，每条曲线对应一个 m1。
        args.solver = args.solver or 'TSGDA-1'
        Kmax = args.eval_ks[-1] if args.eval_ks else 300
        m1s = [1000, 1500, 2000] if args.paper_scale else [100, 500, 1000]
        details = []
        for m1 in m1s:
            for r in range(args.repeats):
                seed = 100 + r
                df = run_once(args, device, seed, K=Kmax, m1=m1,
                              eta_mode='exp', eta_decay=0.95)
                df['m1'] = m1
                details.append(df)
                last = df.iloc[-1]
                print(f'[A] m1={m1} seed={seed} '
                      f'val={last["val_error"]:.4f} test={last["test_error"]:.4f} '
                      f'gap={last["gap"]:.4f} ({time.time()-t0:.0f}s)')
        detail = pd.concat(details, ignore_index=True)
        detail.to_csv(os.path.join(RESULTS_DIR, 'expA_detail.csv'), index=False)

        # 按 (m1, step) 聚合（step = 循环内 k），画 val/test/gap 随 k 的曲线
        summary = detail.groupby(['m1', 'step'])[
            ['val_error', 'test_error', 'gap', 'gap_raw']].agg(['mean', 'std']).reset_index()
        summary.to_csv(os.path.join(RESULTS_DIR, 'expA_summary.csv'), index=False)
        metrics = [('val_error', 'Errors in Meta Set'),
                   ('test_error', 'Errors in Test Set'),
                   ('gap', 'Generalization Gap'),
                   ('gap_raw', 'Generalization Gap (raw)')]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        for ax, (metric, label) in zip(np.atleast_1d(axes), metrics):
            for m1 in m1s:
                sub = summary[summary['m1'] == m1].sort_values('step')
                ax.errorbar(sub['step'], sub[(metric, 'mean')],
                            yerr=sub[(metric, 'std')].fillna(0), marker='o',
                            capsize=3, label=f'm1={m1}')
            ax.set_xlabel('Inner Iteration K')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.suptitle(f'Exp A: {args.solver} (T={args.T})')
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'expA.png'), dpi=150)
        plt.close(fig)
        print(summary)

    else:
        # 论文图2/3: SSGDA (K=1), 变化 T 与步长调度
        args.solver = args.solver or 'SSGDA'
        all_schedules = [('fixed', 0.95, 'Fixed'),
                         ('exp', 0.95, 'Decaying'),
                         ('exp', 0.85, 'Faster Decaying')]
        wanted = getattr(args, 'schedules', None)
        schedules = [s for s in all_schedules
                     if wanted is None or s[2] in wanted]
        records, details = [], []
        for mode, decay, name in schedules:
            for r in range(args.repeats):
                seed = 100 + r
                df = run_once(args, device, seed, K=1, m1=args.m1,
                              eta_mode=mode, eta_decay=decay)
                df['eta_name'] = name
                details.append(df)
                rec = finalize_record(df, args, 1, None, name)
                rec['eta_name'] = name
                records.append(rec)
                print(f'[B] eta={name} seed={seed} '
                      f'val={rec["val_error"]:.4f} test={rec["test_error"]:.4f} '
                      f'gap={rec["gap"]:.4f} ({time.time()-t0:.0f}s)')
        detail = pd.concat(details, ignore_index=True)
        rec_df = pd.DataFrame(records)
        summary = summarize(rec_df, ['eta_name'])
        detail.to_csv(os.path.join(RESULTS_DIR, 'expB_detail.csv'), index=False)
        summary.to_csv(os.path.join(RESULTS_DIR, 'expB_summary.csv'), index=False)

        # 曲线图：gap 与 val_error 随 T 变化
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, label in zip(
                axes, ['gap', 'val_error'],
                ['Generalization Gap', 'Errors in Meta Set']):
            for name in [s[2] for s in schedules]:
                subs = [d for d, rec in zip(details, records)
                        if rec['eta_name'] == name]
                if not subs:
                    continue
                min_len = min(len(s) for s in subs)
                stack = np.stack([s[metric].values[:min_len] for s in subs])
                mean, std = stack.mean(0), stack.std(0)
                steps = subs[0]['step'].values[:min_len]
                axes_x = steps if args.solver == 'SSGDA' else np.arange(min_len)
                ax.plot(axes_x, mean, label=name)
                ax.fill_between(axes_x, mean - std, mean + std, alpha=0.2)
            ax.set_xlabel('Outer Iteration T' if args.solver == 'SSGDA' else 'Inner step k')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'expB_curves.png'), dpi=150)
        plt.close(fig)
        print(summary)

    print(f'完成，用时 {time.time()-t0:.0f}s，结果已保存到 {RESULTS_DIR}')


if __name__ == '__main__':
    main()
