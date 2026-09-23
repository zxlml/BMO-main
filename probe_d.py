# -*- coding: utf-8 -*-
"""探针脚本：训练后直接测量 D 对各数据子集的平均响应，揭示决策结构。"""
import os
import sys
import argparse

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from bmo_core import generate_simulation_data, make_dataloaders, TSGDA1
from run_experiment import build_task, init_nets, FRAMES_DIR


@torch.no_grad()
def probe(task, name, imgs):
    """imgs: (N,1,H,W) 张量，返回 D 输出的 sigmoid 均值（越接近1越'真'）。"""
    task.netD.eval()
    preds = []
    for i in range(0, len(imgs), 64):
        x = imgs[i:i + 64].to(task.device)
        preds.append(torch.sigmoid(task.netD(x)).flatten().cpu())
    p = torch.cat(preds)
    task.netD.train()
    print(f'  {name:28s} n={len(imgs):4d}  D(x)均值={p.mean().item():.4f}  '
          f'中位={p.median().item():.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--T', type=int, default=1)
    p.add_argument('--K', type=int, default=300)
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--width_g', type=int, default=32)
    p.add_argument('--width_d', type=int, default=8)
    p.add_argument('--gamma', type=float, default=0.002)
    p.add_argument('--eta', type=float, default=0.005)
    p.add_argument('--noise_std', type=float, default=0.15)
    p.add_argument('--fake_noise', type=float, default=0.0)
    p.add_argument('--m1', type=int, default=200)
    args_cli = p.parse_args()

    seed = 100
    data = generate_simulation_data(FRAMES_DIR, img_size=args_cli.img_size,
                                    num_copies=20, noise_std=args_cli.noise_std,
                                    m1=args_cli.m1, seed=seed)
    loaders = make_dataloaders(data, batch_size=64, device='cpu')
    args = argparse.Namespace(img_size=args_cli.img_size, latent_dim=16,
                              width_g=args_cli.width_g, width_d=args_cli.width_d,
                              noise_std=args_cli.fake_noise)
    task = build_task(args, 'cpu')
    init_nets(task, seed)
    s = TSGDA1(task, eta=args_cli.eta, gamma1=args_cli.gamma,
               gamma2=args_cli.gamma, eta_schedule='exp', eta_decay=0.95,
               gamma_schedule='fixed', gamma_decay=1.0,
               beta_momentum=0.9, grad_clip=10.0, seed=seed)
    s.run(loaders['train'], loaders['meta'], loaders['test'],
          T=args_cli.T, K=args_cli.K, eval_ks=(args_cli.K,))

    # 重构原始帧（未增广）: train_noisy 按奇数帧×20 排列，meta/test 前17张为自然帧
    n_frames = len(data['meta_clean']) and (data['meta_clean'].shape[0] // args_cli.m1) * 17 or 17
    # 直接从文件重新读自然帧
    from bmo_core.data_generator import _list_frames, _to_gray_array
    from PIL import Image
    files = _list_frames(FRAMES_DIR)
    odd_nat, even_nat = [], []
    for i, f in enumerate(files):
        arr = _to_gray_array(Image.open(f), args_cli.img_size)
        if (i + 1) % 2 == 0:
            even_nat.append(arr)
        else:
            odd_nat.append(arr)
    odd_nat = torch.from_numpy(np.stack(odd_nat)).unsqueeze(1)
    even_nat = torch.from_numpy(np.stack(even_nat)).unsqueeze(1)

    train_noisy = data['train_noisy']
    meta_clean = data['meta_clean']
    test_clean = data['test_clean']
    fake = task.make_fake(256).detach().cpu()

    print(f'\n== 探针结果 (img={args_cli.img_size}, sigma={args_cli.noise_std}, '
          f'fake_noise={args_cli.fake_noise}, T={args_cli.T}, K={args_cli.K}) ==')
    probe(task, '带噪训练图(奇+噪)', train_noisy)
    probe(task, '自然奇数帧(干净,元集内容)', odd_nat)
    probe(task, '自然偶数帧(干净,测试内容)', even_nat)
    probe(task, '增广后元集(val)', meta_clean)
    probe(task, '增广后测试集(test)', test_clean)
    probe(task, '生成假样本 G(eps)', fake)


if __name__ == '__main__':
    main()
