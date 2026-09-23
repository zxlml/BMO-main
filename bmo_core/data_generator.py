# -*- coding: utf-8 -*-
"""仿真数据生成模块。

严格复刻仓库 Create_real_data/Chaplin/Generate_data.m 的仿真数据生成方式：
- 读取帧目录下的 PNG 帧，转灰度并缩放到目标尺寸（论文为 160×160）；
- 1-based 偶数序号帧（即 0-based 文件名 frame_01/03/...）作为测试集（干净图）；
- 1-based 奇数序号帧（frame_00/02/...）加入高斯噪声（σ = noise_std）并复制
  num_copies 份作为带噪训练集；
- 奇数帧的干净版本作为元集/验证集；元集与测试集用同一增广管线
  （翻转+随机平移裁剪）扩到 m1，保证两者分布一致。

同时提供一个轻量合成流形数据生成器（圆/正弦曲线流形 + 高斯噪声），
供单元测试与快速冒烟实验使用。
"""
import os
import glob
import math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader


def _list_frames(frames_dir):
    files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
    if not files:
        raise FileNotFoundError(f'未在 {frames_dir} 找到 PNG 帧')
    return files


def _to_gray_array(img, img_size):
    g = img.convert('L')
    if img_size is not None:
        g = g.resize((img_size, img_size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32) / 255.0
    return a


def _augment(arr, rng):
    """轻量增广：随机水平翻转 + 随机平移裁剪（边缘复制填充，避免黑边伪影）。"""
    if rng.random() < 0.5:
        arr = arr[:, ::-1]
    h, w = arr.shape
    pad = max(2, int(0.1 * min(h, w)))
    dy = int(rng.integers(-pad, pad + 1)) if pad > 0 else 0
    dx = int(rng.integers(-pad, pad + 1)) if pad > 0 else 0
    out = np.pad(arr, ((max(0, -dy), max(0, dy)), (max(0, -dx), max(0, dx))),
                 mode='edge')
    dy0, dx0 = max(0, dy), max(0, dx)
    out = out[dy0:dy0 + h, dx0:dx0 + w]
    return np.ascontiguousarray(out, dtype=np.float32)


def generate_simulation_data(frames_dir, img_size=160, num_copies=20,
                             noise_std=0.1, m1=2000, seed=1,
                             as_tensor=True):
    """按 Generate_data.m 生成仿真数据。

    返回 dict:
        train_noisy: (m2, H, W) 带噪训练图（奇数帧 × num_copies 份）
        meta_clean:  (m1, H, W) 干净元集/验证集（奇数帧增广到 m1）
        test_clean:  (n_test, H, W) 干净测试集（偶数帧）
        img_size, noise_std
    """
    rng = np.random.default_rng(seed)
    files = _list_frames(frames_dir)
    train_noisy, meta_clean, test_clean = [], [], []
    for i, f in enumerate(files):          # i 为 0-based；MATLAB 为 1-based
        arr = _to_gray_array(Image.open(f), img_size)
        if (i + 1) % 2 == 0:               # MATLAB 偶数序号 → 测试集
            test_clean.append(arr)
        else:                              # MATLAB 奇数序号 → 带噪训练 + 干净元集
            meta_clean.append(arr)
            for _ in range(num_copies):
                noisy = arr + (noise_std * rng.standard_normal(arr.shape)).astype(np.float32)
                train_noisy.append(np.clip(noisy, 0.0, 1.0))

    train_noisy = np.stack(train_noisy)
    test_clean = np.stack(test_clean)

    # 元集增广到 m1（翻转 + 平移裁剪，重复采样）；
    # 测试集用同一增广管线扩到相同数量，保证 val/test 分布严格一致，
    # 唯一差异 = 内容是否出现在带噪训练集中（奇数帧 vs 偶数帧）。
    base_m = np.stack(meta_clean).astype(np.float32)
    base_t = np.stack(test_clean).astype(np.float32)
    if m1 is not None and m1 > len(base_m):
        def _expand(base):
            extra = []
            while len(base) + len(extra) < m1:
                idx = int(rng.integers(0, len(base)))
                extra.append(_augment(base[idx], rng))
            return np.concatenate([base, np.stack(extra).astype(np.float32)], axis=0)
        meta_clean = _expand(base_m)
        test_clean = _expand(base_t)
    else:
        meta_clean, test_clean = base_m, base_t

    if as_tensor:
        train_noisy = torch.from_numpy(train_noisy).unsqueeze(1)   # (m2,1,H,W)
        meta_clean = torch.from_numpy(np.ascontiguousarray(meta_clean)).unsqueeze(1)
        test_clean = torch.from_numpy(test_clean).unsqueeze(1)
    return dict(train_noisy=train_noisy, meta_clean=meta_clean,
                test_clean=test_clean, img_size=img_size, noise_std=noise_std)


def make_dataloaders(data, batch_size=64, device='cpu'):
    """将仿真数据打包成 训练(带噪)/元(干净)/测试(干净) 三个 DataLoader。

    drop_last 仅在样本数 >= batch_size 时启用，避免小元集产生空 loader；
    batch_size 超过样本数时自动收缩。
    """
    def _ds(t):
        return TensorDataset(t.to(device))

    def _loader(t, shuffle, drop_last):
        bs = min(batch_size, len(t)) or 1
        dl = drop_last and len(t) >= bs
        return DataLoader(_ds(t), batch_size=bs, shuffle=shuffle, drop_last=dl)

    return dict(
        train=_loader(data['train_noisy'], True, True),
        meta=_loader(data['meta_clean'], True, True),
        test=_loader(data['test_clean'], False, False),
    )


def make_toy_manifold(n_train=400, n_meta=200, n_test=100, noise_std=0.1,
                      latent_dim=2, batch_size=64, seed=1):
    """合成一维流形（单位圆嵌入 2D）+ 高斯噪声，用于快速测试。

    带噪训练样本 = 流形点 + 噪声；元/测试集 = 干净流形点。
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * math.pi, n_train)
    train = np.stack([np.cos(theta), np.sin(theta)], axis=1) + noise_std * rng.standard_normal((n_train, 2))
    theta_m = rng.uniform(0, 2 * math.pi, n_meta)
    meta = np.stack([np.cos(theta_m), np.sin(theta_m)], axis=1)
    theta_t = rng.uniform(0, 2 * math.pi, n_test)
    test = np.stack([np.cos(theta_t), np.sin(theta_t)], axis=1)
    train_t = torch.tensor(train, dtype=torch.float32)
    meta_t = torch.tensor(meta, dtype=torch.float32)
    test_t = torch.tensor(test, dtype=torch.float32)
    return dict(
        train=DataLoader(TensorDataset(train_t), batch_size=batch_size, shuffle=True, drop_last=True),
        meta=DataLoader(TensorDataset(meta_t), batch_size=batch_size, shuffle=True, drop_last=True),
        test=DataLoader(TensorDataset(test_t), batch_size=batch_size, shuffle=False),
        latent_dim=latent_dim,
    )
