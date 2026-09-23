# -*- coding: utf-8 -*-
"""网络结构模块。

- WeightNet: 上层元网络，论文附录 A 指定 1-10-10-1 MLP，隐层 ReLU，输出 Sigmoid。
- MLPGenerator / MLPDiscriminator: 向量数据的生成器与判别器（快速测试用）。
- ConvGenerator / ConvDiscriminator: 低分辨率灰度图的生成器与判别器
  （对应 MFCGAN 的 denoising 任务：G 把隐向量映射到图像空间，
   D 判别 干净图像(真) vs G(隐)+σ·噪声(假)）。
"""
import torch
import torch.nn as nn


class WeightNet(nn.Module):
    """上层元网络：输入单样本损失值，输出该样本的权重 ∈ (0,1)。"""

    def __init__(self, hidden_size=10, num_layers=2):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, loss_vector):
        # loss_vector: (B,) 或 (B,1)
        if loss_vector.dim() == 1:
            loss_vector = loss_vector.unsqueeze(1)
        return torch.sigmoid(self.net(loss_vector)).squeeze(1)


class MLPGenerator(nn.Module):
    def __init__(self, latent_dim=2, hidden=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class MLPDiscriminator(nn.Module):
    def __init__(self, in_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


class ConvGenerator(nn.Module):
    """低分辨率灰度图生成器：latent -> (1,H,W)，输出经 Sigmoid 到 [0,1]。

    width 控制通道宽度（默认 64；减小可制造过拟合压力）。
    """

    def __init__(self, latent_dim=16, img_size=32, width=64):
        super().__init__()
        s = img_size
        self.init_size = s // 8
        self.img_size = s
        self.fc = nn.Linear(latent_dim, width * self.init_size * self.init_size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(width, width // 2, 4, stride=2, padding=1),   # 2x
            nn.BatchNorm2d(width // 2), nn.ReLU(),
            nn.ConvTranspose2d(width // 2, width // 4, 4, stride=2, padding=1),  # 4x
            nn.BatchNorm2d(width // 4), nn.ReLU(),
            nn.ConvTranspose2d(width // 4, 1, 4, stride=2, padding=1),    # 8x
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), self.deconv[0].in_channels,
                            self.init_size, self.init_size)
        return self.deconv(h)


class ConvDiscriminator(nn.Module):
    def __init__(self, img_size=32, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(width, width * 2, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(width * 4 * (img_size // 8) ** 2, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)
