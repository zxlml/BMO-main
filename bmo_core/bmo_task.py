# -*- coding: utf-8 -*-
"""BMO 任务定义：把图像去噪/生成任务绑定到论文公式 (2) 的双层极小极大目标。

    min_{x}  E_{ξ∈D_m1} f(x, y*(x), z*(x); ξ)
    s.t.     y*(x), z*(x) ∈ arg min_y arg max_z E_{ζ∈D_m2} g(x, y, z; ζ)

变量含义（与论文 4.1 节一致）：
    x : 上层 MLP（实例加权网络）参数
    y : 生成器 G 参数（下层极小者）
    z : 判别器 D 参数（下层极大者）

下层极小极大目标（带噪训练集 D_m2 上的加权 GAN 损失，权重由 x 给出）：
    fake_i = G(ε_i) + σ·ν_i                    （带噪生成样本，对应 MFCGAN 前向）
    g_z 部分（z 最大化）: Σ_i w_i·BCE(D(fake_i), 0) + Σ_i w_i·BCE(D(X_i^{noisy}), 1)
    g_y 部分（y 极小化）: Σ_i w_i·BCE(D(fake_i), 1)
    其中 w_i = WeightNet_x(该样本损失值)，权重输入对 D/Gdetach（一阶、无超梯度）。

上层目标（干净元集 D_m1 上）：
    f(x, y, z; ξ) = Σ_j w_j(x, ℓ_j)·BCE(D(clean_j), 1)，ℓ_j 为 D 对该干净样本的
    逐样本 BCE。验证误差 = f 在元集上的取值；测试误差 = f 在测试集上的取值；
    泛化 GAP（论文式 (5) 的估计）= 验证误差 − 测试误差。
"""
import torch
import torch.nn.functional as F

from .networks import WeightNet


class BMOTask:
    """封装 f / g 与评估指标，供三个一阶求解器调用。"""

    def __init__(self, netG, netD, weight_net, noise_std=0.1, latent_dim=16,
                 device='cpu', weight_clip=(0.05, 1.0)):
        self.netG = netG
        self.netD = netD
        self.weight_net = weight_net
        self.noise_std = noise_std
        self.latent_dim = latent_dim
        self.device = device
        self.weight_clip = weight_clip

    # ---------- 基础组件 ----------
    def sample_latent(self, n):
        return torch.randn(n, self.latent_dim, device=self.device)

    def make_fake(self, n):
        """带噪生成样本 fake = G(ε) + σ·ν。"""
        fake = self.netG(self.sample_latent(n))
        if self.noise_std > 0:
            fake = fake + self.noise_std * torch.randn_like(fake)
        return fake

    def _weights(self, loss_vec, weight_grad=False):
        """由元网络按逐样本损失生成权重。

        损失输入始终 detach（保证一阶、无超梯度：D/G 不经权重输入回传梯度）。
        weight_grad=False（下层）：权重作为常量，不回传到 x；
        weight_grad=True （上层 f）：保留 ∂w/∂x，使 ∇_x f 可由一阶反向传播得到。
        """
        if weight_grad:
            w = self.weight_net(loss_vec.detach())
        else:
            with torch.no_grad():
                w = self.weight_net(loss_vec.detach())
        lo, hi = self.weight_clip
        return w.clamp(lo, hi)

    # ---------- 下层极小极大目标 g ----------
    def lower_losses(self, noisy_batch):
        """返回 (g_y_loss, g_z_loss)。

        g_z: z 的极大化目标（BCE 求和后取负号更新，见求解器）。
        g_y: y 的极小化目标。
        注意：fake 对 D 的梯度在 z 更新中使用；对 G 的梯度在 y 更新中使用，
        因此这里分别重新前向，避免跨变量梯度污染。
        """
        x_noisy = noisy_batch[0] if isinstance(noisy_batch, (tuple, list)) else noisy_batch
        x_noisy = x_noisy.to(self.device)
        n = x_noisy.size(0)

        # --- D 的目标（对 G detach）---
        fake_for_d = self.make_fake(n).detach()
        pred_fake = self.netD(fake_for_d)
        loss_d_fake_vec = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake), reduction='none')
        pred_real = self.netD(x_noisy)
        loss_d_real_vec = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real), reduction='none')
        w_fake = self._weights(loss_d_fake_vec)
        w_real = self._weights(loss_d_real_vec)
        g_z = (w_fake * loss_d_fake_vec).mean() + (w_real * loss_d_real_vec).mean()

        # --- G 的目标（对 D detach）---
        for p in self.netD.parameters():
            p.requires_grad_(False)
        try:
            fake_for_g = self.make_fake(n)
            pred_g = self.netD(fake_for_g)
            loss_g_vec = F.binary_cross_entropy_with_logits(pred_g, torch.ones_like(pred_g), reduction='none')
            w_g = self._weights(loss_g_vec)
            g_y = (w_g * loss_g_vec).mean()
        finally:
            for p in self.netD.parameters():
                p.requires_grad_(True)

        return g_y, g_z

    # ---------- 上层目标 f ----------
    def upper_loss(self, clean_batch, reduction='mean'):
        """f(x, y, z; ξ)：干净样本经 D 分类为真，逐样本权重来自元网络。"""
        x_clean = clean_batch[0] if isinstance(clean_batch, (tuple, list)) else clean_batch
        x_clean = x_clean.to(self.device)
        pred = self.netD(x_clean)
        loss_vec = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred), reduction='none')
        w = self._weights(loss_vec, weight_grad=True)
        if reduction == 'none':
            return w * loss_vec
        return (w * loss_vec).mean()

    # ---------- 评估指标 ----------
    @torch.no_grad()
    def eval_error(self, loader, weighted=True):
        """在整个 loader 上评估上层误差（默认加权；weighted=False 为原始 BCE）。"""
        self.netD.eval()
        total, count = 0.0, 0
        for batch in loader:
            x = batch[0].to(self.device)
            pred = self.netD(x)
            loss_vec = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred), reduction='none')
            if weighted:
                w = self.weight_net(loss_vec).clamp(*self.weight_clip)
                total += (w * loss_vec).sum().item()
            else:
                total += loss_vec.sum().item()
            count += x.size(0)
        self.netD.train()
        return total / max(count, 1)

    @torch.no_grad()
    def train_error(self, loader, max_batches=None):
        """训练误差：带噪训练集上 D 将带噪样本判为真的 BCE（越低越 '欺骗' 成功），
        结合加权 G 损失的含义，报告加权的 D-假样本损失。"""
        self.netG.eval()
        total, count = 0.0, 0
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x = batch[0].to(self.device)
            n = x.size(0)
            fake = self.make_fake(n).detach()
            pred = self.netD(fake)
            loss_vec = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred), reduction='none')
            w = self.weight_net(loss_vec).clamp(*self.weight_clip)
            total += (w * loss_vec).sum().item()
            count += n
        self.netG.train()
        return total / max(count, 1)


class MetricHistory:
    """训练/验证/测试误差与泛化 GAP 的记录器。"""

    KEYS = ('step', 'train_error', 'val_error', 'test_error', 'gap',
            'val_error_raw', 'test_error_raw', 'gap_raw')

    def __init__(self):
        self.records = {k: [] for k in self.KEYS}

    def log(self, step, train_error, val_error, test_error,
            val_error_raw=None, test_error_raw=None):
        self.records['step'].append(step)
        self.records['train_error'].append(train_error)
        self.records['val_error'].append(val_error)
        self.records['test_error'].append(test_error)
        # 泛化 GAP（论文式(5)的估计）：测试误差 − 验证误差。
        # 论文图1c 中 GAP 为正且随过量迭代增长（验证误差下降、测试误差上升），
        # 对应 test − val 的定义。
        self.records['gap'].append(test_error - val_error)
        # 未加权（raw）指标：排除元网络 w 的影响，观察 D 本身的泛化。
        if val_error_raw is None or test_error_raw is None:
            val_error_raw = val_error if val_error_raw is None else val_error_raw
            test_error_raw = test_error if test_error_raw is None else test_error_raw
        self.records['val_error_raw'].append(val_error_raw)
        self.records['test_error_raw'].append(test_error_raw)
        self.records['gap_raw'].append(test_error_raw - val_error_raw)

    def last(self, key):
        return self.records[key][-1] if self.records[key] else None
