# -*- coding: utf-8 -*-
"""三个一阶双层极小极大求解器：SSGDA（算法1）、TSGDA-1（算法2）、TSGDA-2（算法3）。

严格按论文附录 D/E 的更新规则实现：

算法1 SSGDA（单时间尺度，每步 t）：
    采样 ξ∈D_m1, ζ∈D_m2
    x_{t+1} = x_t − η ∇_x f(x_t, y_t, z_t; ξ)
    y_{t+1} = y_t − γ1 ∇_y g(x_{t+1}, y_t, z_t; ζ)
    z_{t+1} = z_t + γ2 ∇_z g(x_{t+1}, y_{t+1}, z_t; ζ)

算法2 TSGDA-1（一个内循环 K 步）：
    x_{t+1} = x_t − η ∇_x f(x_t, y_{t,0}, z_{t,0}; ξ_t)
    for k=0..K−1（采样 ζ_k）:
        y_{t,k+1} = y_{t,k} − γ1 ∇_y g(x_{t+1}, y_{t,k}, z_{t,k}; ζ_k)
        z_{t,k+1} = z_{t,k} + γ2 ∇_z g(x_{t+1}, y_{t,k+1}, z_{t,k}; ζ_k)

算法3 TSGDA-2（两个独立内循环 K 步与 Q 步）：
    x_{t+1} = x_t − η ∇_x f(x_t, y_{t,0}, z_{t,0}; ξ_t)
    for k=0..K−1: y_{t,k+1} = y_{t,k} − γ1 ∇_y g(x_{t+1}, y_{t,k}, z_{t,0}; ζ_k)
    y_{t+1,0} = y_{t,K}
    for q=0..Q−1: z_{t,q+1} = z_{t,q} + γ2 ∇_z g(x_{t+1}, y_{t+1,0}, z_{t,q}; ζ_q)

一阶工程设计（对齐 First_Order_BMO / NeurIPS 2024 的 FOSL 实现）：
- 不使用超梯度与内循环展开（无 create_graph / higher）；
- 上层采用动量对偶平均：m ← (1−β)·m + β·grad;  x ← x − η_t·m
  （对应 FOSL 中 z_w = (1−β)z_w + β·grad_w, w −= lr·z_w）；
- 步长调度满足论文引理的衰减条件 η_t ≤ c/((t+1)L)、γ^k ≤ c/((k+1)L)，
  同时支持论文实验中的 fixed / 0.95 / 0.85 指数衰减；
- 梯度范数裁剪，模拟有界梯度假设；可选尾部平均输出（算法输出 x̄,ȳ,z̄）。
"""
import copy
import torch

from .bmo_task import BMOTask, MetricHistory


class StepSizeSchedule:
    """步长调度器。

    mode:
        'fixed'      : 恒定 η0（论文实验的 Fixed）
        'exp'        : η0 · decay^i（论文实验 0.95=Decaying, 0.85=Faster Decaying）
        'inv'        : η0 / (i+1)（理论条件 η_i ≤ c/((i+1)L)）
        'inv_sqrt'   : η0 / sqrt(i+1)
    i 为全局迭代索引（外层传 t，内层传全局 k 计数）。
    """

    MODES = ('fixed', 'exp', 'inv', 'inv_sqrt')

    def __init__(self, eta0, mode='exp', decay=0.95):
        assert mode in self.MODES, f'未知步长调度 {mode}'
        self.eta0 = eta0
        self.mode = mode
        self.decay = decay

    def __call__(self, i):
        if self.mode == 'fixed':
            return self.eta0
        if self.mode == 'exp':
            return self.eta0 * (self.decay ** i)
        if self.mode == 'inv':
            return self.eta0 / (i + 1.0)
        return self.eta0 / ((i + 1.0) ** 0.5)


def _clip_grads(params, max_norm):
    if max_norm is not None and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def _zero_grads(params):
    for p in params:
        if p.grad is not None:
            p.grad = None


class _BaseSolver:
    """公共骨架：持有任务、步长调度、上层动量缓冲与评估记录。"""

    solver_name = 'base'

    def __init__(self, task: BMOTask,
                 eta=1e-3, gamma1=1e-3, gamma2=1e-3,
                 eta_schedule='exp', eta_decay=0.95,
                 gamma_schedule='exp', gamma_decay=0.95,
                 beta_momentum=0.9, grad_clip=10.0,
                 seed=1):
        self.task = task
        self.device = task.device
        self.eta_sched = StepSizeSchedule(eta, eta_schedule, eta_decay)
        self.gamma1_sched = StepSizeSchedule(gamma1, gamma_schedule, gamma_decay)
        self.gamma2_sched = StepSizeSchedule(gamma2, gamma_schedule, gamma_decay)
        self.beta = beta_momentum
        self.grad_clip = grad_clip
        self.rng = torch.Generator().manual_seed(seed)
        # 上层对偶平均动量缓冲（First_Order_BMO 工程设计）
        self.m_x = [torch.zeros_like(p) for p in task.weight_net.parameters()]
        self.t = 0            # 全局外层步计数
        self.k_global = 0     # 全局内层步计数
        self.history = MetricHistory()

    # ----- 数据采样 -----
    def _sample(self, loader):
        if not hasattr(self, '_iter') or self._iter is None:
            self._iter = iter(loader)
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(loader)
            batch = next(self._iter)
        return batch

    # ----- 上层更新（一阶 + 动量对偶平均）-----
    def _update_upper(self, meta_batch, t):
        eta = self.eta_sched(t)
        weight_net = self.task.weight_net
        _zero_grads(weight_net.parameters())
        f = self.task.upper_loss(meta_batch)          # f(x, y_t, z_t; ξ)
        f.backward()                                  # 纯一阶：∂f/∂x
        params = list(weight_net.parameters())
        _clip_grads(params, self.grad_clip)
        with torch.no_grad():
            for p, m in zip(params, self.m_x):
                m.mul_(1.0 - self.beta).add_(p.grad, alpha=self.beta)
                p.add_(m, alpha=-eta)
        return f.item(), eta

    # ----- 下层更新 -----
    def _update_lower_yz(self, train_batch, sched_idx):
        """一次交替的 y 然后 z 更新（算法1/算法2 的内层一步）。

        步长调度索引 sched_idx：
        - TSGDA-1/2 的内循环：按理论条件 γ^k ≤ c/((k+1)L) 以内层索引 k 计，
          每个外层步的内循环重新从 γ0 开始（k 每轮重置）；
        - SSGDA：以外层步 t 计（单时间尺度衰减）。
        """
        gamma1 = self.gamma1_sched(sched_idx)
        gamma2 = self.gamma2_sched(sched_idx)
        g_y, _ = self.task.lower_losses(train_batch)
        _zero_grads(self.task.netG.parameters())
        g_y.backward()
        gp = list(self.task.netG.parameters())
        _clip_grads(gp, self.grad_clip)
        with torch.no_grad():
            for p in gp:
                p.add_(p.grad, alpha=-gamma1)         # y_{k+1} = y_k − γ1 ∇_y g
        self.k_global += 1

        gamma1 = self.gamma1_sched(sched_idx)
        gamma2 = self.gamma2_sched(sched_idx)
        _, g_z = self.task.lower_losses(train_batch)
        _zero_grads(self.task.netD.parameters())
        (-g_z).backward()                             # 最大化 g_z
        dp = list(self.task.netD.parameters())
        _clip_grads(dp, self.grad_clip)
        with torch.no_grad():
            for p in dp:
                p.add_(p.grad, alpha=gamma2)          # z_{k+1} = z_k + γ2 ∇_z g
        self.k_global += 1
        return g_y.item(), g_z.item(), gamma1, gamma2

    def _update_lower_y_only(self, train_batch, k):
        gamma1 = self.gamma1_sched(k)
        g_y, _ = self.task.lower_losses(train_batch)
        _zero_grads(self.task.netG.parameters())
        g_y.backward()
        gp = list(self.task.netG.parameters())
        _clip_grads(gp, self.grad_clip)
        with torch.no_grad():
            for p in gp:
                p.add_(p.grad, alpha=-gamma1)
        self.k_global += 1
        return g_y.item(), gamma1

    def _update_lower_z_only(self, train_batch, k):
        gamma2 = self.gamma2_sched(k)
        _, g_z = self.task.lower_losses(train_batch)
        _zero_grads(self.task.netD.parameters())
        (-g_z).backward()
        dp = list(self.task.netD.parameters())
        _clip_grads(dp, self.grad_clip)
        with torch.no_grad():
            for p in dp:
                p.add_(p.grad, alpha=gamma2)
        self.k_global += 1
        return g_z.item(), gamma2

    # ----- 评估 -----
    @torch.no_grad()
    def evaluate(self, train_loader, meta_loader, test_loader, step_label):
        task = self.task
        train_err = task.train_error(train_loader, max_batches=8)
        val_err = task.eval_error(meta_loader)
        test_err = task.eval_error(test_loader)
        val_raw = task.eval_error(meta_loader, weighted=False)
        test_raw = task.eval_error(test_loader, weighted=False)
        self.history.log(step_label, train_err, val_err, test_err,
                         val_error_raw=val_raw, test_error_raw=test_raw)
        return train_err, val_err, test_err

    def snapshot_state(self):
        return copy.deepcopy(dict(
            G=self.task.netG.state_dict(),
            D=self.task.netD.state_dict(),
            W=self.task.weight_net.state_dict(),
        ))

    # 子类实现
    def run(self, *args, **kwargs):
        raise NotImplementedError


class SSGDA(_BaseSolver):
    """算法1：单时间尺度随机梯度下降-上升。K=1。"""

    solver_name = 'SSGDA'

    def run(self, train_loader, meta_loader, test_loader, T=200,
            eval_every=10, verbose=False):
        self.evaluate(train_loader, meta_loader, test_loader, 0)
        for t in range(self.t, self.t + T):
            meta_batch = self._sample(meta_loader)
            train_batch = self._sample(train_loader)
            f_val, eta = self._update_upper(meta_batch, t)              # 行3
            g_y, g_z, gamma1, gamma2 = self._update_lower_yz(train_batch, t)  # 行4-5
            if verbose and (t + 1) % eval_every == 0:
                print(f'[SSGDA] t={t+1} f={f_val:.4f} g_y={g_y:.4f} g_z={g_z:.4f} '
                      f'eta={eta:.2e} g1={gamma1:.2e} g2={gamma2:.2e}')
            if (t + 1) % eval_every == 0:
                self.evaluate(train_loader, meta_loader, test_loader, t + 1)
        self.t += T
        return self.history


class TSGDA1(_BaseSolver):
    """算法2：两时间尺度、一个内循环（K 步交替 y/z）。"""

    solver_name = 'TSGDA-1'

    def run(self, train_loader, meta_loader, test_loader, T=50, K=300,
            eval_ks=(10, 100, 150, 200, 250, 300), verbose=False):
        """外层 T 步；在内层推进到 k ∈ eval_ks 时评估（论文图1/2/3 的记录方式）。"""
        self.evaluate(train_loader, meta_loader, test_loader, self.k_global)
        for t in range(self.t, self.t + T):
            meta_batch = self._sample(meta_loader)
            f_val, eta = self._update_upper(meta_batch, t)              # 行3
            done_k = 0
            for k in range(K):                                          # 行4-8
                train_batch = self._sample(train_loader)
                g_y, g_z, gamma1, gamma2 = self._update_lower_yz(train_batch, k)
                done_k += 1
                if done_k in eval_ks:
                    # 论文图1/4 协议：以循环内位置 k 为标签记录，
                    # 之后对同一 k 的多次记录（不同外层步）取平均
                    self.evaluate(train_loader, meta_loader, test_loader, done_k)
                    if verbose:
                        print(f'[TSGDA-1] t={t+1} k={done_k} f={f_val:.4f} '
                              f'g_y={g_y:.4f} g_z={g_z:.4f}')
        self.t += T
        return self.history


class TSGDA2(_BaseSolver):
    """算法3：两时间尺度、两个独立内循环（K 步 y，Q 步 z）。"""

    solver_name = 'TSGDA-2'

    def run(self, train_loader, meta_loader, test_loader, T=50, K=300, Q=300,
            eval_ks=(10, 100, 150, 200, 250, 300), verbose=False):
        self.evaluate(train_loader, meta_loader, test_loader, self.k_global)
        for t in range(self.t, self.t + T):
            meta_batch = self._sample(meta_loader)
            f_val, eta = self._update_upper(meta_batch, t)              # 行3
            done = 0
            for k in range(K):                                          # 行4-7: 仅 y
                train_batch = self._sample(train_loader)
                g_y, gamma1 = self._update_lower_y_only(train_batch, k)
                done += 1
                if done in eval_ks:
                    self.evaluate(train_loader, meta_loader, test_loader, done)
                    if verbose:
                        print(f'[TSGDA-2] t={t+1} k={done} f={f_val:.4f} g_y={g_y:.4f}')
            for q in range(Q):                                          # 行9-12: 仅 z
                train_batch = self._sample(train_loader)
                g_z, gamma2 = self._update_lower_z_only(train_batch, q)
                done += 1
                if done - K in eval_ks:
                    self.evaluate(train_loader, meta_loader, test_loader, done - K)
                    if verbose:
                        print(f'[TSGDA-2] t={t+1} q={done-K} f={f_val:.4f} g_z={g_z:.4f}')
        self.t += T
        return self.history


SOLVERS = {'SSGDA': SSGDA, 'TSGDA-1': TSGDA1, 'TSGDA-2': TSGDA2}
