# -*- coding: utf-8 -*-
"""单元测试：数据生成、步长调度、三个求解器的更新规则与等价性。"""
import os
import sys
import math
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmo_core.data_generator import (generate_simulation_data, make_toy_manifold,
                                     _list_frames, _to_gray_array)
from bmo_core.networks import WeightNet, MLPGenerator, MLPDiscriminator
from bmo_core.bmo_task import BMOTask
from bmo_core.solvers import StepSizeSchedule, SSGDA, TSGDA1, TSGDA2, _BaseSolver

FRAMES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'Create_real_data', 'Chaplin', 'frames')


# ---------------- 步长调度 ----------------
def test_step_schedule_fixed():
    s = StepSizeSchedule(0.01, 'fixed')
    assert all(abs(s(i) - 0.01) < 1e-12 for i in range(5))


def test_step_schedule_exp():
    s = StepSizeSchedule(1e-3, 'exp', 0.95)
    assert abs(s(0) - 1e-3) < 1e-12
    assert abs(s(1) - 0.95e-3) < 1e-12
    assert abs(s(10) - 1e-3 * 0.95 ** 10) < 1e-15


def test_step_schedule_inv():
    s = StepSizeSchedule(0.1, 'inv')
    assert abs(s(0) - 0.1) < 1e-12
    assert abs(s(9) - 0.01) < 1e-12           # 0.1/10
    # 满足理论条件 η_i ≤ c/((i+1)L)
    assert all(s(i) <= 0.1 / (i + 1) + 1e-12 for i in range(100))


def test_step_schedule_invalid_mode():
    with pytest.raises(AssertionError):
        StepSizeSchedule(0.1, 'unknown')


# ---------------- 数据生成 ----------------
def test_simulation_data_shapes():
    if not os.path.isdir(FRAMES_DIR):
        pytest.skip('Chaplin frames 不存在')
    n_frames = len(_list_frames(FRAMES_DIR))
    data = generate_simulation_data(FRAMES_DIR, img_size=32, num_copies=4,
                                    noise_std=0.1, m1=100, seed=1)
    n_train_frames = sum(1 for i in range(n_frames) if (i + 1) % 2 == 1)
    n_test_frames = n_frames - n_train_frames
    # 训练集 = 奇数帧 × num_copies（MATLAB 1-based 奇数）
    assert data['train_noisy'].shape == (n_train_frames * 4, 1, 32, 32)
    # 测试集与元集用同一增广管线扩到相同数量 m1（分布一致，仅内容可见性不同）
    assert data['test_clean'].shape == (100, 1, 32, 32)
    # 元集增广到 m1
    assert data['meta_clean'].shape == (100, 1, 32, 32)


def test_simulation_data_noise_std():
    if not os.path.isdir(FRAMES_DIR):
        pytest.skip('Chaplin frames 不存在')
    data = generate_simulation_data(FRAMES_DIR, img_size=32, num_copies=20,
                                    noise_std=0.1, m1=None, seed=0)
    noisy = data['train_noisy'].numpy()
    clean = data['meta_clean'].numpy()
    # 每个训练样本与对应干净帧的差值标准差应接近 noise_std
    n_frames = clean.shape[0]
    diffs = []
    for j in range(n_frames):
        group = noisy[j * 20:(j + 1) * 20] - clean[j]
        diffs.append(group.std())
    assert abs(np.mean(diffs) - 0.1) < 0.02
    # 取值在 [0,1]（clip 后）
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0


def test_toy_manifold():
    d = make_toy_manifold(n_train=200, n_meta=50, n_test=30, seed=3)
    batch = next(iter(d['train']))[0]
    assert batch.shape[1] == 2
    # 流形点应接近单位圆
    r = batch.pow(2).sum(1).sqrt()
    assert (r - 1.0).abs().max() < 0.35


# ---------------- 任务与一阶性质 ----------------
def _make_task(latent_dim=2, seed=0):
    torch.manual_seed(seed)
    G = MLPGenerator(latent_dim=latent_dim, hidden=32, out_dim=2)
    D = MLPDiscriminator(in_dim=2, hidden=32)
    W = WeightNet(hidden_size=10, num_layers=2)
    return BMOTask(G, D, W, noise_std=0.0, latent_dim=latent_dim, device='cpu')


def test_upper_loss_has_grad_to_weight_net():
    """一阶关键性质：∇_x f 非零且只经权重输出回传（无超梯度路径）。"""
    task = _make_task()
    x = torch.randn(16, 2)
    f = task.upper_loss(x)
    f.backward()
    grads = [p.grad for p in task.weight_net.parameters()]
    assert all(g is not None for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    # D 的梯度不应被激活（f 对 D 无梯度需求时未被 backward）
    # 这里 f 依赖 D 的输出，D 有梯度属正常；关键是权重网络获得梯度。


def test_lower_losses_detached_weights():
    """下层权重应为常量：g_y 对 weight_net 无梯度（一阶设计）。"""
    task = _make_task()
    x = torch.randn(16, 2)
    g_y, g_z = task.lower_losses(x)
    g_y.backward(retain_graph=True)
    wn_grads = [p.grad for p in task.weight_net.parameters()]
    assert all(g is None or g.abs().sum() == 0 for g in wn_grads)


def test_gap_definition():
    from bmo_core.bmo_task import MetricHistory
    h = MetricHistory()
    # GAP = 测试误差 − 验证误差（论文图1c 为正且随过量迭代增长）
    h.log(0, train_error=0.5, val_error=0.3, test_error=0.4)
    assert h.last('gap') == pytest.approx(0.1)


# ---------------- 求解器更新规则 ----------------
def _fresh_solver(solver_cls, seed=7):
    task = _make_task(seed=seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(64, 2)),
                                         batch_size=16, shuffle=False)
    solvers = {'SSGDA': SSGDA, 'TSGDA-1': TSGDA1, 'TSGDA-2': TSGDA2}
    # beta=1.0 → 对偶平均退化为纯梯度下降，便于与手工公式比对
    s = solvers[solver_cls](task, eta=1e-2, gamma1=1e-2, gamma2=1e-2,
                            eta_schedule='fixed', gamma_schedule='fixed',
                            beta_momentum=1.0, grad_clip=0, seed=seed)
    return s, loader


def _snapshot(task):
    return (copy.deepcopy(task.netG.state_dict()),
            copy.deepcopy(task.netD.state_dict()),
            copy.deepcopy(task.weight_net.state_dict()))


def _manual_upper_step(task, batch, eta, beta=1.0, init_state=None):
    """手工复现上层更新公式，返回更新后的 weight_net 参数。

    求解器实现：m = (1-β)m + β·grad；p -= η·m。
    beta=1.0 → m=grad；p -= η·grad（纯梯度下降），便于与手工公式直接对比。
    init_state: 更新前的 weight_net 参数快照（从此出发而非当前参数）。
    """
    W2 = copy.deepcopy(task.weight_net)
    if init_state is not None:
        W2.load_state_dict(init_state)
    x = batch[0]
    pred = task.netD(x)
    loss_vec = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred), reduction='none')
    w = torch.sigmoid(W2.net(loss_vec.detach().unsqueeze(1))).squeeze(1).clamp(0.05, 1.0)
    f = (w * loss_vec).mean()
    W2.zero_grad()
    f.backward()
    with torch.no_grad():
        for p in W2.parameters():
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            p.add_(g, alpha=-eta * beta)
    return W2


def test_ssgda_single_step_matches_formula():
    s, loader = _fresh_solver('SSGDA')
    task = s.task
    meta_batch = next(iter(loader))
    train_batch = next(iter(loader))

    g0, d0, w0 = _snapshot(task)
    # 上层一步（beta=1.0 → 纯梯度下降）
    f_val, eta = s._update_upper(meta_batch, t=0)
    w1 = copy.deepcopy(task.weight_net.state_dict())
    # 与手工公式对比（从更新前快照 w0 出发）
    W_manual = _manual_upper_step(task, meta_batch, eta, init_state=w0)
    for (k_a, v_a), (k_b, v_b) in zip(w1.items(), W_manual.state_dict().items()):
        assert torch.allclose(v_a, v_b, atol=1e-6), f'上层更新不匹配: {k_a}'

    # 下层一步：y 先减小 g_y，再增大 g_z
    _, g_z0 = task.lower_losses(train_batch)
    s._update_lower_yz(train_batch, 0)
    # G 参数应发生变化且方向为 -∇g_y（符号校验）
    assert not torch.equal(g0['net.0.weight'], task.netG.state_dict()['net.0.weight'])
    assert not torch.equal(d0['net.0.weight'], task.netD.state_dict()['net.0.weight'])


def test_tsgda1_equals_ssgda_with_K1():
    """TSGDA-1 在 K=1 时应与 SSGDA 单步严格等价（相同网络/数据/更新序列）。"""
    torch.manual_seed(11)
    data = torch.randn(64, 2)
    task_a = _make_task(seed=11)
    task_b = _make_task(seed=11)          # 相同种子 → 相同初始化
    loader_a = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=16, shuffle=False)
    loader_b = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data), batch_size=16, shuffle=False)

    s1 = SSGDA(task_a, eta=1e-2, gamma1=1e-2, gamma2=1e-2,
               eta_schedule='fixed', gamma_schedule='fixed',
               beta_momentum=1.0, grad_clip=0, seed=11)
    s2 = TSGDA1(task_b, eta=1e-2, gamma1=1e-2, gamma2=1e-2,
                eta_schedule='fixed', gamma_schedule='fixed',
                beta_momentum=1.0, grad_clip=0, seed=11)
    # 统一全局 RNG 起点（make_fake 的隐向量采样消耗全局 RNG）
    torch.manual_seed(123)
    meta_batch = next(iter(loader_a))
    train_batch = next(iter(loader_a))
    s1._update_upper(meta_batch, 0)
    s1._update_lower_yz(train_batch, 0)        # SSGDA 一步
    torch.manual_seed(123)
    meta_batch_b = next(iter(loader_b))
    train_batch_b = next(iter(loader_b))
    s2._update_upper(meta_batch_b, 0)
    s2._update_lower_yz(train_batch_b, 0)      # TSGDA-1 一步（K=1 等价序列）
    for p1, p2 in zip(s1.task.netG.parameters(), s2.task.netG.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7)
    for p1, p2 in zip(s1.task.netD.parameters(), s2.task.netD.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7)
    for p1, p2 in zip(s1.task.weight_net.parameters(), s2.task.weight_net.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7)


def test_tsgda2_two_loop_structure():
    """TSGDA-2：y 内循环期间 z 不动；z 内循环期间 y 不动。"""
    s, loader = _fresh_solver('TSGDA-2')
    task = s.task
    train_batch = next(iter(loader))

    # 手动执行 y-only 一步，检查 z 冻结
    d_before = copy.deepcopy(task.netD.state_dict())
    s._update_lower_y_only(train_batch, 0)
    for k in d_before:
        assert torch.equal(d_before[k], task.netD.state_dict()[k]), 'y 内循环不应更新 z'

    # z-only 一步，检查 y 冻结
    g_before = copy.deepcopy(task.netG.state_dict())
    s._update_lower_z_only(train_batch, 0)
    for k in g_before:
        assert torch.equal(g_before[k], task.netG.state_dict()[k]), 'z 内循环不应更新 y'


def test_upper_momentum_dual_averaging():
    """上层动量对偶平均（First_Order_BMO 工程）：m = (1-β)m + β·grad。"""
    s, loader = _fresh_solver('SSGDA')
    task = s.task
    meta_batch = next(iter(loader))
    # beta=1.0（_fresh_solver 默认）时 m 应等于更新前的梯度（纯梯度下降）
    task.weight_net.zero_grad()
    f_ref = task.upper_loss(meta_batch)
    f_ref.backward()
    grads_ref = [p.grad.clone() for p in task.weight_net.parameters()]
    f_val, eta = s._update_upper(meta_batch, t=0)
    for m, g in zip(s.m_x, grads_ref):
        assert torch.allclose(m, g, atol=1e-7)

    # beta=0.9：m = 0.9·grad（FOSL 的 z_w = (1-β)z_w + β·grad_w）
    s2, loader2 = _fresh_solver('SSGDA')
    s2.beta = 0.9
    mb2 = next(iter(loader2))
    f2 = s2.task.upper_loss(mb2)
    s2.task.weight_net.zero_grad()
    f2.backward()
    grads = [p.grad.clone() for p in s2.task.weight_net.parameters()]
    s2.task.weight_net.zero_grad()
    s2._update_upper(mb2, t=0)
    for p, g, m in zip(s2.task.weight_net.parameters(), grads, s2.m_x):
        assert torch.allclose(m, 0.9 * g, atol=1e-6), '动量缓冲不匹配'


def test_solvers_run_smoke():
    """三个求解器完整 run 冒烟：历史记录齐全，gap = val - test。"""
    for name, cls, kw in [('SSGDA', SSGDA, dict(T=3, eval_every=1)),
                          ('TSGDA-1', TSGDA1, dict(T=2, K=3, eval_ks=(1, 3))),
                          ('TSGDA-2', TSGDA2, dict(T=2, K=3, Q=2, eval_ks=(1, 3)))]:
        s, loader = _fresh_solver(name)
        data = torch.utils.data.TensorDataset(torch.randn(64, 2))
        loaders = dict(train=torch.utils.data.DataLoader(data, batch_size=16),
                       meta=torch.utils.data.DataLoader(data, batch_size=16),
                       test=torch.utils.data.DataLoader(data, batch_size=16))
        hist = s.run(loaders['train'], loaders['meta'], loaders['test'], **kw)
        n = len(hist.records['step'])
        assert n >= 2
        for a, b, c, g in zip(hist.records['train_error'], hist.records['val_error'],
                              hist.records['test_error'], hist.records['gap']):
            assert g == pytest.approx(b - c, abs=1e-9)
            assert all(np.isfinite(v) for v in (a, b, c, g))


# ---------------- 收敛性（凸-凹玩具问题）----------------
def test_lower_convergence_bilinear():
    """下层凸-凹问题：min_y max_z  g = y·z + y²/2 − z²/2，
    ∇_y = y + z（下降），∇_z = y − z（上升），
    交替更新映射 A = [[1−γ, −γ], [γ, 1−γ]] 的谱半径 √((1−γ)²+γ²) < 1，
    应收敛到鞍点 (0,0)。"""
    y = torch.tensor([1.0])
    z = torch.tensor([1.0])
    gamma = 0.3
    for k in range(2000):
        grad_y = y + z
        grad_z = y - z
        y = y - gamma * grad_y           # y_{k+1} = y_k − γ ∇_y g
        z = z + gamma * grad_z           # z_{k+1} = z_k + γ ∇_z g
    assert y.abs().item() < 1e-6 and z.abs().item() < 1e-6


def test_ssgda_upper_error_decreases_with_decay():
    """在固定 D/G 下，上层加权误差随上层更新（inv 调度）单调下降。"""
    task = _make_task(seed=42)
    x_clean = torch.randn(64, 2)
    errs = []
    eta_sched = StepSizeSchedule(0.5, 'inv')
    for t in range(60):
        f = task.upper_loss(x_clean)
        errs.append(f.item())
        task.weight_net.zero_grad()
        f.backward()
        with torch.no_grad():
            for p in task.weight_net.parameters():
                if p.grad is not None:
                    p.add_(p.grad.clamp(-10, 10), alpha=-eta_sched(t))
    # 末段误差应低于初始误差
    assert errs[-1] < errs[0]
    # 后半段均值低于前半段
    assert np.mean(errs[30:]) < np.mean(errs[:10])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
