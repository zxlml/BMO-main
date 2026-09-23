"""BMO 训练入口（重写版）。

原实现的缺陷（本版已修复）：
1. `type(self.netG)().to(...).load_state_dict(...)` 误用（load_state_dict 返回 None），
   伪网络从未被真正克隆；本版一阶算法无需展开伪网络。
2. 元网络权重在 torch.no_grad 下使用，meta_loss.backward() 对元网络无梯度；
   本版上层 f 通过权重输出保留 ∂f/∂x（纯一阶，无超梯度/展开）。
3. `create_model(opt)` 被调用两次，产生两套互不相干的 G/D；本版统一构建。
4. 上层更新应使用元集样本且先于下层内循环（算法1-3 的顺序）；原实现顺序错误。
5. 使用固定 Adam；本版按论文使用可衰减步长（fixed/0.95/0.85/理论调度），
   并加入 First_Order_BMO 的动量对偶平均与梯度裁剪以保证收敛。

用法示例：
    python train_test_BMO.py --solver TSGDA-1 --T 10 --K 100 --img_size 32
    python train_test_BMO.py --solver SSGDA --T 100
"""
import os
import sys
import argparse

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bmo_core import (generate_simulation_data, make_dataloaders,
                      WeightNet, ConvGenerator, ConvDiscriminator,
                      BMOTask, SOLVERS)

FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'Create_real_data', 'Chaplin', 'frames')


def parse_opts():
    p = argparse.ArgumentParser(description='BMO bilevel GAN reweighting training')
    p.add_argument('--solver', choices=list(SOLVERS), default='TSGDA-1')
    p.add_argument('--T', type=int, default=10, help='外层迭代数')
    p.add_argument('--K', type=int, default=100, help='内层迭代数（TSGDA-2 时同时用于 Q）')
    p.add_argument('--Q', type=int, default=100, help='TSGDA-2 的 z 内循环数')
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--latent_dim', type=int, default=16)
    p.add_argument('--width_g', type=int, default=64, help='生成器通道宽度')
    p.add_argument('--width_d', type=int, default=4, help='判别器通道宽度')
    p.add_argument('--noise_std', type=float, default=0.1)
    p.add_argument('--fake_noise', type=float, default=None,
                   help='假样本噪声幅度；默认与 --noise_std 一致，0 表示假样本不加噪')
    p.add_argument('--num_copies', type=int, default=20)
    p.add_argument('--m1', type=int, default=500, help='元集大小（不足时增广）')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--eta', type=float, default=1e-3)
    p.add_argument('--gamma1', type=float, default=1e-3)
    p.add_argument('--gamma2', type=float, default=1e-3)
    p.add_argument('--eta_schedule', default='exp', choices=['fixed', 'exp', 'inv', 'inv_sqrt'])
    p.add_argument('--eta_decay', type=float, default=0.95)
    p.add_argument('--gamma_schedule', default='exp', choices=['fixed', 'exp', 'inv', 'inv_sqrt'])
    p.add_argument('--gamma_decay', type=float, default=0.95)
    p.add_argument('--beta', type=float, default=0.9, help='上层对偶平均动量')
    p.add_argument('--grad_clip', type=float, default=10.0)
    p.add_argument('--eval_every', type=int, default=2)
    p.add_argument('--eval_ks', type=int, nargs='+', default=[10, 50, 100])
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--save_dir', default='checkpoints')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def main():
    opt = parse_opts()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(opt.seed)

    # 1. 仿真数据（按 Generate_data.m 的生成方式）
    data = generate_simulation_data(FRAMES_DIR, img_size=opt.img_size,
                                    num_copies=opt.num_copies,
                                    noise_std=opt.noise_std,
                                    m1=opt.m1, seed=opt.seed)
    loaders = make_dataloaders(data, batch_size=opt.batch_size, device=device)
    print(f'训练集(带噪) {data["train_noisy"].shape[0]} | '
          f'元集(干净) {data["meta_clean"].shape[0]} | '
          f'测试集(干净) {data["test_clean"].shape[0]}')

    # 2. 网络与任务（与 run_experiment.build_task 对齐：平衡区配置 强G弱D）
    netG = ConvGenerator(latent_dim=opt.latent_dim, img_size=opt.img_size,
                         width=opt.width_g).to(device)
    netD = ConvDiscriminator(img_size=opt.img_size, width=opt.width_d).to(device)
    weight_net = WeightNet(hidden_size=10, num_layers=2).to(device)  # 1-10-10-1
    task = BMOTask(netG, netD, weight_net,
                   noise_std=opt.noise_std if opt.fake_noise is None else opt.fake_noise,
                   latent_dim=opt.latent_dim, device=device)

    # 3. 一阶求解器（无超梯度；First_Order_BMO 工程设计）
    solver = SOLVERS[opt.solver](
        task, eta=opt.eta, gamma1=opt.gamma1, gamma2=opt.gamma2,
        eta_schedule=opt.eta_schedule, eta_decay=opt.eta_decay,
        gamma_schedule=opt.gamma_schedule, gamma_decay=opt.gamma_decay,
        beta_momentum=opt.beta, grad_clip=opt.grad_clip, seed=opt.seed)

    # 4. 训练 + 评估（训练/验证/测试误差与泛化 GAP）
    if opt.solver == 'SSGDA':
        solver.run(loaders['train'], loaders['meta'], loaders['test'],
                   T=opt.T, eval_every=opt.eval_every, verbose=opt.verbose)
    elif opt.solver == 'TSGDA-1':
        solver.run(loaders['train'], loaders['meta'], loaders['test'],
                   T=opt.T, K=opt.K, eval_ks=opt.eval_ks, verbose=opt.verbose)
    else:
        solver.run(loaders['train'], loaders['meta'], loaders['test'],
                   T=opt.T, K=opt.K, Q=opt.Q, eval_ks=opt.eval_ks, verbose=opt.verbose)

    df = solver.history.records
    print('\nstep | train | val | test | gap')
    for i in range(len(df['step'])):
        print(f"{df['step'][i]} | {df['train_error'][i]:.4f} | "
              f"{df['val_error'][i]:.4f} | {df['test_error'][i]:.4f} | "
              f"{df['gap'][i]:.4f}")

    os.makedirs(opt.save_dir, exist_ok=True)
    tag = f'{opt.solver}_T{opt.T}_K{opt.K}'
    torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                'weight_net': weight_net.state_dict()},
               os.path.join(opt.save_dir, f'bmo_{tag}.pth'))
    print(f'模型已保存到 {os.path.join(opt.save_dir, f"bmo_{tag}.pth")}')


if __name__ == '__main__':
    main()
