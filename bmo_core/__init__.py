"""BMO 核心实现包。

复现论文《Fine-grained Analysis on the Stability and Generalization for
First-order Bilevel Minimax Optimization》中的三类一阶双层极小极大算法：
SSGDA（算法1）、TSGDA-1（算法2）、TSGDA-2（算法3）。

设计要点（对齐 First_Order_BMO / NeurIPS 2024 的工程方案）：
- 全程一阶：不做隐式微分/超梯度，不做内循环展开（无 create_graph）。
- 上层变量 x 使用动量对偶平均（z = (1-β)z + β·grad; x -= η·z）保证收敛平稳。
- 步长支持 fixed / 指数衰减(0.95, 0.85) / 理论一致 1/(t+1)、1/(k+1) 调度。
- 梯度范数裁剪，满足有界梯度假设。
"""

from .data_generator import generate_simulation_data, make_dataloaders
from .networks import WeightNet, MLPGenerator, MLPDiscriminator, ConvGenerator, ConvDiscriminator
from .bmo_task import BMOTask, MetricHistory
from .solvers import StepSizeSchedule, SSGDA, TSGDA1, TSGDA2, SOLVERS
