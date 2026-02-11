"""S1.0N: 带 log_std 下限保护的 ActorCritic 子类。

防止 action std 塌缩到接近确定性策略（如 S1.0M 中 std→0.03），
保留对齐微调所需的最低探索能力。

实现细节：
- 在 _update_distribution 中 clamp log_std，所有下游（log_prob, entropy）
  共享同一个 self.distribution，因此 clamp 自动保证一致性。
- torch.clamp 在越界区间梯度为 0：log_std 参数可能停在 min 以下不回弹，
  但分布实际用的是 min 值。这对防塌缩已足够（目的是兜底，不是推升）。
"""

from __future__ import annotations

import math

import torch
from rsl_rl.modules import ActorCritic

LOG_STD_MIN = math.log(0.05)   # std_min = 0.05, 保守起步
LOG_STD_MAX = math.log(1.5)    # std_max = 1.5, 防爆


class ClampedActorCritic(ActorCritic):
    """ActorCritic with clamped log_std to prevent std collapse.

    仅覆写 _update_distribution，其余逻辑完全继承父类。
    要求 noise_std_type="log" 且 state_dependent_std=False（当前配置）。
    """

    def _update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.actor(obs)
        # 关键：clamp log_std 防止塌缩
        clamped_log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.distribution = torch.distributions.Normal(mean, clamped_log_std.exp())
