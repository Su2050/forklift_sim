"""Forklift Pallet Insert+Lift 任务（direct workflow）。

此文件是任务注册入口点，会在 Isaac Lab 训练脚本导入时被执行：
1) train.py 执行 `import isaaclab_tasks`
2) 触发 `isaaclab_tasks/direct/__init__.py`
3) 进一步导入本模块，执行 `gym.register(...)`

注册完成后，训练脚本可以通过以下 ID 创建环境：
- "Isaac-Forklift-PalletInsertLift-Direct-v0"
"""

import gymnasium as gym

from . import agents  # noqa: F401

# S1.0N: 将 ClampedActorCritic 注册到 rsl_rl.modules 命名空间，
# 使 OnPolicyRunner 的 eval(class_name) 能通过 "rsl_rl.modules.ClampedActorCritic" 找到它。
from .clamped_actor_critic import ClampedActorCritic as _ClampedActorCritic
import rsl_rl.modules as _rsl_modules
_rsl_modules.ClampedActorCritic = _ClampedActorCritic

# 注册 Gym 环境：id 是对外统一入口，entry_point 指向环境类，kwargs 指向配置入口
gym.register(
    id="Isaac-Forklift-PalletInsertLift-Direct-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForkliftPalletInsertLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftInsertLiftPPORunnerCfg",
    },
)
