"""Forklift Pallet Insert+Lift 的 PPO 训练配置。

此文件提供 RSL-RL 的 runner / policy / algorithm 超参数配置，
由 `forklift_pallet_insert_lift/__init__.py` 中的 `gym.register()` 通过
`rsl_rl_cfg_entry_point` 引用。训练脚本会读取此配置来创建 PPO 训练器。
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ForkliftVisionActorCriticCfg(RslRlPpoActorCriticCfg):
    """ActorCritic config with extra knobs for backbone pretraining transfer."""

    pretrained_backbone_path: str | None = None
    freeze_backbone: bool = False
    freeze_backbone_updates: int = 0
    imagenet_backbone_init: bool = True


@configclass
class ForkliftInsertLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Reasonable PPO defaults to get a working policy overnight.

    Tune these once the task is stable:
    - num_envs
    - num_steps_per_env
    - max_iterations
    """

    # runner：训练调度与基础运行参数
    seed = 42  # 随机种子，保证可复现
    device = "cuda:0"  # 训练设备
    num_steps_per_env = 64  # 每个环境每次 rollout 的步数
    obs_groups = {
        "policy": ["image", "proprio"],
        "critic": ["critic"],
    }
    max_iterations = 2000  # 最大训练迭代次数（iteration）
    save_interval = 50  # 保存模型与日志的间隔（iteration）
    experiment_name = "forklift_pallet_insert_lift"  # 训练实验名称（用于日志目录）

    # policy network：中等强度视觉 actor + 低维 critic
    policy = ForkliftVisionActorCriticCfg(
        class_name="rsl_rl.modules.VisionActorCritic",
        init_noise_std=0.4,
        noise_std_type="log",
        actor_obs_normalization=False,  # 图像已归一化，只保留 proprio encoder
        critic_obs_normalization=True,  # Critic 观测归一化
        actor_hidden_dims=[256, 256, 128],  # fusion actor head
        critic_hidden_dims=[256, 256, 128],  # Critic MLP 隐藏层
        activation="elu",  # 激活函数
        pretrained_backbone_path=None,
        freeze_backbone=False,
        freeze_backbone_updates=0,
        imagenet_backbone_init=True,
    )

    # PPO algorithm：优化与损失相关超参数
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,  # 每次迭代的学习 epoch 数
        num_mini_batches=4,  # 每个 epoch 的小批次数
        learning_rate=3e-4,  # 学习率
        schedule="adaptive",  # 学习率调度策略
        gamma=0.99,  # 折扣因子
        lam=0.95,  # GAE 参数
        entropy_coef=0.0005,  # StageB: 0.0015→0.0005，降低探索压力，精修模式
        desired_kl=0.008,  # StageB: 0.01→0.008，更保守更新步幅
        max_grad_norm=1.0,  # 梯度裁剪阈值
        value_loss_coef=1.0,  # 价值函数损失权重
        use_clipped_value_loss=True,  # 是否裁剪 value loss
        clip_param=0.2,  # PPO clip 系数
    )

    # optional: clip actions inside wrapper
    clip_actions = 1.0  # 动作裁剪范围（[-1, 1]）
