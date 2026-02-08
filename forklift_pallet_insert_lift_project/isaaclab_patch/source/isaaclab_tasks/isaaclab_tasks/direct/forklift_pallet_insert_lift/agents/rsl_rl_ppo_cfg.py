from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class ForkliftInsertLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Reasonable PPO defaults to get a working policy overnight.

    Tune these once the task is stable:
    - num_envs
    - num_steps_per_env
    - max_iterations
    """

    # runner
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 50
    experiment_name = "forklift_pallet_insert_lift"

    # policy network
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=3.0,  # S1.0: 从 S0.7 的 1.0 提高到 3.0，增强探索
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    # PPO algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.005,  # S1.0c: 从 S0.7 的 0.005 保持不变，防止 noise_std 爆炸
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )

    # optional: clip actions inside wrapper
    clip_actions = 1.0
