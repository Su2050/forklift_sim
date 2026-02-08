"""Forklift Pallet Insert+Lift 的训练器配置入口。

此包用于存放 RL 训练相关的超参数配置（如 PPO runner/algorithm 参数）。
在 `forklift_pallet_insert_lift/__init__.py` 中通过 `from . import agents`
导入，以便 `gym.register()` 的 rsl_rl_cfg_entry_point 能找到配置类。
"""
