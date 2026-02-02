"""Forklift Pallet Insert+Lift task (direct workflow).

This module registers the gym environment IDs.
"""

import gymnasium as gym

from . import agents  # noqa: F401

# Register Gym environment.
gym.register(
    id="Isaac-Forklift-PalletInsertLift-Direct-v0",
    entry_point=f"{__name__}.env:ForkliftPalletInsertLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ForkliftPalletInsertLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftInsertLiftPPORunnerCfg",
    },
)
