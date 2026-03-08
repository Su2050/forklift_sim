from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation

LOG_STD_MIN = math.log(0.05)
LOG_STD_MAX = math.log(1.5)


class VisionActorCritic(nn.Module):
    """视觉 actor + 低维 critic。

    约定输入结构：
    - obs["policy"]["image"]   : (N, 3, H, W)
    - obs["policy"]["proprio"] : (N, P)
    - obs["critic"]            : (N, C)
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "VisionActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.noise_std_type = noise_std_type

        activation_mod = resolve_nn_activation(activation)
        sample_image, sample_proprio = self._extract_policy_terms(obs)
        sample_critic = obs["critic"]

        proprio_dim = int(sample_proprio.shape[-1])
        critic_dim = int(sample_critic.shape[-1])

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            activation_mod,
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            activation_mod,
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            activation_mod,
            nn.Flatten(),
        )

        with torch.no_grad():
            image_feat_dim = int(self.image_encoder(sample_image[:1].detach().cpu()).shape[-1])

        self.image_proj = nn.Sequential(
            nn.Linear(image_feat_dim, 256),
            resolve_nn_activation(activation),
            nn.Linear(256, 256),
            resolve_nn_activation(activation),
        )
        self.proprio_encoder = MLP(proprio_dim, 128, [128, 128], activation)
        self.actor = MLP(256 + 128, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(critic_dim, 1, critic_hidden_dims, activation)

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(proprio_dim)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(critic_dim)
        else:
            self.critic_obs_normalizer = nn.Identity()

        print(f"Vision image encoder: {self.image_encoder}")
        print(f"Vision image projection: {self.image_proj}")
        print(f"Vision proprio encoder: {self.proprio_encoder}")
        print(f"Actor head: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _ensure_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor with 4 dims, got shape={tuple(image.shape)}")
        if image.shape[1] == 3:
            return image.float()
        if image.shape[-1] == 3:
            return image.permute(0, 3, 1, 2).float()
        raise ValueError(f"Unexpected image shape={tuple(image.shape)}")

    def _extract_policy_terms(self, obs):
        if "image" in obs.keys() and "proprio" in obs.keys():
            return obs["image"], obs["proprio"]

        policy_obs = obs["policy"]
        if hasattr(policy_obs, "keys") and "image" in policy_obs.keys() and "proprio" in policy_obs.keys():
            return policy_obs["image"], policy_obs["proprio"]

        raise KeyError("Cannot resolve policy image/proprio from observations")

    def _encode_policy_obs(self, obs):
        image, proprio = self._extract_policy_terms(obs)
        image = self._ensure_image_tensor(image)
        if image.max() > 1.0:
            image = image / 255.0
        image = torch.clamp(image, 0.0, 1.0)

        proprio = proprio.float()
        proprio = self.actor_obs_normalizer(proprio)

        image_feat = self.image_proj(self.image_encoder(image))
        proprio_feat = self.proprio_encoder(proprio)
        return torch.cat([image_feat, proprio_feat], dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(self._encode_policy_obs(obs))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            clamped_log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(clamped_log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(self._encode_policy_obs(obs))

    def evaluate(self, obs, **kwargs):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actor_obs(self, obs):
        return self._encode_policy_obs(obs)

    def get_critic_obs(self, obs):
        return obs["critic"].float()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            _, proprio = self._extract_policy_terms(obs)
            self.actor_obs_normalizer.update(proprio.float())
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
