from collections import OrderedDict
from typing import Tuple, Union

import gym
import nle  # noqa: F401
import numpy as np
from ray.tune.registry import register_env


def create_env(name, observation_keys, *args, **kwargs):
    return gym.make(name, observation_keys=observation_keys, *args, **kwargs)


class RLLibNLEEnv(gym.Env):
    def __init__(self, env_config: dict) -> None:
        # We sort the observation keys so we can create the OrderedDict output
        # in a consistent order
        self._observation_keys = sorted(
            env_config.pop("observation_keys", ("blstats", "glyphs"))
        )
        self.gym_env = create_env(
            env_config.pop("env"), observation_keys=self._observation_keys, **env_config
        )

    @property
    def action_space(self) -> gym.Space:
        return self.gym_env.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self.gym_env.observation_space

    def reset(self) -> dict:
        return self._process_obs(self.gym_env.reset())

    def _process_obs(self, obs: dict) -> dict:
        return OrderedDict({key: obs[key] for key in self._observation_keys})

    def step(
        self, action: Union[int, np.int64]
    ) -> Tuple[dict, Union[np.number, int], Union[np.bool_, bool], dict]:
        obs, reward, done, info = self.gym_env.step(action)
        return self._process_obs(obs), reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


register_env("rllib_nle_env", RLLibNLEEnv)
