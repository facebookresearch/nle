# Copyright (c) Facebook, Inc. and its affiliates.

from nle.minihack import MiniHack
from nle import nethack
from gym.envs import registration


MOVE_ACTIONS = tuple(nethack.CompassDirection)


class MiniHackNavigation(MiniHack):
    """Base class for maze-type task.

    Maze environments have
    - Restricted action space (move only by default)
    - No pet
    - One-letter menu questions are NOT allowed by default
    - Restricted observations, only glyphs by default
    - No random monster generation

    The goal is to reach the staircase.
    """

    def __init__(self, *args, des_file: str = None, **kwargs):
        # No pet
        kwargs["options"] = kwargs.pop("options", list(nethack.NETHACKOPTIONS))
        kwargs["options"].append("pettype:none")
        # Actions space - move only
        kwargs["actions"] = kwargs.pop("actions", MOVE_ACTIONS)
        # Disallowing one-letter menu questions
        kwargs["allow_all_yn_questions"] = kwargs.pop("allow_all_yn_questions", False)
        # Override episode limit
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        # Restrict the observation space to glyphs only
        kwargs["observation_keys"] = kwargs.pop("observation_keys", ["chars_crop"])
        # No random monster generation after every timestep
        self._no_rand_mon()

        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackMazeWalk(MiniHackNavigation):
    """Environment for "mazewalk" task."""

    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)
        self._no_rand_mon()
        super().__init__(*args, des_file="mazewalk.des", **kwargs)


registration.register(
    id="MiniHack-MazeWalk-v0",
    entry_point="nle.minihack.navigation:MiniHackMazeWalk",
)
