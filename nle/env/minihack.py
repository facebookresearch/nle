# Copyright (c) Facebook, Inc. and its affiliates.
import enum

# import gym
#
# import numpy as np

# from nle.env import base
from nle import nethack

# from nle.env.base import FULL_ACTIONS, NLE_SPACE_ITEMS
from nle.env.tasks import NetHackScore

# from nle.env.tasks import NetHackScoreFullKeyboard

import subprocess
import os

TASK_ACTIONS = tuple(
    [nethack.MiscAction.MORE]
    + list(nethack.CompassDirection)
    + list(nethack.CompassDirectionLonger)
    + list(nethack.MiscDirection)
    + [nethack.Command.KICK, nethack.Command.EAT, nethack.Command.SEARCH]
)


class MiniHackEmpty(NetHackScore):
    """Environment for "empty" task.

    This environment is an empty room, and the goal of the agent is to reach
    the staircase, which provides a sparse reward.  A small penalty
    is subtracted for the number of steps to reach the goal. This environment
    is useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration.
    """

    class StepStatus(enum.IntEnum):
        ABORTED = -1
        RUNNING = 0
        DEATH = 1
        TASK_SUCCESSFUL = 2

    def __init__(self, *args, **kwargs):

        kwargs["options"] = [
            el
            for el in kwargs.pop("options", list(nethack.NETHACKOPTIONS))
            if not el.startswith("pickup_types")
        ]

        # Select Race and alignment
        kwargs["options"].extend(["role:cav", "race:hum", "align:neu", "gender:mal"])
        # No pet
        kwargs["options"].append("pettype:none")

        level_description = """# NetHack 3.6	oracle.des
#

LEVEL: \"oracle\"

ROOM: \"ordinary\" , lit, (3,3), (center,center), (5,5) {
    STAIR: random, down
    }
"""  # noqa

        fname = "./mylevel.des"
        try:
            with open(fname, "w") as f:
                f.writelines(level_description)
            _ = subprocess.call("nle/scripts/patch_nhdat.sh")
        except Exception as e:
            print("Something went wrong at level generation", e.args[0])
        finally:
            os.remove(fname)

        super().__init__(*args, **kwargs)

    def _is_episode_end(self, observation):
        internal = observation[self._internal_index]
        stairs_down = internal[4]
        if stairs_down:
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_observation, observation, end_status):
        time_penalty = self._get_time_penalty(last_observation, observation)
        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward = 1
        else:
            reward = 0
        return reward + time_penalty
