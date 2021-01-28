# Copyright (c) Facebook, Inc. and its affiliates.
# import enum
#
# import gym
#
# import numpy as np


from nle import nethack
from nle.env.tasks import NetHackStaircase

# from nle.env import base
# from nle.env.base import FULL_ACTIONS, NLE_SPACE_ITEMS
# from nle.env.tasks import NetHackScore
# from nle.env.tasks import NetHackScoreFullKeyboard

import subprocess
import os


def replace_nhdat(ascii_descr):
    fname = "./mylevel.des"
    try:
        with open(fname, "w") as f:
            f.writelines(ascii_descr)
        _ = subprocess.call("nle/scripts/patch_nhdat.sh")
    except Exception as e:
        print("Something went wrong at level generation", e.args[0])
    finally:
        os.remove(fname)


class MiniHackEmpty(NetHackStaircase):
    """Environment for "empty" task.

    This environment is an empty room, and the goal of the agent is to reach
    the staircase, which provides a sparse reward.  A small penalty
    is subtracted for the number of steps to reach the goal. This environment
    is useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration.
    """

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
        replace_nhdat(level_description)

        super().__init__(*args, **kwargs)


class MiniHackFourRooms(NetHackStaircase):
    """Environment for "four rooms" task.

    Classic four room reinforcement learning environment. The agent must navigate
    in a maze composed of four rooms interconnected by 4 gaps in the walls.
    To obtain a reward, the agent must reach the green goal square. Both the agent
    and the goal square are randomly placed in any of the four rooms.
    """

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

ROOM: \"ordinary\" , lit, random, random, random {
    STAIR: random, up
    }
    
ROOM: \"ordinary\" , lit, random, random, random {
    STAIR: random, down
    }
    
ROOM: \"ordinary\" , lit, random, random, random {
    }
    
ROOM: \"ordinary\" , lit, random, random, random {
    }
    
    RANDOM_CORRIDORS
"""  # noqa
        replace_nhdat(level_description)

        super().__init__(*args, **kwargs)
