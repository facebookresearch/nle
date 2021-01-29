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
from shutil import copyfile

PATH_DAT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dat")


def patch_nhdat(level_des):
    fname = "./mylevel.des"
    try:
        with open(fname, "w") as f:
            f.writelines(level_des)
        _ = subprocess.call("nle/scripts/patch_nhdat.sh")
    except Exception as e:
        print("Something went wrong at level generation", e.args[0])
    finally:
        os.remove(fname)


def patch_nhdat_existing(des_name):
    try:
        des_path = os.path.join(PATH_DAT_DIR, des_name)
        if not os.path.exists(des_path):
            print(
                "{} file doesn't exist. Please provide a path to a valid .des \
                    file".format(
                    des_path
                )
            )
        fname = "./mylevel.des"
        copyfile(des_path, fname)
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

        patch_nhdat_existing("empty.des")

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

        patch_nhdat_existing("four_rooms.des")

        super().__init__(*args, **kwargs)
