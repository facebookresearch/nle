# Copyright (c) Facebook, Inc. and its affiliates.

from nle.env.tasks import NetHackStaircase
from nle.nethack import CompassDirection
from nle.nethack import Command


import subprocess
import os
from shutil import copyfile

PATH_DAT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dat")
MOVE_ACTIONS = tuple(CompassDirection)


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


class MiniHackMaze(NetHackStaircase):
    """Base environment for maze-type task. """

    def __init__(self, *args, des_file: str = None, **kwargs):
        # No pet
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("pettype:none")
        # Actions space - move only
        kwargs["actions"] = kwargs.get("actions", MOVE_ACTIONS)
        # Enter Wizard mode
        kwargs["wizard"] = kwargs.pop("wizard", True)
        # Override episode limit
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)

        # Patch the nhddat library by compling the given .des file
        if des_file is None:
            raise ValueError("Description filename is not provided.")

        if des_file.endswith(".des"):
            patch_nhdat_existing(des_file)
        else:
            patch_nhdat_existing(des_file)

        super().__init__(*args, **kwargs)


class MiniHackEmpty(MiniHackMaze):
    """Environment for "empty" task.

    This environment is an empty room, and the goal of the agent is to reach
    the staircase, which provides a sparse reward.  A small penalty
    is subtracted for the number of steps to reach the goal. This environment
    is useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration.
    """

    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 50)
        super().__init__(*args, des_file="empty.des", **kwargs)


class MiniHackFourRooms(MiniHackMaze):
    """Environment for "four rooms" task.

    Classic four room reinforcement learning environment. The agent must navigate
    in a maze composed of four rooms interconnected by 4 gaps in the walls.
    To obtain a reward, the agent must reach the green goal square. Both the agent
    and the goal square are randomly placed in any of the four rooms.
    """

    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        super().__init__(*args, des_file="four_rooms.des", **kwargs)

    # def reset(self):
    #     wizkit_items = []
    #     _ = super().reset(wizkit_items)
    #     for c in "#wizmap\r":
    #         self.env.step(ord(c))
    #     return self.env._step_return()


class MiniHackMultiRoom(MiniHackMaze):
    """Environment for "multi rooms" task.

    # The agent has to come through multiple rooms with closed (but not locked!)
    # doors to get to the goal: standing on the stairs upwards.
    """

    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        # Multiroom has doors, need Coomand.OPEN here.
        kwargs["actions"] = (*MOVE_ACTIONS, Command.OPEN)
        super().__init__(*args, des_file="multiroom.des", **kwargs)

    # TODO implement logic so that agent always ends up
    #  in a room different from the one with the stairs.
    #  most likely we will need rndcoord for this


class MiniHackLavaCrossing(MiniHackMaze):
    """Environment for "lava crossing" task.

    The agent has to reach the green goal square on the other corner of the room
    while avoiding rivers of deadly lava which terminate the episode in failure.
    Each lava stream runs across the room either horizontally or vertically, and
    has a single crossing point which can be safely used; Luckily, a path to the
    goal is guaranteed to exist. This environment is useful for studying safety
    and safe exploration.
    """

    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        kwargs["wizard"] = False
        super().__init__(*args, des_file="lava_crossing.des", **kwargs)


class MiniHackSimpleCrossing(MiniHackMaze):
    """Environment for "lava crossing" task.

    Similar to the LavaCrossing environment, the agent has to reach the green
    goal square on the other corner of the room, however lava is replaced by
    walls. This MDP is therefore much easier and and maybe useful for quickly
    testing your algorithms.
    """

    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        super().__init__(*args, des_file="simple_crossing.des", **kwargs)
