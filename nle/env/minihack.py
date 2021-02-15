# Copyright (c) Facebook, Inc. and its affiliates.

from nle.env.tasks import NetHackStaircase
from nle.env.base import FULL_ACTIONS, NLE_SPACE_ITEMS, ASCII_y
from nle import nethack

from shutil import copyfile
import numpy as np
import subprocess
import os
import gym


PATH_DAT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dat")
MOVE_ACTIONS = tuple(nethack.CompassDirection)
APPLY_ACTIONS = tuple(
    list(MOVE_ACTIONS) + [nethack.Command.PICKUP, nethack.Command.APPLY]
)


def patch_nhdat(des_file):
    fname = "./mylevel.des"
    if not des_file.endswith(".des"):
        # If the des-file is passed as a string
        try:
            with open(fname, "w") as f:
                f.writelines(des_file)
            _ = subprocess.call("nle/scripts/patch_nhdat.sh")
        except Exception as e:
            print("Something went wrong at level generation", e.args[0])
        finally:
            os.remove(fname)
    else:
        # Use the .des file if exists, otherwise search in minihack directory
        des_path = os.path.abspath(des_file)
        if not os.path.exists(des_path):
            des_path = os.path.join(PATH_DAT_DIR, des_file)
        if not os.path.exists(des_path):
            print(
                "{} file doesn't exist. Please provide a path to a valid .des \
                    file".format(
                    des_path
                )
            )
        try:
            copyfile(des_path, fname)
            _ = subprocess.call("nle/scripts/patch_nhdat.sh")
        except Exception as e:
            print("Something went wrong at level generation", e.args[0])
        finally:
            os.remove(fname)


class MiniHackCustom(NetHackStaircase):
    """Base class for custom MiniHack environments.

    Features:
    - Default nethack options
    - Full action space by default
    - Wizard mode is turned off by default
    - One-letter menu questions are allowed by default
    - Includes all NLE observations

    The goal is to reach the staircase.

    Use cases:
    - Use this class if you want to experiment with different description files
    and require rich (full) action space.
    - Use a MiniHackMaze class for maze-type environments where there is no pet,
    action space is severely restricted and no one-letter questions are required.
    - Inherit from this class if you require a different reward function and
    dynamics. You might need to override the following methods
        - self._is_episode_end()
        - self._reward_fn()
        - self.step()
        - self.reset()
    """

    def __init__(self, *args, des_file: str = None, **kwargs):
        # No pet
        kwargs["options"] = kwargs.pop("options", list(nethack.NETHACKOPTIONS))
        # Actions space - move only
        kwargs["actions"] = kwargs.pop("actions", FULL_ACTIONS)
        # Enter Wizard mode - turned off by default
        kwargs["wizard"] = kwargs.pop("wizard", False)
        # Allowing one-letter menu questions
        kwargs["allow_all_yn_questions"] = kwargs.pop("allow_all_yn_questions", True)
        # Episode limit
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        # Using all NLE observations by default
        space_dict = dict(NLE_SPACE_ITEMS)
        # Not currently passing the observation keys to the base class
        # because they are used in render(), which is used when developing
        # new environments. Instead, we filter the observations in the
        # _get_observation() method we override.
        self._minihack_obs_keys = kwargs.pop(
            "observation_keys", list(space_dict.keys())
        )

        # Patch the nhddat library by compling the given .des file
        if des_file is None:
            raise ValueError("Description file is not provided.")

        patch_nhdat(des_file)

        super().__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Dict(
            {key: space_dict[key] for key in self._minihack_obs_keys}
        )

    def _get_observation(self, observation):
        # Filter out observations that we don't need
        observation = super()._get_observation(observation)
        return {
            key: val
            for key, val in observation.items()
            if key in self._minihack_obs_keys
        }

    def key_in_inventory(self, name):
        """Returns key of the object in the inventory.

        Arguments:
            name [str]: name of the object
        Returns:
            the key of the first item in the inventory that includes the
            argument name as a substring
        """
        assert "inv_strs" in self._observation_keys
        assert "inv_letters" in self._observation_keys

        inv_strs_index = self._observation_keys.index("inv_strs")
        inv_letters_index = self._observation_keys.index("inv_letters")

        inv_strs = self.last_observation[inv_strs_index]
        inv_letters = self.last_observation[inv_letters_index]

        for letter, line in zip(inv_letters, inv_strs):
            if np.all(line == 0):
                break
            if name in line.tobytes().decode("utf-8"):
                return letter.tobytes().decode("utf-8")

        return None

    def index_to_dir_action(self, index):
        """Returns the ASCII code for direction corresponding to given
        index in reshaped vector of adjacent 9 tiles (None for agent's
        position).
        """
        assert 0 <= index < 9
        index_to_dir_dict = {
            0: ord("y"),
            1: ord("k"),
            2: ord("u"),
            3: ord("h"),
            4: None,
            5: ord("l"),
            6: ord("b"),
            7: ord("j"),
            8: ord("n"),
        }
        return index_to_dir_dict[index]


class MiniHackMaze(MiniHackCustom):
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
        # No random monster generation after every timestep
        # As a workaround to a current issue, we are utilizing the nudist option instead
        kwargs["options"].append("nudist")
        # Actions space - move only
        kwargs["actions"] = kwargs.pop("actions", MOVE_ACTIONS)
        # Disallowing one-letter menu questions
        kwargs["allow_all_yn_questions"] = kwargs.pop("allow_all_yn_questions", False)
        # Override episode limit
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        # Restrict the observation space to glyphs only
        kwargs["observation_keys"] = kwargs.pop("observation_keys", ["glyphs"])

        super().__init__(*args, des_file=des_file, **kwargs)


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


class MiniHackKeyDoor(MiniHackMaze):
    """Environment for "key and door" task.

    This environment has a key that the agent must pick up in order to
    unlock a goal and then get to the green goal square. This environment
    is difficult, because of the sparse reward, to solve using classical
    RL algorithms. It is useful to experiment with curiosity or curriculum
    learning.
    """

    def __init__(self, *args, **kwargs):
        kwargs["options"] = kwargs.pop("options", list(nethack.NETHACKOPTIONS))
        kwargs["options"].append("!autopickup")
        kwargs["character"] = kwargs.pop("charachter", "rog-hum-cha-mal")
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        kwargs["actions"] = APPLY_ACTIONS
        super().__init__(*args, des_file="key_and_door.des", **kwargs)

        self.closed_door_glyph_ids = [2375, 2374]  # this is probably not the best way

    def step(self, action: int):
        # If apply action is chosen
        if self._actions[action] == nethack.Command.APPLY:
            key_key = self.key_in_inventory("key")
            # if key is in the inventory
            if key_key is not None:
                # Get information about adjacent glyphs
                glyphs = self.last_observation[self._glyph_index]
                blstats = self.last_observation[self._blstats_index]
                x, y = blstats[:2]
                neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2].reshape(-1).tolist()
                # Check if there is a closed door nearby
                for index in range(len(neighbors)):
                    if neighbors[index] in self.closed_door_glyph_ids:
                        dir_key = self.index_to_dir_action(index)
                        # Perform the following NetHack steps
                        self.env.step(nethack.Command.APPLY)  # press apply
                        self.env.step(ord(key_key))  # choose key from the inv
                        self.env.step(dir_key)  # select the door's direction
                        self.env.step(ASCII_y)  # press y
                        # TODO the door opens with < 100% probability. We might want to
                        # try this a few times (check if not successfull)
                        dir_action = self._actions.index(dir_key)
                        obs, reward, done, info = super().step(dir_action)
                        return obs, reward, done, info

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
