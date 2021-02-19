# Copyright (c) Facebook, Inc. and its affiliates.

from nle.env.base import FULL_ACTIONS, NLE_SPACE_ITEMS
from nle.env.tasks import NetHackStaircase
from nle import nethack

import pkg_resources
import numpy as np
import subprocess
import os
import gym

PATH_DAT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dat")
LIB_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "lib")
PATCH_SCRIPT = os.path.join(
    pkg_resources.resource_filename("nle", "scripts"), "mh_patch_nhdat.sh"
)


class MiniHack(NetHackStaircase):
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
        if des_file is None:
            raise ValueError("Description file is not provided.")

        super().__init__(*args, **kwargs)

        # Patch the nhdat library by compling the given .des file
        self._patch_nhdat(des_file)

        self._scr_descr_index = self._observation_keys.index("screen_descriptions")
        self.observation_space = gym.spaces.Dict(
            {key: space_dict[key] for key in self._minihack_obs_keys}
        )

    def update(self, des_file):
        """Update the current environment by replacing its description file """
        self._patch_nhdat(des_file)

    def _patch_nhdat(self, des_file):
        """Patch the nhdat library. This includes compiling the given
        description file and replacing the new nhdat file in the temporary
        hackdir directory of the environment.
        """
        if not des_file.endswith(".des"):
            fname = "./mylevel.des"
            # If the des-file is passed as a string
            try:
                with open(fname, "w") as f:
                    f.writelines(des_file)
                _ = subprocess.call(
                    [PATCH_SCRIPT, self.env._vardir, nethack.HACKDIR, LIB_DIR]
                )
            except Exception as e:
                print("Something went wrong at level generation", e.args[0])
            finally:
                os.remove(fname)
        else:
            # Use the .des file if exists, otherwise search in minihack directory
            des_path = os.path.abspath(des_file)
            if not os.path.exists(des_path):
                des_path = os.path.abspath(os.path.join(PATH_DAT_DIR, des_file))
            if not os.path.exists(des_path):
                print(
                    "{} file doesn't exist. Please provide a path to a valid .des \
                        file".format(
                        des_path
                    )
                )
            try:
                _ = subprocess.call(
                    [PATCH_SCRIPT, self.env._vardir, nethack.HACKDIR, LIB_DIR, des_path]
                )
            except Exception as e:
                print("Something went wrong at level generation", e.args[0])

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

    def get_direction_obj(self, name, observation=None):
        """Find the game direction of the (first) object in neighboring nine
        tiles that contains given name in its description.
        Return None if not found.
        """
        if observation is None:
            observation = self.last_observation

        neighbors = self.get_neighbor_descriptions(observation)
        for i, tile_description in enumerate(neighbors):
            if name in tile_description:
                return self.index_to_dir_action(i)
        return None

    def get_neighbor_descriptions(self, observation=None):
        """Returns the description of nine neighboring glyphs of the agent."""
        if observation is None:
            observation = self.last_observation
        blstats = observation[self._blstats_index]
        x, y = blstats[:2]

        neighbors = [
            self.get_screen_description(i, j, observation)
            for j in range(y - 1, y + 2)
            for i in range(x - 1, x + 2)
        ]
        return neighbors

    def get_screen_description(self, x, y, observation=None):
        """Returns the description of the screen on (x,y) coordinates."""
        if observation is None:
            observation = self.last_observation

        des_arr = observation[self._scr_descr_index][y, x]
        symb_len = np.where(des_arr == 0)[0][0]

        return des_arr[:symb_len].tobytes().decode("utf-8")
