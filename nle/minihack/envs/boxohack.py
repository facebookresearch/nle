# Copyright (c) Facebook, Inc. and its affiliates.
import os
from tempfile import NamedTemporaryFile

from nle.minihack import MiniHackNavigation
from nle import nethack
from nle.nethack import Command

from gym.envs import registration

MOVE_ACTIONS = tuple(nethack.CompassDirection)
APPLY_ACTIONS = tuple(list(MOVE_ACTIONS) + [Command.PICKUP, Command.APPLY])
NAVIGATE_ACTIONS = tuple(
    list(MOVE_ACTIONS) + [Command.OPEN, Command.KICK, Command.SEARCH]
)

# use fountain as a goal for boulders
BOXOBAN_GOAL_CHAR_ORD = ord("{")
LEVELS_PATH = ".boxoban_levels/"
BOXOBAN_REPO_URL = (
    "https://github.com/deepmind/boxoban-levels/archive/refs/heads/master.zip"
)


def load_boxoban_levels(cur_levels_path):
    levels = []
    for file in os.listdir(cur_levels_path):
        if file.endswith(".txt"):
            with open(os.path.join(cur_levels_path, file)) as f:
                cur_lines = f.readlines()
            cur_level = []
            for el in cur_lines:
                if el != "\n":
                    cur_level.append(el)
                else:
                    # 0th element is a level number, we don't need it
                    levels.append("".join(cur_level[1:]))
                    cur_level = []
    return levels


class BoxoHack(MiniHackNavigation):
    def __init__(self, *args, max_episode_steps=1000, **kwargs):

        level_set = kwargs.get("level_set", "unfiltered")
        level_mode = kwargs.get("level_mode", "train")

        if not os.path.exists(LEVELS_PATH):
            os.mkdir(LEVELS_PATH)

        cur_levels_path = os.path.join(
            LEVELS_PATH, "boxoban-levels-master", level_set, level_mode
        )
        if not os.path.exists(cur_levels_path):
            print("Boxoban levels file not found. Downloading...")
            os.system(
                f"wget -c --read-timeout=5 --tries=0 "
                f'"{BOXOBAN_REPO_URL}" -P {LEVELS_PATH}'
            )
            print("Boxoban levels downloaded, unpacking...")
            import zipfile

            with zipfile.ZipFile(
                os.path.join(LEVELS_PATH, "master.zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(LEVELS_PATH)

        self._levels = load_boxoban_levels(cur_levels_path)
        import random

        level = random.choice(self._levels)
        level = level.split("\n")

        object_strs = []

        level[0] = "-" * len(level[0])
        level[-1] = "-" * len(level[-1])
        for row in range(1, len(level) - 1):
            level[row] = f"|{level[row][1:-1]}|"
            for col in range(len(level[row])):
                if level[row][col] == "$":
                    object_strs.append(f'OBJECT: "boulder", ({col}, {row})')
                if level[row][col] == ".":
                    object_strs.append(f"FOUNTAIN:({col}, {row})")
            if "@" in level[row]:
                py = level[row].index("@")
                level[row] = level[row].replace("@", ".")
                player_str = f"BRANCH:{py, row, py, row},(0,0,0,0)"

            level[row] = level[row].replace(" ", ".")
            level[row] = level[row].replace("#", " ")
            level[row] = level[row].replace("$", ".")

        env_desc = [
            "MAZE: \"mylevel\", ' '",
            "FLAGS: noteleport, hardfloor, premapped",
            "INIT_MAP: solidfill,'#'",
            "GEOMETRY: center, center",
            "MAP",
        ]
        env_desc.extend(level)

        env_desc.append("ENDMAP")
        env_desc.append(player_str)
        env_desc.extend(object_strs)
        f = NamedTemporaryFile(delete=False, suffix=".des")
        with open(f.name, "w") as tmp:
            tmp.write("\n".join(env_desc))
        f.close()
        super().__init__(des_file=f.name, max_episode_steps=max_episode_steps)
        os.unlink(f.name)

    def _is_episode_end(self, observation):
        # If no goals in the observation, all of them are covered with boulders.
        if (
            BOXOBAN_GOAL_CHAR_ORD
            not in observation[self._original_observation_keys.index("chars")]
        ):
            return self.StepStatus.TASK_SUCCESSFUL
        else:
            return self.StepStatus.RUNNING


registration.register(
    id="MiniHack-Boxoban-v0", entry_point="nle.minihack.envs.boxohack:BoxoHack"
)
