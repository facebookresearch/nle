# Copyright (c) Facebook, Inc. and its affiliates.
import os
import random

from nle.minihack import MiniHackNavigation, LevelGenerator

from gym.envs import registration

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


def download_boxoban_levels():
    print("Boxoban levels file not found. Downloading...")
    os.system(
        f"wget -c --read-timeout=5 --tries=0 " f'"{BOXOBAN_REPO_URL}" -P {LEVELS_PATH}'
    )
    print("Boxoban levels downloaded, unpacking...")
    import zipfile

    with zipfile.ZipFile(os.path.join(LEVELS_PATH, "master.zip"), "r") as zip_ref:
        zip_ref.extractall(LEVELS_PATH)


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
            download_boxoban_levels()

        self._levels = load_boxoban_levels(cur_levels_path)

        level = random.choice(self._levels)
        level = level.split("\n")
        map, info = self.get_env_map(level)
        flags = kwargs.get("flags", [])
        flags.append("noteleport")
        flags.append("premapped")
        lvl_gen = LevelGenerator(map=map, lit=True, flags=flags, solidfill="#")
        for b in info["boulders"]:
            lvl_gen.add_boulder(b)
        for f in info["fountains"]:
            lvl_gen.add_fountain(f)
        lvl_gen.add_stair_up(info["player"])
        super().__init__(*args, des_file=lvl_gen.get_des(), **kwargs)

    def get_env_map(self, level):
        info = {"fountains": [], "boulders": []}
        level[0] = "-" * len(level[0])
        level[-1] = "-" * len(level[-1])
        for row in range(1, len(level) - 1):
            level[row] = f"|{level[row][1:-1]}|"
            for col in range(len(level[row])):
                if level[row][col] == "$":
                    info["boulders"].append((col, row))
                if level[row][col] == ".":
                    info["fountains"].append((col, row))
            if "@" in level[row]:
                py = level[row].index("@")
                level[row] = level[row].replace("@", ".")
                info["player"] = (py, row)
            level[row] = level[row].replace(" ", ".")
            level[row] = level[row].replace("#", " ")
            level[row] = level[row].replace("$", ".")
        return "\n".join(level), info

    def _is_episode_end(self, observation):
        # If no goals in the observation, all of them are covered with boulders.
        if (
            ord("{")  # use fountain as a goal for boulders
            not in observation[self._original_observation_keys.index("chars")]
        ):
            return self.StepStatus.TASK_SUCCESSFUL
        else:
            return self.StepStatus.RUNNING


registration.register(
    id="MiniHack-Boxoban-v0", entry_point="nle.minihack.envs.boxohack:BoxoHack"
)
