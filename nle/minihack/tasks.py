# Copyright (c) Facebook, Inc. and its affiliates.
import os
from tempfile import NamedTemporaryFile

from nle.minihack import MiniHack
from nle import nethack
from nle.nethack import Command


MOVE_ACTIONS = tuple(nethack.CompassDirection)
APPLY_ACTIONS = tuple(list(MOVE_ACTIONS) + [Command.PICKUP, Command.APPLY])
NAVIGATE_ACTIONS = tuple(
    list(MOVE_ACTIONS) + [Command.OPEN, Command.KICK, Command.SEARCH]
)


class MiniHackMaze(MiniHack):
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
        kwargs["observation_keys"] = kwargs.pop("observation_keys", ["chars_crop"])

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


class MiniHackCorridor(MiniHackMaze):
    """Environment for "corridor" task.

    The agent has to navigate itself through randomely generated corridors that
    connect several rooms and find the goal.
    """

    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)
        kwargs["actions"] = NAVIGATE_ACTIONS
        super().__init__(*args, des_file="corridor.des", **kwargs)


class MiniHackMazeWalk(MiniHack):
    """Environment for "mazewalk" task.

    TODO understand how can remove monsters and objects from the maze
    """

    def __init__(self, *args, **kwargs):
        kwargs["options"] = kwargs.pop("options", list(nethack.NETHACKOPTIONS))
        kwargs["options"].append("pettype:none")
        kwargs["options"].append("nudist")
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)
        super().__init__(*args, des_file="mazewalk.des", **kwargs)


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

    def step(self, action: int):
        # If apply action is chosen
        if self._actions[action] == Command.APPLY:
            key_key = self.key_in_inventory("key")
            # if key is in the inventory
            if key_key is not None:
                # Check if there is a closed door nearby
                dir_key = self.get_direction_obj("closed door")
                if dir_key is not None:
                    # Perform the following NetHack steps
                    self.env.step(Command.APPLY)  # press apply
                    self.env.step(ord(key_key))  # choose key from the inv
                    self.env.step(dir_key)  # select the door's direction
                    obs, done = self.env.step(ord("y"))  # press y
                    obs, done = self._perform_known_steps(obs, done, exceptions=True)
                    # Make sure the door is open
                    while True:
                        obs, done = self.env.step(dir_key)
                        obs, done = self._perform_known_steps(
                            obs, done, exceptions=True
                        )
                        if self.get_direction_obj("closed door", obs) is None:
                            break

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info


class MiniGridHackMultiroom(MiniHackMaze):
    def __init__(self, *args, **kwargs):
        import gym
        import gym_minigrid  # noqa: F401

        self._minigrid_env = gym.make(kwargs["env_name"])

        env_desc = self.get_env_desc()
        f = NamedTemporaryFile(delete=False, suffix=".des")
        with open(f.name, "w") as tmp:
            tmp.write("\n".join(env_desc))
        f.close()
        super().__init__(des_file=f.name)
        os.unlink(f.name)

    def get_env_desc(self):
        self._minigrid_env.reset()
        env = self._minigrid_env

        env_desc = [
            "MAZE: \"mylevel\", ' '",
            "FLAGS: premapped",
            "INIT_MAP: solidfill, ' '",
            "GEOMETRY: center, center",
            "MAP",
        ]

        door_positions = []
        player_pos = (env.agent_pos[0], env.agent_pos[1])
        goal_position = None
        empty_strs = 0
        empty_str = True
        for j in range(env.grid.height):
            str = ""
            for i in range(env.width):
                c = env.grid.get(i, j)
                if c is None:
                    str += "."
                    continue
                empty_str = False
                if c.type == "wall":
                    str += "|"
                if c.type == "door":
                    str += "+"
                    door_positions.append((i, j - empty_strs))
                    continue
                if c.type == "floor":
                    str += "."
                    continue
                if c.type == "goal":
                    goal_position = (i, j - empty_strs)
                    str += "."
                    continue
                if c.type == "player":
                    str += "."
                    continue
            if not empty_str and j < env.grid.height - 1:
                if set(str) != {"."}:
                    str = str.replace(".", " ", str.index("|"))
                    inv = str[::-1]
                    str = inv.replace(".", " ", inv.index("|"))[::-1]
                    env_desc.append(str)
            elif empty_str:
                empty_strs += 1
        env_desc.append("ENDMAP")
        env_desc.extend([f"DOOR: closed, {d}" for d in door_positions])
        env_desc.append(f"STAIR: {goal_position}, down")
        ppx = player_pos[0]
        ppy = player_pos[1]
        env_desc.append(
            f"BRANCH:" f"{ppx, ppy-empty_strs, ppx, ppy-empty_strs}" f",(0,0,0,0)"
        )
        env_desc.append("WALLIFY")
        return env_desc

    def get_env_desc_slow(self):

        self._minigrid_env.reset()
        env = self._minigrid_env

        env_desc = [
            "MAZE: \"mylevel\", ' '",
            "FLAGS: premapped",
            "INIT_MAP: solidfill, ' '",
            "GEOMETRY: center, center",
            "MAP",
        ]

        mg_level = env.__str__().split("\n")
        mg_level = [el for el in mg_level if el.strip() != ""]
        mg_level[0] = mg_level[0].replace("WG", "|")
        mg_level[-1] = mg_level[-1].replace("WG", "|")
        mg_level[0] = mg_level[0].replace(" ", "|")
        mg_level[-1] = mg_level[-1].replace(" ", "|")

        player_chars = [">>", "<<", "VV", "^^"]
        COLS = ["R", "G", "B", "P", "Y", "G"]
        door_chars = [f"D{c}" for c in COLS]

        door_strs = []
        for i in range(1, len(mg_level) - 1):
            for pc in player_chars:
                if pc in mg_level[i]:
                    pcx = (mg_level[i].index(pc) + 1) // 2
                    player_str = f"BRANCH:{pcx, i, pcx, i},(0,0,0,0)"
                    break

            if "GG" in mg_level[i]:
                stair_str = f"STAIR: {((mg_level[i].index('GG') + 1) // 2, i)}, down"
            for dc in door_chars:
                if dc in mg_level[i]:
                    door_strs.append(
                        f"DOOR: closed, {(mg_level[i].index(dc) + 1) // 2, i}"
                    )

            for pc in player_chars:
                mg_level[i] = mg_level[i].replace(pc, ".")
            for dc in door_chars:
                mg_level[i] = mg_level[i].replace(dc, "+")

            mg_level[i] = mg_level[i].replace("  ", ".")
            mg_level[i] = mg_level[i].replace("WG", "|")
            mg_level[i] = mg_level[i].replace("GG", ".")
            mg_level[i] = mg_level[i].replace(".", " ", mg_level[i].index("|"))
            inv = mg_level[i][::-1]
            mg_level[i] = inv.replace(".", " ", inv.index("|"))[::-1]

        env_desc.extend(mg_level)
        env_desc.append("ENDMAP")
        env_desc.extend(door_strs)
        env_desc.append(stair_str)
        env_desc.append(player_str)
        return env_desc

    def reset(self):
        env_desc = self.get_env_desc()
        f = NamedTemporaryFile(delete=False, suffix=".des")
        with open(f.name, "w") as tmp:
            tmp.write("\n".join(env_desc))
        f.close()
        self.update(f.name)
        os.unlink(f.name)
        return super().reset()


# use fountain as a goal for boulders
BOXOBAN_GOAL_CHAR_ORD = ord("{")


class BoxoHack(MiniHackMaze):
    def __init__(self, *args, max_episode_steps=1000, **kwargs):

        level = (
            "##########\n##########\n#######  #\n### .# #.#\n#     .  #\n"
            "# #  $ $ #\n# #####  #\n##### $$.#\n### @    #\n##########"
        )
        print(level)
        level = level.split("\n")
        # nethack does not like when there are two rows of the walls
        # using # for solidfill gets rid of this issue
        # consequtive_walls_st = 0
        # for el in level:
        #     if set(el) == {'#'}:
        #         consequtive_walls_st+=1
        #     else:
        #         break
        # if consequtive_walls_st > 1:
        #     level = level[consequtive_walls_st-1:]
        # consequtive_walls = 0
        # for el in level[::-1]:
        #     if set(el) == {'#'}:
        #         consequtive_walls += 1
        #     else:
        #         break
        # if consequtive_walls > 1:
        #     level = level[:1-consequtive_walls:]
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
