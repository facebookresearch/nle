# Copyright (c) Facebook, Inc. and its affiliates.
import enum
import gym

import numpy as np

from nle.env import base
from nle import nethack
from nle.env.base import FULL_ACTIONS, NLE_SPACE_ITEMS

import subprocess
import os

TASK_ACTIONS = tuple(
    [nethack.MiscAction.MORE]
    + list(nethack.CompassDirection)
    + list(nethack.CompassDirectionLonger)
    + list(nethack.MiscDirection)
    + [nethack.Command.KICK, nethack.Command.EAT, nethack.Command.SEARCH]
)


class NetHackScore(base.NLE):
    """Environment for "score" task.

    The task is an augmentation of the standard NLE task. The return function is
    defined as:
    :math:`\text{score}_t - \text{score}_{t-1} + \text{TP}`,
    where the :math:`\text{TP}` is a time penalty that grows with the amount of
    environment steps that do not change the state (such as navigating menus).

    Args:
        penalty_mode (str): name of the mode for calculating the time step
            penalty. Can be ``constant``, ``exp``, ``square``, ``linear``, or
            ``always``. Defaults to ``constant``.
        penalty_step (float): constant applied to amount of frozen steps.
            Defaults to -0.01.
        penalty_time (float): constant applied to amount of frozen steps.
            Defaults to -0.0.

    """

    def __init__(
        self,
        *args,
        penalty_mode="constant",
        penalty_step: float = -0.01,
        penalty_time: float = -0.0,
        **kwargs,
    ):
        self.penalty_mode = penalty_mode
        self.penalty_step = penalty_step
        self.penalty_time = penalty_time

        self._frozen_steps = 0

        actions = kwargs.pop("actions", TASK_ACTIONS)
        super().__init__(*args, actions=actions, **kwargs)

    def _get_time_penalty(self, last_observation, observation):
        blstats_old = last_observation[self._blstats_index]
        blstats_new = observation[self._blstats_index]

        old_time = blstats_old[20]  # moves
        new_time = blstats_new[20]  # moves

        if old_time == new_time:
            self._frozen_steps += 1
        else:
            self._frozen_steps = 0

        penalty = 0
        if self.penalty_mode == "constant":
            if self._frozen_steps > 0:
                penalty += self.penalty_step
        elif self.penalty_mode == "exp":
            penalty += 2 ** self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "square":
            penalty += self._frozen_steps ** 2 * self.penalty_step
        elif self.penalty_mode == "linear":
            penalty += self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "always":
            penalty += self.penalty_step
        else:  # default
            raise ValueError("Unknown penalty_mode '%s'" % self.penalty_mode)
        penalty += (new_time - old_time) * self.penalty_time
        return penalty

    def _reward_fn(self, last_observation, observation, end_status):
        """Score delta, but with added a state loop penalty."""
        score_diff = super()._reward_fn(last_observation, observation, end_status)
        time_penalty = self._get_time_penalty(last_observation, observation)
        return score_diff + time_penalty


class NetHackStaircase(NetHackScore):
    """Environment for "staircase" task.

    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.
    """

    class StepStatus(enum.IntEnum):
        ABORTED = -1
        RUNNING = 0
        DEATH = 1
        TASK_SUCCESSFUL = 2

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


class NetHackStaircasePet(NetHackStaircase):
    """Environment for "staircase-pet" task.

    This task requires the agent to get on top of a staircase down (>), while
    having their pet next to it. See `NetHackStaircase` for the reward function.
    """

    def _is_episode_end(self, observation):
        internal = observation[self._internal_index]
        stairs_down = internal[4]
        if stairs_down:
            glyphs = observation[self._glyph_index]
            blstats = observation[self._blstats_index]
            x, y = blstats[:2]

            neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2].reshape(-1).tolist()
            # TODO: vectorize
            for glyph in neighbors:
                if nethack.glyph_is_pet(glyph):
                    return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


class NetHackOracle(NetHackStaircase):
    """Environment for "oracle" task.

    This task requires the agent to reach the oracle (by standing next to it).
    See `NetHackStaircase` for the reward function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_glyph = None
        for glyph in range(nethack.GLYPH_MON_OFF, nethack.GLYPH_PET_OFF):
            if nethack.permonst(nethack.glyph_to_mon(glyph)).mname == "Oracle":
                self.oracle_glyph = glyph
                break
        assert self.oracle_glyph is not None

    def _is_episode_end(self, observation):
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]
        x, y = blstats[:2]

        neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2]
        if np.any(neighbors == self.oracle_glyph):
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


class NetHackGold(NetHackScore):
    """Environment for the "gold" task.

    The task is similar to the one defined by `NetHackScore`, but the reward
    uses changes in the amount of gold collected by the agent, rather than the
    score.

    The agent will pickup gold automatically by walking on top of it.
    """

    def __init__(self, *args, **kwargs):
        options = kwargs.pop("options", None)

        if options is None:
            # Copy & swap out "pickup_types".
            options = []
            for option in nethack.NETHACKOPTIONS:
                if option.startswith("pickup_types"):
                    options.append("pickup_types:$")
                    continue
                options.append(option)

        super().__init__(*args, options=options, **kwargs)

    def _reward_fn(self, last_observation, observation, end_status):
        """Difference between previous gold and new gold."""
        del end_status  # Unused
        if not self.env.in_normal_game():
            # Before game started or after it ended stats are zero.
            return 0.0

        old_blstats = last_observation[self._blstats_index]
        blstats = observation[self._blstats_index]

        old_gold = old_blstats[13]
        gold = blstats[13]

        time_penalty = self._get_time_penalty(last_observation, observation)

        return gold - old_gold + time_penalty


# FIXME: the way the reward function is currently structured means the
# agents gets a penalty of -1 every other step (since the
# uhunger increases by that)
# thus the step penalty becomes irrelevant
class NetHackEat(NetHackScore):
    """Environment for the "eat" task.

    The task is similar to the one defined by `NetHackScore`, but the reward
    uses positive changes in the character's hunger level (e.g. by consuming
    comestibles or monster corpses), rather than the score.
    """

    def _reward_fn(self, last_observation, observation, end_status):
        """Difference between previous hunger and new hunger."""
        del end_status  # Unused

        if not self.env.in_normal_game():
            # Before game started or after it ended stats are zero.
            return 0.0

        old_internal = last_observation[self._internal_index]
        internal = observation[self._internal_index]

        old_uhunger = old_internal[7]
        uhunger = internal[7]

        reward = max(0, uhunger - old_uhunger)

        time_penalty = self._get_time_penalty(last_observation, observation)

        return reward + time_penalty


class NetHackScout(NetHackScore):
    """Environment for the "scout" task.

    The task is similar to the one defined by `NetHackScore`, but the score is
    defined by the changes in glyphs discovered by the agent.
    """

    def reset(self, *args, **kwargs):
        self.dungeon_explored = {}
        return super().reset(*args, **kwargs)

    def _reward_fn(self, last_observation, observation, end_status):
        del end_status  # Unused

        if not self.env.in_normal_game():
            # Before game started or after it ended stats are zero.
            return 0.0

        reward = 0
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]

        dungeon_num, dungeon_level = blstats[23:25]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != 0)
        explored_old = 0
        if key in self.dungeon_explored:
            explored_old = self.dungeon_explored[key]
        reward = explored - explored_old
        self.dungeon_explored[key] = explored
        time_penalty = self._get_time_penalty(last_observation, observation)
        return reward + time_penalty


NLE_LEVEL_OBS_KEYS = ("episode_goal_str", "episode_goal_glyph")
# different OS might have different messages,
# that's why values are lists here check eat.c for more inspiration
# TODO This is most likely to fail somewhere, need to keep an eye
# on rollouts at the beginning
delicious = lambda x: f"This {x} is delicious"
EDIBLE_GOALS = {
    item: [delicious(item)]
    for item in [
        "orange",
        "meatball",
        "meat ring",
        "meat stick",
        "kelp frond",
        "eucalyptus leaf",
        "clove of garlic",
        "sprig of wolfsbane",
        "carrot",
        "egg",
        "banana",
        "melon",
        "candy bar",
        "lump of royal jelly",
    ]
}

EDIBLE_GOALS.update(
    {
        "apple": ["Delicious!  Must be a Macintosh!", "Core dumped."],
        "pear": ["Core dumped."],
    }
)


class NetHackScoreFullKeyboard(NetHackScore):
    def __init__(self, *args, **kwargs):
        actions = kwargs.pop("actions", FULL_ACTIONS)
        super().__init__(*args, actions=actions, allow_all_yn_questions=True, **kwargs)


def get_object(name):
    for index in range(nethack.NUM_OBJECTS):
        obj = nethack.objclass(index)
        if nethack.OBJ_NAME(obj) == name:
            return obj
    else:
        raise ValueError("'%s' not found!" % name)


class NetHackInventoryManagement(NetHackScoreFullKeyboard):
    """Environment for "eat object" task.
    This task requires the agent to eat a specific item in the inventory.
    The inventory and the goal object can be randomised at each reset.
    The environment stops as soon as the agent eats a specified object.
    See `NetHackStaircase` for the reward function.
    """

    class StepStatus(enum.IntEnum):
        ABORTED = -1
        RUNNING = 0
        DEATH = 1
        TASK_SUCCESSFUL = 2

    def __init__(self, *args, **kwargs):
        self._randomise_goal = kwargs.pop("randomise_goal", True)
        self._randomise_inventory_order = kwargs.pop("randomise_inventory_order", True)
        self._wizkit_list_size = kwargs.pop("wizkit_list_size", 12)
        self._randomise_inventory_selection = False

        self._episode_goal_str = None

        # this will work in wizard mode only because we need to use wizkit
        kwargs["wizard"] = True
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 10)
        actions = kwargs.pop("actions", FULL_ACTIONS)

        super().__init__(*args, actions=actions, **kwargs)

        # update observation space with the goal, we need that to not insert
        # goal_related observations
        # to the base NLE class.
        # In the future, when we can change nle, we can define
        # self._set_obs_space() method
        # and overload it in this class adding goal-related observations.
        space_dict = dict(NLE_SPACE_ITEMS)
        obs_dict = {key: space_dict[key] for key in self._original_observation_keys}

        obs_dict["episode_goal_str"] = gym.spaces.Box(
            low=0,
            high=128,
            shape=(nethack.OBSERVATION_DESC["inv_strs"]["shape"][1],),
            dtype=nethack.OBSERVATION_DESC["inv_strs"]["dtype"],
        )
        obs_dict["episode_goal_glyph"] = gym.spaces.Box(
            low=0,
            high=nethack.MAX_GLYPH,
            shape=(1,),
            dtype=nethack.OBSERVATION_DESC["inv_glyphs"]["dtype"],
        )
        self.observation_space = gym.spaces.Dict(obs_dict)

    def step(self, action: int):
        obs, reward, done, info = super().step(action)
        self._add_goal_to_obs(obs)
        return obs, reward, done, info

    def render(self, mode="human"):
        print(f"Current goal: {self._episode_goal_str}")
        return super().render(mode)

    def reset(self, wizkit_items: list = None, episode_goal: str = None):
        """Sets up the inventory and the goal which will terminate the episode
        when eaten.
        """
        if wizkit_items is None:
            wizkit_items = list(EDIBLE_GOALS.keys())
        wizkit_items = wizkit_items[: self._wizkit_list_size]

        self._episode_goal_str = episode_goal
        if episode_goal is None:
            self._episode_goal_str = (
                np.random.choice(wizkit_items)
                if self._randomise_goal
                else wizkit_items[0]
            )

        if self._randomise_inventory_order:
            np.random.shuffle(wizkit_items)

        if self._randomise_inventory_selection:
            # TODO Implement me
            raise NotImplementedError

        obs = super().reset(wizkit_items=wizkit_items)
        self._add_goal_to_obs(obs)
        return obs

    def _add_goal_to_obs(self, obs):

        key = "episode_goal_str"
        cval = np.zeros(self.observation_space.spaces[key].shape, dtype=np.uint8)
        cvalbytes = bytearray(self._episode_goal_str, "utf-8")
        cval[: len(cvalbytes)] = cvalbytes
        obs.update({key: cval})

        goal_glyph = (
            nethack.GLYPH_OBJ_OFF + get_object(self._episode_goal_str).oc_name_idx
        )
        obs.update({"episode_goal_glyph": np.array([goal_glyph])})
        return obs

    def _is_episode_end(self, observation):
        """if the message contains the target message (e.g. Delicious, must be
        macintosh) for apple, then finish
        """
        possible_goal_msgs = EDIBLE_GOALS[self._episode_goal_str]
        curr_msg = (
            observation[self._original_observation_keys.index("message")]
            .tobytes()
            .decode("utf-8")
        )

        for msg in possible_goal_msgs:
            if msg in curr_msg:
                # TODO Check the key and encoding of the message.
                # Check that it really stops.
                # Write a test which presses e and then apple in the inventory
                # to see that it stops.
                return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_response, response, end_status):
        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward = 100
        elif end_status == self.StepStatus.RUNNING:
            reward = self._get_time_penalty(last_response, response)
        else:  # death or aborted
            reward = -100
        return reward


class NetHackPickAndEat(NetHackInventoryManagement):
    """Environment for "eat object" task.
    This task requires the agent to eat a specific item in the inventory.
    The inventory and the goal object can be randomised at each reset.
    The environment stops as soon as the agent eats a specified object.
    See `NetHackStaircase` for the reward function.
    """

    def __init__(self, *args, **kwargs):
        # TODO item positions in a room are not randomised atm, we should
        # probably walk around the room and drop stuff
        # agent is spawning randomly, so, probably it's not that important
        # at least for now
        # implement this at the reset level
        kwargs["randomise_goal"] = kwargs.pop("randomise_goal", True)
        kwargs["randomise_inventory_order"] = kwargs.pop(
            "randomise_inventory_order", True
        )

        kwargs["wizkit_list_size"] = kwargs.pop("wizkit_list_size", 1)
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_step", 1000)
        # the next line will remove constraints on autopickup types and make
        # the agent collect everything automatically
        kwargs["options"] = [
            el
            for el in kwargs.pop("options", list(nethack.NETHACKOPTIONS))
            if not el.startswith("pickup_types")
        ]
        kwargs["options"].extend(["role:cav", "race:hum", "align:neu", "gender:mal"])
        kwargs["options"].append("pettype:none")

        self.n_distractors = kwargs.pop("inventory_distractors", 0)
        self.randomise_n_distractors = kwargs.pop("randomise_num_distractors", True)

        # TODO this is automated now, but still hardcoded, stay tuned
        level_description = """# NetHack 3.6	oracle.des	
#

LEVEL: \"oracle\"

ROOM: \"ordinary\" , lit, (3,3), (center,center), (5,5) {
    OBJECT:('%',\"orange\"),random
    OBJECT:('%',\"meatball\"),random
    OBJECT:('%',\"meat ring\"),random
    OBJECT:('%',\"meat stick\"),random
    OBJECT:('%',\"kelp frond\"),random
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

    def reset(self, wizkit_items: list = None, episode_goal: str = None):
        """Sets up the inventory and the goal which will terminate the episode
        when eaten.
        """
        if wizkit_items is None:
            wizkit_items = list(EDIBLE_GOALS.keys())
        wizkit_items = wizkit_items[: self._wizkit_list_size]
        if episode_goal is not None:
            print(
                "Be careful, if you provide a goal for an item which is not in \
                        the inventory or on the map, you will not finish a \
                        game."
            )
        self._episode_goal_str = episode_goal
        if episode_goal is None:
            self._episode_goal_str = (
                np.random.choice(wizkit_items)
                if self._randomise_goal
                else wizkit_items[0]
            )

        n_distractors = self.n_distractors
        if self.randomise_n_distractors:
            n_distractors = np.random.randint(n_distractors + 1)
        distractors = [
            el for el in list(EDIBLE_GOALS.keys()) if el != self._episode_goal_str
        ][:n_distractors]

        if self._randomise_inventory_order:
            np.random.shuffle(distractors)

        # not providing wizkit_list as the argument here, otherwise it will
        # just be InventoryManagement task
        # call reset from the Score task, not to put stuff to the wizkit
        obs = super(NetHackScoreFullKeyboard, self).reset(wizkit_items=distractors)
        self._add_goal_to_obs(obs)
        return obs
