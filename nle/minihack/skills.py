# Copyright (c) Facebook, Inc. and its affiliates.
from nle.nethack.actions import MiscDirection, CompassDirection, CompassDirectionLonger
from nle.env.base import FULL_ACTIONS
from nle.minihack import MiniHack, LevelGenerator
from nle.minihack.actions import InventorySelection, ACTION_STR_DICT
from nle.nethack import Command, CompassIntercardinalDirection
import enum
import string
import numpy as np

FULL_ACTIONS_INV = tuple(list(FULL_ACTIONS) + list(InventorySelection))

COMESTIBLES = [
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


def delicious(x):
    return f"This {x} is delicious"


EDIBLE_GOALS = {item: [delicious(item)] for item in COMESTIBLES}
EDIBLE_GOALS.update(
    {
        "apple": ["Delicious!  Must be a Macintosh!", "Core dumped."],
        "pear": ["Core dumped."],
    }
)


def wielded(x):
    return [f"{x} welds itself to your hand!", f"{x} (weapon in hand)"]


def wearing(x):
    return [f"You are now wearing a {x}"]


class GoalEvent(enum.IntEnum):
    MESSAGE = 0
    LOC_ACTION = 1
    NAVIGATION = 2


class MiniHackSkill(MiniHack):
    """Base environment for single skill acquisition tasks."""

    def __init__(
        self,
        *args,
        des_file,
        goal_msgs=None,
        goal_loc_action=None,
        inv_actions=False,
        **kwargs,
    ):
        """If goal_msgs == None, the goal is to reach the staircase."""
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("pettype:none")
        kwargs["options"].append("!autopickup")
        kwargs["character"] = kwargs.pop("charachter", "cav-hum-new-mal")
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        self._no_rand_mon()

        self.inv_actions = inv_actions
        if self.inv_actions:
            kwargs["actions"] = kwargs.pop("actions", FULL_ACTIONS_INV)
            self._inventory_map = {}
            self.action_names = None

        if goal_msgs is not None:
            self.goal_msgs = goal_msgs
            self.goal_event = GoalEvent.MESSAGE
        elif goal_loc_action is not None:
            if not isinstance(goal_loc_action, tuple) or len(goal_loc_action) != 2:
                raise AttributeError("goal_loc_action must be a tuple of strings")
            try:
                self.goal_action = Command[goal_loc_action[1].upper()]
            except KeyError:
                raise KeyError(
                    "Action {} is not in the action space.".format(
                        goal_loc_action[0].upper()
                    )
                )

            self.goal_loc = goal_loc_action[0].lower()
            # TODO check if goal_loc is there in the begining
            self.goal_event = GoalEvent.LOC_ACTION
        else:
            self.goal_event = GoalEvent.NAVIGATION

        default_keys = [
            "chars_crop",
            "colors_crop",
            "screen_descriptions_crop",
            "message",
            "inv_strs",
        ]
        if not self.inv_actions:
            default_keys.append("inv_letters")

        kwargs["observation_keys"] = kwargs.pop("observation_keys", default_keys)
        super().__init__(*args, des_file=des_file, **kwargs)

    def reset(self, *args, **kwargs):
        if self.goal_event == GoalEvent.LOC_ACTION:
            self.loc_action_check = False
            self.action_confirmed = False
        return super().reset(*args, **kwargs)

    def step(self, action: int):
        if self.inv_actions:
            internal = self.last_observation[self._internal_index]
            in_yn_function = internal[1]

            # Check if action not allowed and replace with WAIT
            nle_action = self._actions[action]
            if (in_yn_function and not isinstance(nle_action, InventorySelection)) or (
                not in_yn_function and isinstance(nle_action, InventorySelection)
            ):
                action = self._actions.index(MiscDirection.WAIT)

        if self.goal_event == GoalEvent.LOC_ACTION:
            if self._actions[action] == self.goal_action and self._standing_on_top(
                self.goal_loc
            ):
                self.loc_action_check = True
            elif (
                self.loc_action_check
                and self._actions[action] == CompassIntercardinalDirection.NW
            ):
                self.action_confirmed = True
            else:
                self.loc_action_check = False

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def _is_episode_end(self, observation):
        """Finish if the message contains the target message. """
        if self.goal_event == GoalEvent.MESSAGE:
            curr_msg = (
                observation[self._original_observation_keys.index("message")]
                .tobytes()
                .decode("utf-8")
            )

            for msg in self.goal_msgs:
                if msg in curr_msg:
                    return self.StepStatus.TASK_SUCCESSFUL
            return self.StepStatus.RUNNING

        elif self.goal_event == GoalEvent.LOC_ACTION:
            if self.action_confirmed:
                return self.StepStatus.TASK_SUCCESSFUL
            return self.StepStatus.RUNNING

        else:
            internal = observation[self._internal_index]
            stairs_down = internal[4]
            if stairs_down:
                return self.StepStatus.TASK_SUCCESSFUL
            return self.StepStatus.RUNNING

    def _standing_on_top(self, name):
        """Returns whether the agents is standing on top of the given object.
        The object name (e.g. altar, sink, fountain) must exist on the map.
        The function will return True if the object name is not in the screen
        descriptions (with agent info taking the space of the corresponding
        tile rather than the object).
        """
        return not self.screen_contains(name)

    def _get_observation(self, observation):
        # Add language-related observations
        observation = super()._get_observation(observation)

        if self.inv_actions:
            observation = self._update_inventory(observation)

        if self.goal_event == GoalEvent.LOC_ACTION and self._standing_on_top(
            self.goal_loc
        ):
            observation = self._update_screen_description(observation)

        return observation

    def _update_screen_description(self, observation):
        if "screen_descriptions_crop" in self._minihack_obs_keys:
            # Chaning only the middle description (agent's location)
            x = self.obs_crop_w // 2
            y = self.obs_crop_h // 2
            length = len(observation["screen_descriptions_crop"][x, y])
            observation["screen_descriptions_crop"][x, y] = self.str_to_arr(
                self.goal_loc, length
            )
        if "screen_descriptions" in self._minihack_obs_keys:
            # TODO check this is correct
            blstats = self.last_observation[self._blstats_index]
            x, y = blstats[:2]
            length = len(observation["screen_descriptions_crop"][x, y])
            observation["screen_descriptions"][x, y] = self.str_to_arr(
                self.goal_loc, length
            )

        return observation

    def _update_inventory(self, observation):
        """Updates the inventory map."""
        inv_strs_index = self._observation_keys.index("inv_strs")
        inv_letters_index = self._observation_keys.index("inv_letters")

        inv_strs = self.last_observation[inv_strs_index]
        inv_letters = self.last_observation[inv_letters_index]

        letters = [letter.tobytes().decode("utf-8") for letter in inv_letters]

        # This new vector will be used as inventory observation
        inv_strs_new = np.zeros(inv_strs.shape, dtype=inv_strs.dtype)

        for i, char in enumerate(["$"] + list(string.ascii_letters)):
            if char in letters:
                self._inventory_map[char] = i
                old_i = letters.index(char)
                inv_strs_new[i] = inv_strs[old_i]
            else:
                self._inventory_map[char] = -1

        # Remove old inv_strs from obs and add the new one
        observation.pop("inv_strs")
        observation["inv_strs"] = inv_strs_new

        return observation

    # TODO Rewrite get_action_names completely
    def get_action_names(self, observation=None):
        if self.inv_actions:
            return self.get_action_names_fixed()

        ret_val = np.ndarray((len(self._actions), 20), dtype=np.int8)
        for i in range(len(self._actions)):
            ret_val[i, :] = self.str_to_arr(self.action_to_name(i, observation), 20)

        return ret_val

    def get_action_names_fixed(self, observation=None):
        if self.action_names is None:
            self.action_names = np.ndarray((len(self._actions), 20), dtype=np.int8)
            for i in range(len(self._actions)):
                self.action_names[i, :] = self.str_to_arr(
                    self.action_to_name(i, observation), 20
                )

        return self.action_names

    def print_action_names(self, observation):
        names = self.get_action_names(observation)
        str_val = ""
        for number, letter in enumerate(names):
            str_val += " " + str((number, letter))
        print(str_val)

    @staticmethod
    def arr_to_str(arr):
        arr = arr[np.where(arr != 0)]
        arr = bytes(arr).decode("utf-8")
        return arr

    @staticmethod
    def str_to_arr(name, length):
        pad_len = length - len(name)
        byte_arr = bytearray()
        byte_arr.extend(map(ord, name))
        byte_arr += bytearray(pad_len)
        return np.array(byte_arr, dtype=np.uint8)

    def action_to_name(self, action, observation=None):
        # TODO add a flag to also use longer description of action names
        if self.inv_actions:
            assert observation is not None
            in_yn_function = self.last_observation[self._internal_index][1]
            # If in YN_function
            if in_yn_function:
                if isinstance(self._actions[action], InventorySelection):
                    # If inventory action: corresponding items descriptions are returned
                    inv_map_index = action - len(FULL_ACTIONS) + 1
                    key = list(self._inventory_map.keys())[inv_map_index]
                    inv_item_index = self._inventory_map[key]
                    description = observation["inv_strs"][inv_item_index]
                    description_str = self.arr_to_str(description)
                    return description_str
                else:
                    # For regular actions, an empty string is returned
                    return ""
            else:
                # if NOT in YN-function
                if isinstance(self._actions[action], InventorySelection):
                    return ""

        if isinstance(self._actions[action], CompassDirection) or isinstance(
            self._actions[action], CompassDirectionLonger
        ):
            return ACTION_STR_DICT[self._actions[action].name]

        return self._actions[action].name.lower()


class MiniHackEat(MiniHackSkill):
    """Environment for "eat" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("apple", "%")

        goal_msgs = EDIBLE_GOALS["apple"]

        super().__init__(
            *args, des_file=lvl_gen.get_des(), goal_msgs=goal_msgs, **kwargs
        )


class MiniHackPray(MiniHackSkill):
    """Environment for "pray" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_altar("random", "neutral", "altar")

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            goal_loc_action=("altar", "pray"),
            **kwargs,
        )


class MiniHackSink(MiniHackSkill):
    """Environment for "sink" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_sink()

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            goal_loc_action=("sink", "quaff"),
            **kwargs,
        )


# class MiniHackQuaff(MiniHackSkill):
#     """Environment for "quaff" task."""

#     def __init__(self, *args, **kwargs):
#         lvl_gen = LevelGenerator(x=5, y=5, lit=True)
#         lvl_gen.add_object("gain level", "!", cursestate="cursed")

#         super().__init__(
#             *args,
#             des_file=lvl_gen.get_des(),
#             goal_msgs=["This seems like acid."],
#             **kwargs
#         )


class MiniHackClosedDoor(MiniHackSkill):
    """Environment for "open" task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="closed_door.des", **kwargs)


class MiniHackLockedDoor(MiniHackSkill):
    """Environment for "kick" task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="locked_door.des", **kwargs)


class MiniHackWield(MiniHackSkill):
    """Environment for "wield" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("dagger", ")")

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            goal_msgs=wielded("dagger"),
            **kwargs,
        )


class MiniHackWear(MiniHackSkill):
    """Environment for "wear" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("robe", "[")

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            goal_msgs=wearing("robe"),
            **kwargs,
        )


class MiniHackTakeOff(MiniHackSkill):
    """Environment for "take off" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("leather jacket", "[")

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            goal_msgs=wearing("leather jacket"),
            **kwargs,
        )


class MiniHackPutOn(MiniHackSkill):
    """Environment for "put on" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("amulet of life saving", '"')

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            goal_msgs=["amulet (being worn)."],
            **kwargs,
        )


class MiniHackZap(MiniHackSkill):
    """Environment for "zap" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("enlightenment", "/")

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            goal_msgs=["The feeling subsides."],
            **kwargs,
        )


class MiniHackRead(MiniHackSkill):
    """Environment for "read" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("blank paper", "?")

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            goal_msgs=["This scroll seems to be blank."],
            **kwargs,
        )
