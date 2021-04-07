# Copyright (c) Facebook, Inc. and its affiliates.
from nle.minihack import MiniHack, LevelGenerator, action_to_str
from nle.nethack import Command, CompassIntercardinalDirection
import enum

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

    def __init__(self, *args, des_file, goal_msgs=None, goal_loc_action=None, **kwargs):
        """If goal_msgs == None, the goal is to reach the staircase."""
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("pettype:none")
        kwargs["options"].append("nudist")
        kwargs["options"].append("!autopickup")
        kwargs["character"] = kwargs.pop("charachter", "cav-hum-new-mal")
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)

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

            self.goal_loc = goal_loc_action = goal_loc_action[0].lower()
            # TODO check if goal_loc is there in the begining
            self.goal_event = GoalEvent.LOC_ACTION
        else:
            self.goal_event = GoalEvent.NAVIGATION

        default_keys = [
            "chars_crop",
            "colors_crop",
            "screen_descriptions_crop",
            "inv_strs",
            "inv_letters",
        ]
        kwargs["observation_keys"] = kwargs.pop("observation_keys", default_keys)
        super().__init__(*args, des_file=des_file, **kwargs)

    def reset(self, *args, **kwargs):
        if self.goal_event == GoalEvent.LOC_ACTION:
            self.loc_action_check = False
            self.action_confirmed = False
        return super().reset(*args, **kwargs)

    def step(self, action: int):
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

    def get_action_names(self):
        return [action_to_str(a) for a in self._actions]


class MiniHackEat(MiniHackSkill):
    """Environment for "eat" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(x=5, y=5, lit=True)
        lvl_gen.add_object("apple", "%")

        goal_msgs = EDIBLE_GOALS["apple"]

        super().__init__(
            *args, des_file=lvl_gen.get_des(), goal_msgs=goal_msgs, **kwargs
        )


class MiniHackPray(MiniHackSkill):
    """Environment for "pray" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(x=5, y=5, lit=True)
        lvl_gen.add_altar()

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            goal_loc_action=("altar", "pray"),
            **kwargs,
        )


class MiniHackSink(MiniHackSkill):
    """Environment for "sink" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(x=5, y=5, lit=True)
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
        lvl_gen = LevelGenerator(x=5, y=5, lit=True)
        lvl_gen.add_object("dagger", ")")

        super().__init__(
            *args, des_file=lvl_gen.get_des(), goal_msgs=wielded("dagger"), **kwargs
        )


class MiniHackWear(MiniHackSkill):
    """Environment for "wear" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(x=5, y=5, lit=True)
        lvl_gen.add_object("robe", "[")

        super().__init__(
            *args, des_file=lvl_gen.get_des(), goal_msgs=wearing("robe"), **kwargs
        )


class MiniHackTakeOff(MiniHackSkill):
    """Environment for "take off" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(x=5, y=5, lit=True)
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
        lvl_gen = LevelGenerator(x=5, y=5, lit=True)
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
        lvl_gen = LevelGenerator(x=5, y=5, lit=True)
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
        lvl_gen = LevelGenerator(x=5, y=5, lit=True)
        lvl_gen.add_object("blank paper", "?")

        super().__init__(
            *args,
            des_file=lvl_gen.get_des(),
            goal_msgs=["This scroll seems to be blank."],
            **kwargs,
        )
