# Copyright (c) Facebook, Inc. and its affiliates.
import enum
from nle.nethack import Command
from dataclasses import dataclass


class GoalEvent(enum.IntEnum):
    MESSAGE = 0
    LOC_ACTION = 1
    NAVIGATION = 2


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


@dataclass
class LocActionGoal:
    loc: str
    action: Command
    status: bool = False
    index: int = -1


class GoalGenerator:
    """This class is used for generating goals for MiniHack tasks."""

    def __init__(self):
        self.goals = []
        # TODO maybe add goal order

    def get_goals(self):
        return tuple(self.goals)

    def _add_message_goal(self, msgs):
        self.goals.append((GoalEvent.MESSAGE, msgs))

    def _add_loc_action_goal(self, loc, action):
        try:
            action = Command[action.upper()]
        except KeyError:
            raise KeyError(
                "Action {} is not in the action space.".format(action.upper())
            )

        goal = LocActionGoal(loc.lower(), action)
        self.goals.append((GoalEvent.LOC_ACTION, goal))

    def add_eat_goal(self, name):
        msgs = [f"This {name} is delicious"]
        if name == "apple":
            msgs.append("Delicious!  Must be a Macintosh!")
            msgs.append("Core dumped.")
        if name == "pear":
            msgs.append("Core dumped.")

        self._add_message_goal(msgs)

    def add_wield_goal(self, name):
        msgs = [f"{name} wields itself to your hand!", f"{name} (weapon in hand)"]
        self._add_message_goal(msgs)

    def add_wear_goal(self, name):
        msgs = [f"You are now wearing a {name}"]
        self._add_message_goal(msgs)

    def add_amulet_goal(self, name=None):
        self._add_message_goal(["amulet (being worn)."])

    def add_kill_goal(self, name):
        # TODO investigate
        self._add_message_goal([f"You kill the {name}"])

    def add_positional_goal(self, place_name, action_name):
        self._add_loc_action_goal(place_name, action_name)
