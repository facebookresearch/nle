# Copyright (c) Facebook, Inc. and its affiliates.
from nle.minihack import MiniHack, LevelGenerator, GoalGenerator
from nle.minihack.goal_generator import GoalEvent
from nle.nethack import CompassDirection
from gym.envs import registration
import numpy as np

Y_cmd = CompassDirection.NW


class MiniHackSkillEnv(MiniHack):
    """Base environment skill acquisition tasks."""

    def __init__(
        self,
        *args,
        des_file,
        goals=None,
        **kwargs,
    ):
        """If goal_msgs == None, the goal is to reach the staircase."""
        kwargs["options"] = kwargs.pop("options", [])
        kwargs["options"].append("pettype:none")
        kwargs["options"].append("!autopickup")
        kwargs["character"] = kwargs.pop("charachter", "cav-hum-new-mal")
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        self._no_rand_mon()

        self.goals_orig = goals
        self._init_goals()

        default_keys = [
            "chars_crop",
            "colors_crop",
            "screen_descriptions_crop",
            "message",
            "inv_strs",
            "inv_letters",
        ]

        kwargs["observation_keys"] = kwargs.pop("observation_keys", default_keys)
        super().__init__(*args, des_file=des_file, **kwargs)

    def _init_goals(self):
        if self.goals_orig is None:
            self.is_navigation_task = True
            self.goals_achieved = [False]
            self.goals = None
        else:
            self.goals = list(self.goals_orig)
            self.is_navigation_task = False
            self.goals_achieved = [False] * len(self.goals)

            self.loc_action_goals = []
            for i, (goal_type, goal) in enumerate(self.goals):
                if goal_type == GoalEvent.LOC_ACTION:
                    goal.index = i
                    self.loc_action_goals.append(goal)  # index, status

    def reset(self, *args, **kwargs):
        self._init_goals()
        return super().reset(*args, **kwargs)

    def step(self, action: int):
        if self.goals is not None:
            for goal in self.loc_action_goals:
                if self._actions[action] == goal.action and self._standing_on_top(
                    goal.loc
                ):
                    goal.status = True
                elif self._actions[action] == Y_cmd and goal.status:
                    self.goals_achieved[goal.index] = True
                else:
                    goal.status = False

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def _is_episode_end(self, observation):
        # IF the goal is to reach the staircase
        if self.is_navigation_task:
            internal = observation[self._internal_index]
            stairs_down = internal[4]
            if stairs_down:
                return self.StepStatus.TASK_SUCCESSFUL
            return self.StepStatus.RUNNING

        # Else, iterator through goals
        for i, (goal_type, details) in enumerate(self.goals):
            if self.goals_achieved[i]:
                # if goal already achieved, continue
                continue

            # If message goal
            if goal_type == GoalEvent.MESSAGE:
                goal_msgs = details
                curr_msg = (
                    observation[self._original_observation_keys.index("message")]
                    .tobytes()
                    .decode("utf-8")
                )
                for msg in goal_msgs:
                    if msg in curr_msg:
                        self.goals_achieved[i] = True

            # Loc-action goal checks are handled in step()

        if all(goal for goal in self.goals_achieved):
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


class MiniHackEat(MiniHackSkillEnv):
    """Environment for "eat" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("apple", "%")
        des_file = lvl_gen.get_des()

        goal_gen = GoalGenerator()
        goal_gen.add_eat_goal("apple")
        goals = goal_gen.get_goals()

        super().__init__(*args, des_file=des_file, goals=goals, **kwargs)


class MiniHackWield(MiniHackSkillEnv):
    """Environment for "wield" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("dagger", ")")
        des_file = lvl_gen.get_des()

        goal_gen = GoalGenerator()
        goal_gen.add_wield_goal("dagger")
        goals = goal_gen.get_goals()

        super().__init__(*args, des_file=des_file, goals=goals, **kwargs)


class MiniHackWear(MiniHackSkillEnv):
    """Environment for "wear" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("robe", "[")
        des_file = lvl_gen.get_des()

        goal_gen = GoalGenerator()
        goal_gen.add_wear_goal("robe")
        goals = goal_gen.get_goals()

        super().__init__(*args, des_file=des_file, goals=goals, **kwargs)


class MiniHackTakeOff(MiniHackSkillEnv):
    """Environment for "take off" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("leather jacket", "[")
        des_file = lvl_gen.get_des()

        goal_gen = GoalGenerator()
        goal_gen.add_wear_goal("leather jacket")
        goals = goal_gen.get_goals()

        super().__init__(*args, des_file=des_file, goals=goals, **kwargs)


class MiniHackPutOn(MiniHackSkillEnv):
    """Environment for "put on" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("amulet of life saving", '"')
        des_file = lvl_gen.get_des()

        goal_gen = GoalGenerator()
        goal_gen.add_amulet_goal()
        goals = goal_gen.get_goals()

        super().__init__(*args, des_file=des_file, goals=goals, **kwargs)


class MiniHackZap(MiniHackSkillEnv):
    """Environment for "zap" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("enlightenment", "/")
        des_file = lvl_gen.get_des()

        goal_gen = GoalGenerator()
        goal_gen._add_message_goal(["The feeling subsides."])  # TODO change
        goals = goal_gen.get_goals()

        super().__init__(*args, des_file=des_file, goals=goals, **kwargs)


class MiniHackRead(MiniHackSkillEnv):
    """Environment for "read" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("blank paper", "?")
        des_file = lvl_gen.get_des()

        goal_gen = GoalGenerator()
        goal_gen._add_message_goal(["This scroll seems to be blank."])  # TODO change
        goals = goal_gen.get_goals()

        super().__init__(*args, des_file=des_file, goals=goals, **kwargs)


class MiniHackPray(MiniHackSkillEnv):
    """Environment for "pray" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_altar("random", "neutral", "altar")
        des_file = lvl_gen.get_des()

        goal_gen = GoalGenerator()
        goal_gen.add_positional_goal("altar", "pray")
        goals = goal_gen.get_goals()

        super().__init__(*args, des_file=des_file, goals=goals, **kwargs)


class MiniHackSink(MiniHackSkillEnv):
    """Environment for "sink" task."""

    def __init__(self, *args, **kwargs):
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_sink()
        des_file = lvl_gen.get_des()

        goal_gen = GoalGenerator()
        goal_gen.add_positional_goal("sink", "quaff")
        goals = goal_gen.get_goals()

        super().__init__(*args, des_file=des_file, goals=goals, **kwargs)


class MiniHackClosedDoor(MiniHackSkillEnv):
    """Environment for "open" task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="closed_door.des", **kwargs)


class MiniHackLockedDoor(MiniHackSkillEnv):
    """Environment for "kick" task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="locked_door.des", **kwargs)


class MiniHackWandOfDeath(MiniHackSkillEnv):
    """Environment for "sink" task."""

    def __init__(self, *args, **kwargs):
        map = """
-------------
|...........|
|...........|
|...........|
|...........|
|....|.|....|
|....|.|....|
|-----.-----|
|...........|
|...........|
|...........|
-------------
"""
        lvl_gen = LevelGenerator(map=map, lit=True)

        def get_safe_coord():
            return np.random.randint(1, 11), np.random.randint(1, 5)

        def get_dangerous_coord():
            return np.random.randint(1, 11), np.random.randint(8, 10)

        lvl_gen.add_object("death", "/", cursestate="blessed", place=get_safe_coord())
        lvl_gen.add_stair_up(get_safe_coord())
        lvl_gen.add_monster("minotaur", args=("asleep",), place=(6, 8))
        lvl_gen.add_stair_down(get_dangerous_coord())
        des_file = lvl_gen.get_des()

        super().__init__(*args, des_file=des_file, **kwargs)


registration.register(
    id="MiniHack-WandOfDeath-v0",
    entry_point="nle.minihack.skills:MiniHackWandOfDeath",
)


class MiniHackLabyrinth(MiniHackSkillEnv):
    """Environment for "read" task."""

    def __init__(self, *args, **kwargs):
        map = """
-------------------------------------
|.................|.|...............|
|.|-------------|.|.|.------------|.|
|.|.............|.|.|.............|.|
|.|.|----------.|.|.|------------.|.|
|.|.|...........|.|.............|.|.|
|.|.|.|----------.|-----------|.|.|.|
|.|.|.|...........|.......|...|.|.|.|
|.|.|.|.|----------------.|.|.|.|.|.|
|.|.|.|.|.................|.|.|.|.|.|
|.|.|.|.|.-----------------.|.|.|.|.|
|.|.|.|.|...................|.|.|.|.|
|.|.|.|.|--------------------.|.|.|.|
|.|.|.|.......................|.|.|.|
|.|.|.|-----------------------|.|.|.|
|.|.|...........................|.|.|
|.|.|---------------------------|.|.|
|.|...............................|.|
|.|-------------------------------|.|
|...................................|
-------------------------------------
"""
        lvl_gen = LevelGenerator(map=map, lit=True)
        lvl_gen.add_stair_up((19, 1))
        lvl_gen.add_stair_down((19, 7))
        lvl_gen.add_monster(name="minotaur", place=(19, 9))
        lvl_gen.add_object("death", "/", cursestate="blessed")
        des_file = lvl_gen.get_des()

        super().__init__(
            *args,
            des_file=des_file,
            **kwargs,
        )


registration.register(
    id="MiniHack-Labyrinth-v0",
    entry_point="nle.minihack.skills:MiniHackLabyrinth",
)


# Skill Tasks
registration.register(
    id="MiniHack-Eat-v0",
    entry_point="nle.minihack.skills:MiniHackEat",
)
registration.register(
    id="MiniHack-Pray-v0",
    entry_point="nle.minihack.skills:MiniHackPray",
)
registration.register(
    id="MiniHack-Sink-v0",
    entry_point="nle.minihack.skills:MiniHackSink",
)
registration.register(
    id="MiniHack-ClosedDoor-v0",
    entry_point="nle.minihack.skills:MiniHackClosedDoor",
)
registration.register(
    id="MiniHack-LockedDoor-v0",
    entry_point="nle.minihack.skills:MiniHackLockedDoor",
)
registration.register(
    id="MiniHack-Wield-v0",
    entry_point="nle.minihack.skills:MiniHackWield",
)
registration.register(
    id="MiniHack-Wear-v0",
    entry_point="nle.minihack.skills:MiniHackWear",
)
registration.register(
    id="MiniHack-TakeOff-v0",
    entry_point="nle.minihack.skills:MiniHackTakeOff",
)
registration.register(
    id="MiniHack-PutOn-v0",
    entry_point="nle.minihack.skills:MiniHackPutOn",
)
registration.register(
    id="MiniHack-Zap-v0",
    entry_point="nle.minihack.skills:MiniHackZap",
)
registration.register(
    id="MiniHack-Read-v0",
    entry_point="nle.minihack.skills:MiniHackRead",
)
