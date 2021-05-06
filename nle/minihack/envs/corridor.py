from nle.minihack import MiniHackNavigation
from gym.envs import registration
from nle.nethack import Command
from nle import nethack

MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS = tuple(
    list(MOVE_ACTIONS) + [Command.OPEN, Command.KICK, Command.SEARCH]
)


class MiniHackCorridor(MiniHackNavigation):
    """Environment for "corridor" task.

    The agent has to navigate itself through randomely generated corridors that
    connect several rooms and find the goal.
    """

    def __init__(self, *args, des_file, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)
        kwargs["actions"] = NAVIGATE_ACTIONS
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackCorridor2(MiniHackCorridor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="corridor2.des", **kwargs)


class MiniHackCorridor3(MiniHackCorridor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="corridor3.des", **kwargs)


class MiniHackCorridor5(MiniHackCorridor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="corridor5.des", **kwargs)


class MiniHackCorridor8(MiniHackCorridor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="corridor8.des", **kwargs)


class MiniHackCorridor10(MiniHackCorridor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file="corridor10.des", **kwargs)


registration.register(
    id="MiniHack-Corridor-R2-v0",
    entry_point="nle.minihack.envs.corridor:MiniHackCorridor2",
)
registration.register(
    id="MiniHack-Corridor-R3-v0",
    entry_point="nle.minihack.envs.corridor:MiniHackCorridor3",
)
registration.register(
    id="MiniHack-Corridor-R5-v0",
    entry_point="nle.minihack.envs.corridor:MiniHackCorridor5",
)
registration.register(
    id="MiniHack-Corridor-R8-v0",
    entry_point="nle.minihack.envs.corridor:MiniHackCorridor8",
)
registration.register(
    id="MiniHack-Corridor-R10-v0",
    entry_point="nle.minihack.envs.corridor:MiniHackCorridor10",
)
