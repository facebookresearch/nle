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

    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)
        kwargs["actions"] = NAVIGATE_ACTIONS
        rooms = kwargs.pop("rooms", 2)
        assert rooms in [2, 3, 5, 8, 10]
        super().__init__(*args, des_file=f"corridor{rooms}.des", **kwargs)


# Corridor
registration.register(
    id="MiniHack-Corridor-R2-v0",
    entry_point="nle.minihack.envs.corridor:MiniHackCorridor",
    kwargs={"rooms": 2},
)
registration.register(
    id="MiniHack-Corridor-R3-v0",
    entry_point="nle.minihack.envs.corridor:MiniHackCorridor",
    kwargs={"rooms": 3},
)
registration.register(
    id="MiniHack-Corridor-R5-v0",
    entry_point="nle.minihack.envs.corridor:MiniHackCorridor",
    kwargs={"rooms": 5},
)
registration.register(
    id="MiniHack-Corridor-R8-v0",
    entry_point="nle.minihack.envs.corridor:MiniHackCorridor",
    kwargs={"rooms": 8},
)
registration.register(
    id="MiniHack-Corridor-R10-v0",
    entry_point="nle.minihack.envs.corridor:MiniHackCorridor",
    kwargs={"rooms": 10},
)
