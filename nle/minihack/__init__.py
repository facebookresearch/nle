# Copyright (c) Facebook, Inc. and its affiliates.
from gym.envs import registration

from nle.minihack.level_gen import LevelGenerator
from nle.minihack.base import MiniHack
from nle.minihack.navigation import MiniHackNavigation
from nle.minihack.skills import MiniHackSkill

__all__ = ["MiniHack", "MiniHackNavigation", "MiniHackSkill", "LevelGenerator"]

# Empty
registration.register(
    id="MiniHack-Empty-5x5-v0",
    entry_point="nle.minihack.navigation:MiniHackEmpty",
    kwargs={"size": 5, "random": False},
)
registration.register(
    id="MiniHack-Empty-Random-5x5-v0",
    entry_point="nle.minihack.navigation:MiniHackEmpty",
    kwargs={"size": 5, "random": True},
)
registration.register(
    id="MiniHack-Empty-10x10-v0",
    entry_point="nle.minihack.navigation:MiniHackEmpty",
    kwargs={"size": 10, "random": False},
)
registration.register(
    id="MiniHack-Empty-Random-10x10-v0",
    entry_point="nle.minihack.navigation:MiniHackEmpty",
    kwargs={"size": 10, "random": True},
)
registration.register(
    id="MiniHack-Empty-15x15-v0",
    entry_point="nle.minihack.navigation:MiniHackEmpty",
    kwargs={"size": 15, "random": False},
)
registration.register(
    id="MiniHack-Empty-Random-15x15-v0",
    entry_point="nle.minihack.navigation:MiniHackEmpty",
    kwargs={"size": 15, "random": True},
)

# Corridor
registration.register(
    id="MiniHack-Corridor-R2-v0",
    entry_point="nle.minihack.navigation:MiniHackCorridor",
    kwargs={"rooms": 2},
)
registration.register(
    id="MiniHack-Corridor-R3-v0",
    entry_point="nle.minihack.navigation:MiniHackCorridor",
    kwargs={"rooms": 3},
)
registration.register(
    id="MiniHack-Corridor-R5-v0",
    entry_point="nle.minihack.navigation:MiniHackCorridor",
    kwargs={"rooms": 5},
)
registration.register(
    id="MiniHack-Corridor-R8-v0",
    entry_point="nle.minihack.navigation:MiniHackCorridor",
    kwargs={"rooms": 8},
)
registration.register(
    id="MiniHack-Corridor-R10-v0",
    entry_point="nle.minihack.navigation:MiniHackCorridor",
    kwargs={"rooms": 10},
)

# KeyRoom
registration.register(
    id="MiniHack-KeyRoom-S5-3-v0",
    entry_point="nle.minihack.navigation:MiniHackKeyRoom",
    kwargs={"room_size": 5, "subroom_size": 3, "lit": True},
)
registration.register(
    id="MiniHack-KeyRoom-S12-4-v0",
    entry_point="nle.minihack.navigation:MiniHackKeyRoom",
    kwargs={"room_size": 12, "subroom_size": 4, "lit": True},
)
registration.register(
    id="MiniHack-KeyRoom-Unlit-S5-3-v0",
    entry_point="nle.minihack.navigation:MiniHackKeyRoom",
    kwargs={"room_size": 5, "subroom_size": 3, "lit": False},
)
registration.register(
    id="MiniHack-KeyRoom-Unlit-S12-4-v0",
    entry_point="nle.minihack.navigation:MiniHackKeyRoom",
    kwargs={"room_size": 12, "subroom_size": 4, "lit": False},
)

# Skill Tasks

registration.register(
    id="MiniHack-MazeWalk-v0",
    entry_point="nle.minihack.navigation:MiniHackMazeWalk",
)
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
# registration.register(
#     id="MiniHack-Quaff-v0",
#     entry_point="nle.minihack.skills:MiniHackQuaff",
# )
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

# MiniGrid: MultiRoom
registration.register(
    id="MiniHack-MultiRoom-N2-S4-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-MultiRoom-N2-S4-v0"},
)
registration.register(
    id="MiniHack-MultiRoom-N4-S5-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-MultiRoom-N4-S5-v0"},
)
registration.register(
    id="MiniHack-MultiRoom-N6-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-MultiRoom-N6-v0"},
)

# MiniGrid: LockedMultiRoom
registration.register(
    id="MiniHack-LockedMultiRoom-N2-S4-M1-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-MultiRoom-N2-S4-v0", "door_state": "locked"},
)
registration.register(
    id="MiniHack-LockedMultiRoom-N4-S5-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-MultiRoom-N4-S5-v0", "door_state": "locked"},
)
registration.register(
    id="MiniHack-LockedMultiRoom-N6-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-MultiRoom-N6-v0", "door_state": "locked"},
)

# MiniGrid: MonsterMultiRoom
registration.register(
    id="MiniHack-MonsterMultiRoom-N2-S4-M1-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-MultiRoom-N2-S4-v0", "num_mon": 1},
)
registration.register(
    id="MiniHack-MonsterMultiRoom-N4-S5-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-MultiRoom-N4-S5-v0", "num_mon": 4},
)
registration.register(
    id="MiniHack-MonsterMultiRoom-N6-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-MultiRoom-N6-v0", "num_mon": 6},
)

# MiniGrid: LavaCrossing
registration.register(
    id="MiniHack-LavaCrossingS9N1-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-LavaCrossingS9N1-v0"},
)
registration.register(
    id="MiniHack-LavaCrossingS9N2-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-LavaCrossingS9N2-v0"},
)
registration.register(
    id="MiniHack-LavaCrossingS9N3-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-LavaCrossingS9N3-v0"},
)
registration.register(
    id="MiniHack-LavaCrossingS11N5-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-LavaCrossingS11N5-v0"},
)

# MiniGrid: Simple Crossing
registration.register(
    id="MiniHack-SimpleCrossingS9N1-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-SimpleCrossingS9N1-v0"},
)
registration.register(
    id="MiniHack-SimpleCrossingS9N2-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-SimpleCrossingS9N2-v0"},
)
registration.register(
    id="MiniHack-SimpleCrossingS9N3-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-SimpleCrossingS9N3-v0"},
)
registration.register(
    id="MiniHack-SimpleCrossingS11N5-v0",
    entry_point="nle.minihack.minigrid:MiniGridHack",
    kwargs={"env_name": "MiniGrid-SimpleCrossingS11N5-v0"},
)
