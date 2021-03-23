# Copyright (c) Facebook, Inc. and its affiliates.
from gym.envs import registration

from nle.minihack.base import MiniHack
from nle.minihack.level_gen import LevelGenerator

registration.register(
    id="MiniHack-Empty-v0",
    entry_point="nle.minihack.tasks:MiniHackEmpty",
)
registration.register(
    id="MiniHack-FourRooms-v0",
    entry_point="nle.minihack.tasks:MiniHackFourRooms",
)
registration.register(
    id="MiniHack-Corridor-v0",
    entry_point="nle.minihack.tasks:MiniHackCorridor",
)
registration.register(
    id="MiniHack-Multiroom-N2-S4-v0",
    entry_point="nle.minihack.tasks:MiniGridHackMultiroom",
    kwargs={"env_name": "MiniGrid-MultiRoom-N2-S4-v0"},
)
registration.register(
    id="MiniHack-Multiroom-N4-S5-v0",
    entry_point="nle.minihack.tasks:MiniGridHackMultiroom",
    kwargs={"env_name": "MiniGrid-MultiRoom-N4-S5-v0"},
)
registration.register(
    id="MiniHack-Multiroom-N6-v0",
    entry_point="nle.minihack.tasks:MiniGridHackMultiroom",
    kwargs={"env_name": "MiniGrid-MultiRoom-N6-v0"},
)

registration.register(
    id="MiniHack-LavaCrossing-v0",
    entry_point="nle.minihack.tasks:MiniHackLavaCrossing",
)
registration.register(
    id="MiniHack-SimpleCrossing-v0",
    entry_point="nle.minihack.tasks:MiniHackSimpleCrossing",
)
registration.register(
    id="MiniHack-KeyDoor-v0",
    entry_point="nle.minihack.tasks:MiniHackKeyDoor",
)
registration.register(
    id="MiniHack-MazeWalk-v0",
    entry_point="nle.minihack.tasks:MiniHackMazeWalk",
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


__all__ = ["MiniHack", "LevelGenerator"]
