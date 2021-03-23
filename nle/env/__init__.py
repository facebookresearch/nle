# Copyright (c) Facebook, Inc. and its affiliates.
from gym.envs import registration

from nle.env.base import NLE, DUNGEON_SHAPE
from nle.env.mh_base import MiniHack

registration.register(id="NetHack-v0", entry_point="nle.env.base:NLE")

registration.register(id="NetHackScore-v0", entry_point="nle.env.tasks:NetHackScore")
registration.register(
    id="NetHackStaircase-v0", entry_point="nle.env.tasks:NetHackStaircase"
)
registration.register(
    id="NetHackStaircasePet-v0", entry_point="nle.env.tasks:NetHackStaircasePet"
)
registration.register(id="NetHackOracle-v0", entry_point="nle.env.tasks:NetHackOracle")
registration.register(id="NetHackGold-v0", entry_point="nle.env.tasks:NetHackGold")
registration.register(id="NetHackEat-v0", entry_point="nle.env.tasks:NetHackEat")
registration.register(id="NetHackScout-v0", entry_point="nle.env.tasks:NetHackScout")

registration.register(
    id="NetHackInventoryManagement-v0",
    entry_point="nle.env.tasks:NetHackInventoryManagement",
)

registration.register(
    id="NetHackPickAndEat-v0",
    entry_point="nle.env.tasks:NetHackPickAndEat",
)

registration.register(
    id="MiniHack-Empty-v0",
    entry_point="nle.env.mh_tasks:MiniHackEmpty",
)
registration.register(
    id="MiniHack-FourRooms-v0",
    entry_point="nle.env.mh_tasks:MiniHackFourRooms",
)
registration.register(
    id="MiniHack-Corridor-v0",
    entry_point="nle.env.mh_tasks:MiniHackCorridor",
)
registration.register(
    id="MiniHack-Multiroom-N2-S4-v0",
    entry_point="nle.env.mh_tasks:MiniGridHackMultiroom",
    kwargs={"env_name": "MiniGrid-MultiRoom-N2-S4-v0"},
)
registration.register(
    id="MiniHack-Multiroom-N4-S5-v0",
    entry_point="nle.env.mh_tasks:MiniGridHackMultiroom",
    kwargs={"env_name": "MiniGrid-MultiRoom-N4-S5-v0"},
)
registration.register(
    id="MiniHack-Multiroom-N6-v0",
    entry_point="nle.env.mh_tasks:MiniGridHackMultiroom",
    kwargs={"env_name": "MiniGrid-MultiRoom-N6-v0"},
)

registration.register(
    id="MiniHack-LavaCrossing-v0",
    entry_point="nle.env.mh_tasks:MiniHackLavaCrossing",
)
registration.register(
    id="MiniHack-SimpleCrossing-v0",
    entry_point="nle.env.mh_tasks:MiniHackSimpleCrossing",
)
registration.register(
    id="MiniHack-KeyDoor-v0",
    entry_point="nle.env.mh_tasks:MiniHackKeyDoor",
)
registration.register(
    id="MiniHack-MazeWalk-v0",
    entry_point="nle.env.mh_tasks:MiniHackMazeWalk",
)

registration.register(
    id="MiniHack-Eat-v0",
    entry_point="nle.env.mh_skills:MiniHackEat",
)
registration.register(
    id="MiniHack-Pray-v0",
    entry_point="nle.env.mh_skills:MiniHackPray",
)
registration.register(
    id="MiniHack-Sink-v0",
    entry_point="nle.env.mh_skills:MiniHackSink",
)
registration.register(
    id="MiniHack-Quaff-v0",
    entry_point="nle.env.mh_skills:MiniHackQuaff",
)
registration.register(
    id="MiniHack-ClosedDoor-v0",
    entry_point="nle.env.mh_skills:MiniHackClosedDoor",
)
registration.register(
    id="MiniHack-LockedDoor-v0",
    entry_point="nle.env.mh_skills:MiniHackLockedDoor",
)
registration.register(
    id="MiniHack-Wield-v0",
    entry_point="nle.env.mh_skills:MiniHackWield",
)
registration.register(
    id="MiniHack-Wear-v0",
    entry_point="nle.env.mh_skills:MiniHackWear",
)
registration.register(
    id="MiniHack-TakeOff-v0",
    entry_point="nle.env.mh_skills:MiniHackTakeOff",
)
registration.register(
    id="MiniHack-PutOn-v0",
    entry_point="nle.env.mh_skills:MiniHackPutOn",
)
registration.register(
    id="MiniHack-Zap-v0",
    entry_point="nle.env.mh_skills:MiniHackZap",
)
registration.register(
    id="MiniHack-Read-v0",
    entry_point="nle.env.mh_skills:MiniHackRead",
)


__all__ = ["NLE", "DUNGEON_SHAPE", "MiniHack"]
