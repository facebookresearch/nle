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


__all__ = ["NLE", "DUNGEON_SHAPE", "MiniHack"]
