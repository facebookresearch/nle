# Copyright (c) Facebook, Inc. and its affiliates.
from gym.envs import registration

from nle.env.base import NLE, DUNGEON_SHAPE

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
    id="NetHackChallenge-v0", entry_point="nle.env.tasks:NetHackChallenge"
)

__all__ = ["NLE", "DUNGEON_SHAPE"]
