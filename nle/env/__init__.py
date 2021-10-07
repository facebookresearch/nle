# Copyright (c) Facebook, Inc. and its affiliates.
import gym
from gym.envs import registration

from nle.env.base import NLE, DUNGEON_SHAPE

_version = "v0"

for name in (
    "NetHack",
    "NetHackScore",
    "NetHackStaircase",
    "NetHackStaircasePet",
    "NetHackOracle",
    "NetHackGold",
    "NetHackEat",
    "NetHackScout",
    "NetHackChallenge",
):
    entry_point = "nle.env.tasks:" + name
    if name == "NetHack":
        entry_point = "nle.env.base:NLE"
    kwargs = {}
    if gym.__version__ >= "0.21":
        # Starting with version 0.21, gym wraps everything by the
        # OrderEnforcing wrapper by default (which isn't in gym.wrappers).
        # This breaks our seed() calls and some other code. Disable.
        kwargs["order_enforce"] = False
    registration.register(
        id="%s-%s" % (name, _version), entry_point=entry_point, **kwargs
    )


__all__ = ["NLE", "DUNGEON_SHAPE"]
