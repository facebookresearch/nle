# Copyright (c) Facebook, Inc. and its affiliates.
import ctypes

import gym
import pytest

import nle
from nle.env import wrappers


class Mvital(ctypes.Structure):
    """From decl.h."""

    _fields_ = [
        ("born", ctypes.c_ubyte),
        ("died", ctypes.c_ubyte),
        ("mvflags", ctypes.c_ubyte),
    ]


Mvitals = Mvital * nle.nethack.NUMMONS


def get_monids(*names):
    names = set(names)
    result = {}

    for i in range(nle.nethack.NUMMONS):
        pm = nle.nethack.permonst(i)
        if pm.mname in names:
            result[pm.mname] = i
            names.remove(pm.mname)
            if not names:
                break
    else:
        raise ValueError("Couldn't find %s" % names)
    return result


class TestCTypesAPIWrapper:
    def test_nroom(self):
        env = gym.make("NetHack-v0", wizard=False)
        env = wrappers.CTypesAPIWrapper(env)

        nroom = ctypes.c_int.in_dll(env.get_dll(), "nroom")
        assert nroom.value == 0
        env.reset()
        assert nroom.value > 0
        del nroom
        env.reset()
        nroom = ctypes.c_int.in_dll(env.get_dll(), "nroom")
        assert nroom.value > 0

    def test_mvitals(self):
        env = wrappers.CTypesAPIWrapper(gym.make("NetHack-v0"))

        monids = get_monids("Angel", "minotaur")

        for _ in range(2):
            env.reset()
            mvitals = Mvitals.in_dll(env.get_dll(), "mvitals")
            assert mvitals[monids["Angel"]].mvflags == 16
            minotaurs = mvitals[monids["minotaur"]]
            assert minotaurs.born == minotaurs.died == 0
            del mvitals

    def test_extra_reference(self):
        env = wrappers.CTypesAPIWrapper(gym.make("NetHack-v0"))
        env.reset()
        dll = env.get_dll()
        with pytest.raises(RuntimeError, match="References.*reset"):
            env.reset()
        del dll
        env.reset()

    def test_same_reference(self):
        env = wrappers.CTypesAPIWrapper(gym.make("NetHack-v0"))
        env.reset()
        dll1 = env.get_dll()
        dll2 = env.get_dll()
        assert dll1 is dll2
