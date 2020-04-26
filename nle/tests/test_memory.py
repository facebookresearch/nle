# Copyright (c) Facebook, Inc. and its affiliates.
import gc
import weakref

import gym

import nle  # noqa: F401


def get_ref_objects(exclude=None):
    result = {}
    for o in gc.get_objects():
        if o is exclude:
            continue
        try:
            result[id(o)] = weakref.ref(o)
        except TypeError:
            pass
    return result


def play(env):
    env.reset()
    for _ in range(50):
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            break


class TestObjectLifetime:
    def test_restart(self):
        env = gym.make("NetHack-v0")
        play(env)
        play(env)

    def test_object_allocation(self):
        env = gym.make("NetHack-v0")
        play(env)

        before = get_ref_objects()

        for _ in range(10):
            play(env)

        after = get_ref_objects(before)
        assert len(before) == len(after)
