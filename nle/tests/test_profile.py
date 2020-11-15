#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.

import pytest

import nle
from nle.nethack import OBSERVATION_DESC
import gym
import numpy as np

ACTIONS = [nle.nethack.MiscAction.MORE]
ACTIONS += list(nle.nethack.CompassDirection)
ACTIONS += list(nle.nethack.CompassDirectionLonger)

@pytest.mark.benchmark(min_rounds=30, disable_gc=True, warmup=False)
def test_1k_steps_performance(benchmark):
    obs = [o for o in OBSERVATION_DESC.keys() if o not in ['screen_descriptions']]
    env = nle.env.base.NLE(observation_keys=obs  )

    steps = 1000
    actions = np.random.choice(len(env._actions), size=steps)

    def play_1k_steps():
        env.reset()
        for a in actions:
            _, _, done, _ = env.step(a)
            if done:
                env.reset()

    benchmark(play_1k_steps)
