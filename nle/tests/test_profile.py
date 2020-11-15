#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.

import pytest

import nle
from nle.nethack import OBSERVATION_DESC
import gym
import numpy as np


BASE_KEYS = ['glyphs', 'message', 'blstats']
MAPPED_GLYPH = ['chars', 'colors', 'specials']
INV_GLYPH = ['inv_glyphs', 'inv_strs', 'inv_letters', 'inv_oclasses']
SCREEN_DESC = ['screen_descriptions']

OBS_KEYS = [
    BASE_KEYS,
    BASE_KEYS + MAPPED_GLYPH,
    BASE_KEYS + MAPPED_GLYPH + INV_GLYPH,
    BASE_KEYS + MAPPED_GLYPH + INV_GLYPH + SCREEN_DESC
]

@pytest.mark.parametrize('observation_keys', OBS_KEYS)
@pytest.mark.benchmark(min_rounds=30, disable_gc=True, warmup=False)
def test_1k_steps_performance(observation_keys, benchmark):
    env = nle.env.base.NLE(observation_keys=observation_keys)

    steps = 1000
    actions = np.random.choice(len(env._actions), size=steps)

    def play_1k_steps():
        env.reset()
        for a in actions:
            _, _, done, _ = env.step(a)
            if done:
                env.reset()

    benchmark(play_1k_steps)
