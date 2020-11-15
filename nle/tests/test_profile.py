#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.

import pytest

import nle
import numpy as np
import gym

BASE_KEYS = ["glyphs", "message", "blstats"]
MAPPED_GLYPH = ["chars", "colors", "specials"]
INV_GLYPH = ["inv_glyphs", "inv_strs", "inv_letters", "inv_oclasses"]
SCREEN_DESC = ["screen_descriptions"]
TTY = ["tty_chars", "tty_colors", "tty_cursor"]

EXPERIMENTS = {
    "(1): glyphs/msg/blstats": BASE_KEYS,
    "(2): (1)... + char/col/spec": BASE_KEYS + MAPPED_GLYPH,
    "(3): (2)... + inv_*": BASE_KEYS + MAPPED_GLYPH + INV_GLYPH,
    "(4): (3)... + screen_desc": BASE_KEYS + MAPPED_GLYPH + INV_GLYPH + SCREEN_DESC,
}


@pytest.mark.parametrize(
    "observation_keys", EXPERIMENTS.values(), ids=EXPERIMENTS.keys()
)
@pytest.mark.benchmark(disable_gc=True, warmup=False)
def test_run_1k_steps(observation_keys, benchmark):
    env = gym.make('NetHack-v0', savedir=None, observation_keys=observation_keys)
    seeds = [123456]
    steps = 1000

    np.random.seed(seeds[0])
    actions = np.random.choice(len(env._actions), size=steps)

    def seed():
        seeds[0] += 1
        env.seed(seeds[0], 2 * seeds[0])

    def play_1k_steps():
        env.reset()
        for a in actions:
            _, _, done, _ = env.step(a)
            if done:
                seed()
                env.reset()

    benchmark.pedantic(play_1k_steps, setup=seed, rounds=100, warmup_rounds=10)
