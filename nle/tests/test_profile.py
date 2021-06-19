#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.

# Requires
#   pip install pytest-benchmark
# to run

import pytest

import numpy as np
import nle  # noqa: F401
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
    "(4): (3)... + tty_*": BASE_KEYS + MAPPED_GLYPH + INV_GLYPH + TTY,
    "(5): (4)... + screen_desc": BASE_KEYS
    + MAPPED_GLYPH
    + INV_GLYPH
    + TTY
    + SCREEN_DESC,
}


@pytest.mark.parametrize(
    "observation_keys", EXPERIMENTS.values(), ids=EXPERIMENTS.keys()
)
class TestProfile:
    @pytest.yield_fixture(autouse=True)  # will be applied to all tests in class
    def make_cwd_tmp(self, tmpdir):
        """Makes cwd point to the test's tmpdir."""
        with tmpdir.as_cwd():
            yield

    @pytest.mark.benchmark(disable_gc=True, warmup=False)
    def test_run_1k_steps(self, observation_keys, make_cwd_tmp, benchmark):
        env = gym.make("NetHack-v0", observation_keys=observation_keys)
        seeds = 123456
        steps = 1000

        np.random.seed(seeds)
        actions = np.random.choice(len(env._actions), size=steps)

        def seed():
            if not nle.nethack.NLE_ALLOW_SEEDING:
                return
            nonlocal seeds
            seeds += 1
            env.seed(seeds, 2 * seeds)

        def play_1k_steps():
            env.reset()
            for a in actions:
                _, _, done, _ = env.step(a)
                if done:
                    seed()
                    env.reset()

        benchmark.pedantic(play_1k_steps, setup=seed, rounds=100, warmup_rounds=10)
