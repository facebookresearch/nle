#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
import random
import sys

import numpy as np
import pytest
import gym

import nle
import nle.env


def get_nethack_env_ids():
    specs = gym.envs.registry.all()
    # Ignoring base environment, since we can't handle random actions yet with
    # the full action space, and this requires a whole different set of tests.
    # For now this is OK, since NetHackScore-v0 is very similar.
    return [
        spec.id
        for spec in specs
        if spec.id.startswith("NetHack") and spec.id != "NetHack-v0"
    ]


def rollout_env(env, max_rollout_len):
    """Produces a rollout and asserts step outputs.

    Does *not* assume that the environment has already been reset.
    """
    obs = env.reset()
    assert env.observation_space.contains(obs)

    step = 0
    while True:
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        if done or step >= max_rollout_len:
            break
    env.close()


def compare_rollouts(env0, env1, max_rollout_len):
    """Checks that two active environments return the same rollout.

    Assumes that the environments have already been reset.
    """
    step = 0
    while True:
        a = env0.action_space.sample()
        obs0, reward0, done0, info0 = env0.step(a)
        obs1, reward1, done1, info1 = env1.step(a)
        step += 1
        np.testing.assert_equal(obs0, obs1)
        assert reward0 == reward1
        assert done0 == done1

        # pid entries (and thus ttyrecs) won't match. Copy before removing.
        info0, info1 = info0.copy(), info1.copy()
        del info0["pid"], info1["pid"]

        if done0:
            assert "stats" in info0  # just to be sure
            assert "stats" in info1

            for k in ["ttyrec"]:
                assert info0["stats"][k] != info1["stats"][k]

                del info0["stats"][k]
                del info1["stats"][k]

        assert info0 == info1

        if done0 or step >= max_rollout_len:
            return


@pytest.mark.parametrize("env_name", get_nethack_env_ids())
class TestGymEnv:
    @pytest.yield_fixture(autouse=True)  # will be applied to all tests in class
    def make_cwd_tmp(self, tmpdir):
        """Makes cwd point to the test's tmpdir."""
        with tmpdir.as_cwd():
            yield

    def test_init(self, env_name):
        """Tests default initialization given standard env specs."""
        gym.make(env_name)

    def test_reset(self, env_name):
        """Tests default initialization given standard env specs."""
        env = gym.make(env_name)
        obs = env.reset()
        assert env.observation_space.contains(obs)


@pytest.mark.parametrize("env_name", get_nethack_env_ids())
@pytest.mark.parametrize("rollout_len", [100])
class TestGymEnvRollout:
    @pytest.yield_fixture(autouse=True)  # will be applied to all tests in class
    def make_cwd_tmp(self, tmpdir):
        """Makes cwd point to the test's tmpdir."""
        with tmpdir.as_cwd():
            yield

    def test_rollout(self, env_name, rollout_len):
        """Tests rollout_len steps (or until termination) of random policy."""
        env = gym.make(env_name)
        rollout_env(env, rollout_len)

    def test_rollout_no_archive(self, env_name, rollout_len):
        """Tests rollout_len steps (or until termination) of random policy."""
        env = gym.make(env_name, archivefile=None)
        assert env.savedir is None
        assert env.archivefile is None
        assert env._stats_file is None
        assert env._stats_logger is None
        rollout_env(env, rollout_len)

    def test_seed_interface_output(self, env_name, rollout_len):
        """Tests whether env.seed output can be reused correctly."""
        env0 = gym.make(env_name)
        env1 = gym.make(env_name)

        seed_list0 = env0.seed()
        env0.reset()

        seed_dict = nle.env.seed_list_to_dict(seed_list0)
        assert env0.get_seeds() == seed_dict

        seed_list1 = env1.seed(seed_dict)
        assert seed_list0 == seed_list1

    def test_seed_rollout_from_nethack(self, env_name, rollout_len):
        """Tests that two NetHack instances with same seeds return same obs."""

        env0 = gym.make(env_name)
        env1 = gym.make(env_name)

        obs0 = env0.reset()  # no env.seed() call, so uses NetHack's seeds
        seeds = env0.get_seeds()

        env1.seed(seeds)
        obs1 = env1.reset()

        del obs0["message"]  # because of different names
        del obs1["message"]
        np.testing.assert_equal(obs0, obs1)
        compare_rollouts(env0, env1, rollout_len)

    def test_seed_rollout_seeded(self, env_name, rollout_len):
        """Tests that two seeded envs return same step data."""
        env0 = gym.make(env_name)
        env1 = gym.make(env_name)

        env0.seed()
        obs0 = env0.reset()
        seeds0 = env0.get_seeds()

        env1.seed(seeds0)
        obs1 = env1.reset()
        seeds1 = env1.get_seeds()

        assert seeds0 == seeds1

        del obs0["message"]  # because of different names
        del obs1["message"]
        np.testing.assert_equal(obs0, obs1)
        compare_rollouts(env0, env1, rollout_len)

    def test_seed_rollout_seeded_int(self, env_name, rollout_len):
        """Tests that two seeded envs return same step data."""
        env0 = gym.make(env_name)
        env1 = gym.make(env_name)

        env0.seed(random.randrange(sys.maxsize))
        obs0 = env0.reset()
        seeds0 = env0.get_seeds()

        env1.seed(seeds0)
        obs1 = env1.reset()
        seeds1 = env1.get_seeds()

        assert seeds0 == seeds1

        del obs0["message"]  # because of different names
        del obs1["message"]
        np.testing.assert_equal(obs0, obs1)
        compare_rollouts(env0, env1, rollout_len)

    def test_render_ansi(self, env_name, rollout_len):
        env = gym.make(env_name)
        env.reset()
        for _ in range(rollout_len):
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            if done:
                env.reset()
            output = env.render(mode="ansi")
            assert isinstance(output, str)
            assert len(output.replace("\n", "")) == np.prod(nle.env.DUNGEON_SHAPE)
