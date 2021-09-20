import multiprocessing as mp
import random

import gym
import pytest

import nle  # noqa: F401

START_METHODS = [m for m in ("fork", "spawn") if m in mp.get_all_start_methods()]


def new_env_one_step():
    env = gym.make("NetHackScore-v0")
    env.reset()
    obs, reward, done, _ = env.step(0)
    return done


@pytest.mark.parametrize(
    "ctx", [mp.get_context(m) for m in START_METHODS], ids=START_METHODS
)
class TestEnvSubprocess:
    def test_env_in_subprocess(self, ctx):
        p = ctx.Process(target=new_env_one_step)
        p.start()
        p.join()
        assert p.exitcode == 0

    def test_env_before_and_in_subprocess(self, ctx):
        new_env_one_step()
        p = ctx.Process(target=new_env_one_step)
        p.start()
        p.join()
        assert p.exitcode == 0


ACTIONS = [0, 1, 2]


class TestParallelEnvs:
    def test_two_nles(self):
        envs = [gym.make("NetHackScore-v0") for _ in range(2)]

        env, *queue = envs
        env.reset()

        num_resets = 1

        while num_resets < 10:
            _, _, done, _ = env.step(random.choice(ACTIONS))
            if done:
                print("one env done")
                queue.append(env)
                env = queue.pop(0)
                print("about to reset one env")
                env.reset()
                num_resets += 1
