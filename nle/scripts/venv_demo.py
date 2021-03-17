from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


# pip install stable_baselines3
import gym
from nle.env import NLE
import os
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="Vectorized environment demo.")

parser.add_argument(
    "--env", type=str, default="MiniHack-FourRooms-v0", help="MiniHack gym environment."
)
parser.add_argument("--num_env", type=int, default=4, help="Number of environments.")
parser.add_argument(
    "--num_steps", type=int, default=100000, help="Number of environments."
)
parser.add_argument(
    "--subproc", type=bool, default=False, help="Whether to use subprocesses or not"
)


class NLE_VecEnv_Wrapper:
    def __init__(self, env):
        self.env = env

    def step(self, action: int):
        os.chdir(self.env.env._vardir)
        return self.env.step(action)

    def reset(self):
        os.chdir(self.env.env._vardir)
        return self.env.reset()

    def close(self):
        os.chdir(self.env.env._vardir)
        self.env.close()

    def seed(self, core=None, disp=None, reseed=False):
        os.chdir(self.env.env._vardir)
        self.env.seed(core, disp, reseed)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)


def make_env(env_id, subproc, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        if not subproc:
            env = NLE_VecEnv_Wrapper(env)
        return env

    set_random_seed(seed)
    return _init


def make_venv(args):
    if not args.subproc:
        # Performs actions sequentially
        venv = DummyVecEnv(
            [make_env(args.env, args.subproc, i) for i in range(args.num_env)]
        )
    else:
        # Performs actions in parallel processes
        venv = SubprocVecEnv(
            [make_env(args.env, args.subproc, i) for i in range(args.num_env)]
        )

    return venv


def main(args):
    env = make_venv(args)
    env.reset()

    start_time = time.time()
    for i in range(args.num_steps):
        env.step([np.random.randint(8)] * args.num_env)
        if (i - 1) % 200 == 0:
            env.reset()
    total_time_multi = time.time() - start_time

    print(
        "Took {:.2f}s with subproc={} on {} steps- {:.2f} FPS".format(
            total_time_multi,
            args.subproc,
            args.num_steps,
            args.num_steps / total_time_multi,
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

    x = NLE()  # to make flake8 happy
