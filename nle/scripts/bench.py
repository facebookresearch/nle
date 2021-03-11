import timeit
import gym
from nle.env.mh_base import MiniHack  # noqa: F401

envs = [
    "MiniHack-Empty-v0",
    "MiniHack-LavaCrossing-v0",
    "MiniHack-Multiroom-N2-S4-v0",
    "MiniHack-Multiroom-N4-S5-v0",
    "MiniHack-Multiroom-N6-v0",
]
maxlen = max([len(el) for el in envs])
N_CALLS = 1000


def reset():
    env = gym.make(e)
    env.reset()


if __name__ == "__main__":
    for e in envs:
        t = timeit.timeit("reset()", setup="from __main__ import reset", number=N_CALLS)
        print(
            f"{e:<{maxlen}}: total: {t:.2f}s. {t/N_CALLS:.4f}s per init() and reset()."
        )
