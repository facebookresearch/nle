import contextlib
import termios
import timeit
import tty
import random
import os

import numpy as np

from nle import pynle


SELF_PLAY = True


@contextlib.contextmanager
def no_echo():
    tt = termios.tcgetattr(0)
    try:
        tty.setraw(0)
        yield
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, tt)


def main():
    # MORE + compass directions + long compass directions.
    ACTIONS = [
        13,
        107,
        108,
        106,
        104,
        117,
        110,
        98,
        121,
        75,
        76,
        74,
        72,
        85,
        78,
        66,
        89,
    ]

    os.environ["NETHACKOPTIONS"] = "nolegacy,nocmdassist"

    nle = pynle.NLE(observation_keys=("chars", "blstats"))
    nle.reset()

    nle.step(ord("y"))
    nle.step(ord("y"))
    nle.step(ord("\n"))

    steps = 0
    start_time = timeit.default_timer()
    start_steps = steps

    mean_sps = 0
    sps_n = 0

    for episode in range(2):
        while True:
            ch = random.choice(ACTIONS)
            _, done = nle.step(ch)
            if done:
                break

            steps += 1

            if steps % 1000 == 0:
                end_time = timeit.default_timer()
                sps = (steps - start_steps) / (end_time - start_time)
                sps_n += 1
                mean_sps += (sps - mean_sps) / sps_n
                print("%f SPS" % sps)
                start_time = end_time
                start_steps = steps
        print("Finished episode %i after %i steps." % (episode + 1, steps))
        nle.reset()

    print("Finished after %i steps. Mean sps: %f" % (steps, mean_sps))

    if not SELF_PLAY:
        return

    done = False
    while not done:
        with no_echo():
            (chars, blstats), done = nle.step(ord(os.read(0, 1)))
        for line in chars:
            print(line.tobytes().decode("utf-8"))
        print(blstats)


main()
