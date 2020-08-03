import contextlib
import termios
import time
import timeit
import tty
import random
import os

import pynle


SELF_PLAY = False


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

    nle = pynle.NLE()

    nle.step(ord("y"))
    nle.step(ord("y"))
    nle.step(ord("\n"))

    steps = 0
    start_time = timeit.default_timer()
    start_steps = steps

    mean_sps = 0
    sps_n = 0

    for episode in range(10):
        while not nle.done():
            ch = random.choice(ACTIONS)
            nle.step(ch)

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

    while not nle.done():
        obs = nle.observation()
        obs = obs.reshape((21, 79))
        for line in obs:
            print(line.tobytes().decode("utf-8"))
        with no_echo():
            nle.step(ord(os.read(0, 1)))


main()
