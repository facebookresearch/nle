# Copyright (c) Facebook, Inc. and its affiliates.
"""Debugging script for the NLE dev team."""

import random
import timeit

from nle import nethack

ACTIONS = [nethack.MiscAction.MORE]
ACTIONS += list(nethack.CompassDirection)
ACTIONS += list(nethack.CompassDirectionLonger)


def play(game, print_tombstone=True):
    observation = game.reset()

    steps = 0

    while True:
        ch = random.choice(ACTIONS)
        last_observation = observation
        observation, done, unused_info = game.step(ch)
        steps += 1
        if steps > 10000:
            print("Game abandonned after", steps, "steps.")
            return steps
        if done:
            if last_observation.ProgramState().Gameover():
                # Print tombstone.
                if last_observation.WindowsLength() < 1:
                    return steps
                window = last_observation.Windows(1)

                if print_tombstone:
                    for i in range(window.StringsLength()):
                        print(window.Strings(i).decode("ascii"))
                    killer_name = last_observation.Internal().KillerName()
                    if killer_name:
                        print(killer_name.decode("utf-8"))

            return steps


def main():
    episodes = 0
    steps = 0

    game = nethack.NetHack(archivefile="random.zip")

    start = timeit.default_timer()
    while True:
        steps_delta = play(game, False)
        time_delta = timeit.default_timer() - start

        episodes += 1
        steps += steps_delta

        print(
            "Episde: %i. Steps: %i. SPS: %f"
            % (episodes, steps, steps_delta / time_delta)
        )
        start = timeit.default_timer()


if __name__ == "__main__":
    main()
