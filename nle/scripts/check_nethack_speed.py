# Copyright (c) Facebook, Inc. and its affiliates.
import multiprocessing as mp
import random
import sys
import time
import timeit
import traceback

from nle import nethack

ACTIONS = [nethack.MiscAction.MORE]
ACTIONS += list(nethack.CompassDirection)
ACTIONS += list(nethack.CompassDirectionLonger)


def target(i, should_stop, queue):
    print("Starting", i)
    try:
        play(should_stop, queue)
    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        traceback.print_exc()
        print()
        raise e


def play(should_stop, queue):
    game = nethack.Nethack()

    done = True
    steps = 0

    while not should_stop.is_set():
        if done or steps >= 1000:
            queue.put(steps)
            steps = 0
            observation = game.reset()

        ch = random.choice(ACTIONS)

        observation, done = game.step(ch)
        steps += 1


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    should_stop = ctx.Event()

    steps = 0

    processes = []
    for i in range(num_games):
        p = ctx.Process(target=target, args=(i, should_stop, queue))
        p.start()
        processes.append(p)

    try:
        while steps < 1000000:
            start_time = timeit.default_timer()
            start_steps = steps
            time.sleep(5)
            while not queue.empty():
                steps += queue.get()
            end_time = timeit.default_timer()

            print((steps - start_steps) / (end_time - start_time))

        should_stop.set()
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
