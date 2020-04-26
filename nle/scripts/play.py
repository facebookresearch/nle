#!/usr/bin/env python

import argparse
import ast
import contextlib
import pprint
import random
import os
import termios
import timeit
import tty

import gym

import nle  # noqa: F401
from nle import nethack
from nle.nethack import print_message


@contextlib.contextmanager
def dummy_context():
    yield None


@contextlib.contextmanager
def no_echo():
    tt = termios.tcgetattr(0)
    try:
        tty.setraw(0)
        yield
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, tt)


def get_action(env, action_mode, is_raw_env):
    if action_mode == "random":
        if not is_raw_env:
            action = env.action_space.sample()
        else:
            action = random.choice(env._actions)
    elif action_mode == "human":
        while True:
            with no_echo():
                ch = ord(os.read(0, 1))
            if ch in [nethack.C("c"), ord(b"q")]:
                print("Received exit code {}. Aborting.".format(ch))
                return None
            try:
                action = env._actions.index(ch)
                break
            except ValueError:
                print(
                    ("Selected action '%s' is not in action list. Please try again.")
                    % chr(ch)
                )
                continue
    return action


def play(env_name, play_mode, ngames, max_steps, seeds, no_clear, no_render):
    del max_steps  # TODO
    is_raw_env = env_name == "nethack"

    if is_raw_env:
        # TODO save data somewhere reasonable
        env = nethack.NetHack(archivefile="./nle_data/play_data.zip")
    else:
        env = gym.make(env_name)
        if seeds is not None:
            env.seed(seeds)

    obs = env.reset()
    last_obs = None  # needed for "nethack" env

    steps = 0
    episodes = 0
    reward = 0.0
    action = None
    ch = None

    start_time = timeit.default_timer()
    while True:
        if not no_render:
            if not no_clear:
                os.system("cls" if os.name == "nt" else "clear")

            if not is_raw_env:
                print("Previous reward:", reward)
                print("Available actions:", env._actions)
                print(
                    "Previous action: {}{!r})".format(
                        "{} --".format(chr(ch)) if ch is not None else "",
                        env._actions[action] if action is not None else None,
                    )
                )
                env.render()
            else:
                print("Available actions:", env._actions)
                print("Previous actions:", action)
                print_message.print_message(obs)

        action = get_action(env, play_mode, is_raw_env)
        if action is None:
            break

        if is_raw_env:
            last_obs = obs
            obs, done, info = env.step(action)
        else:
            obs, reward, done, info = env.step(action)
        steps += 1

        if not done:
            continue

        if not is_raw_env:
            print("Final reward:", reward)
            print("End status:", info["end_status"].name)
            print("Env stats:")
            pprint.pprint(info["stats"])
        else:
            if last_obs.ProgramState().Gameover():
                # Print tombstone.
                if last_obs.WindowsLength() < 1:
                    return steps
                window = last_obs.Windows(1)

                if not no_render:
                    for i in range(window.StringsLength()):
                        print(window.Strings(i).decode("ascii"))

        time_delta = timeit.default_timer() - start_time
        print(
            "Episode: {}. Steps: {}. SPS: {:f}".format(
                episodes, steps, steps / time_delta
            )
        )
        start_time = timeit.default_timer()

        episodes += 1
        steps = 0
        if episodes == ngames:
            break
        env.reset()


def main():
    parser = argparse.ArgumentParser(description="NLE Play tool.")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enables debug mode, which will drop stack into "
        "an ipdb shell if an exception is raised.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="human",
        choices=["human", "random"],
        help="Control mode.",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="NetHackStaircase-v0",
        help="Gym environment spec.",
    )
    parser.add_argument(
        "-n",
        "--ngames",
        type=int,
        default=1,
        help="Number of games to be played before exiting. "
        "NetHack will auto-restart if > 1.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Number of maximum steps per episode.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Seeds to send to NetHack. Can be a dict or int. "
        "Defaults to None (no seeding).",
    )

    parser.add_argument("--no-clear", action="store_true", help="Disables tty clears.")
    parser.add_argument(
        "--no-render", action="store_true", help="Disables env.render()."
    )
    flags = parser.parse_args()

    if flags.debug:
        import ipdb

        cm = ipdb.launch_ipdb_on_exception
    else:
        cm = dummy_context

    with cm():
        if flags.seeds is not None:
            # to handle both int and dicts
            flags.seeds = ast.literal_eval(flags.seeds)

        play(
            flags.env,
            flags.mode,
            flags.ngames,
            flags.max_steps,
            flags.seeds,
            flags.no_clear,
            flags.no_render,
        )


if __name__ == "__main__":
    main()
